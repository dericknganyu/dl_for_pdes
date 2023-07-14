import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import torchphysics as tp 
import pytorch_lightning as pl
import numpy as np
import torch                                                                                                                                                                    

### Code to compute the inverse Darcy problem for PINN and QRES.
### We compute the average of multiple runs, to get the average perfomance of these methods.
average_exmaples = 100

### Define problem domain, scaling parameters and discrete coordinates
### Stays the same for all runs:
X = tp.spaces.R2('x')
U = tp.spaces.R1('u')
A = tp.spaces.R1('a')

scaling = 100
noise = 0.05 # Parameter for nosie 
steps = 8 # Resolution of point grid that will be used
xmin, xmax = 0, 1
ymin, ymax = 0, 1
Nx, Ny = 512, 512 
xx = np.linspace(xmin, xmax, Nx+1)
xx = np.array(np.meshgrid(xx, xx)).T
x_points = xx[::steps, ::steps].reshape(-1, 2).astype(np.float32)
xx = xx.reshape(-1, 2).astype(np.float32)
xx = tp.spaces.Points(torch.tensor(xx), X)
x_points = tp.spaces.Points(torch.tensor(x_points), X)

### Define PDE
pde_sampler = tp.samplers.DataSampler(x_points)
def pde_residual(a, u, x):
    return tp.utils.div(a * tp.utils.grad(u, x), x) + 1.0

### Load data and scale the input.
### Load data and scale
#path = 'darcy/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5'
#hf = h5py.File(path, "r")
#output_data = hf['test_in'][:average_exmaples, :, :]
#input_data = hf['test_out'][:average_exmaples, :, :]
#hf.close()
current_path = os.path.dirname(os.path.abspath(__file__))
output_data = np.load(current_path + "/f_data.npy")[:average_exmaples]
input_data = np.load(current_path + "/u_data.npy")[:average_exmaples, ::steps, ::steps]
input_data = scaling * torch.tensor(input_data).reshape(average_exmaples, 65*65) ### 65*65, 128*128, 257*257, 513*513
output_data = torch.tensor(output_data)

### Define array to store error of each run
l2_error_array = np.zeros(average_exmaples)

for i in range(average_exmaples):
    print("current step:", i)
    ### Define model (or reset for next example)
    #model_u = tp.models.FCN(input_space=X, output_space=U, hidden=(50, 50, 20))
    model_u = tp.models.QRES(input_space=X, output_space=U, hidden=(36, 36, 15))
    #model_a = tp.models.FCN(input_space=X, output_space=A, hidden=(30, 40, 40, 20, 20))
    model_a = tp.models.QRES(input_space=X, output_space=A, hidden=(25, 35, 25, 15, 15))
    parallel_model = tp.models.Parallel(model_u, model_a)
    
    ### Set current data
    data = input_data[i].unsqueeze(-1)
    data += torch.max(data) * noise * torch.rand_like(data)
    data = tp.spaces.Points(data,  U)
    
    ### Define conditions
    pde_cond = tp.conditions.PINNCondition(module=parallel_model, sampler=pde_sampler, 
                                           residual_fn=pde_residual)
    data_loader = tp.utils.PointsDataLoader((x_points.as_tensor, data), batch_size=50000)
    data_condition = tp.conditions.DataCondition(module=model_u,
                                                dataloader=data_loader,
                                                norm=2, 
                                                weight=100) 
    
    ### Start training
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.005)
    solver = tp.solver.Solver(train_conditions=[pde_cond, data_condition], optimizer_setting=optim)
    trainer = pl.Trainer(gpus=1, max_steps=5000, logger=False, benchmark=True,
                         checkpoint_callback=False)         
    trainer.fit(solver)
   
    ### Switch to LBFGS
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.LBFGS, lr=0.5, 
                                optimizer_args={'max_iter': 2, 'history_size': 100})
    data_condition.use_full_dataset = True
    solver = tp.solver.Solver(train_conditions=[pde_cond, data_condition], optimizer_setting=optim)
    trainer = pl.Trainer(gpus=1, max_steps=15000, logger=False, benchmark=True,
                         checkpoint_callback=False)         
    trainer.fit(solver)

    ### Compute current error:
    model_a.to('cpu')
    a_out = model_a(xx)
    a_out = (a_out.as_tensor * scaling).detach().cpu().reshape(1, Nx+1, Nx+1)
    diff = np.abs(a_out[0] - output_data[i])
    l2_rel = torch.sqrt(torch.sum(diff**2))/torch.sqrt(torch.sum(output_data[i]**2))
    l2_error_array[i] = l2_rel
    print("In step ", i, "had error:", l2_rel)
    np.save(current_path + "/inv_l2_error_noise" + str(noise) + ".npy", l2_error_array)

    ### Save some solution:
    #if i == 0:
    #    np.save(current_path + "first_sol_noise" + str(noise) + ".npy", f_out)

print("Average error:", np.sum(l2_error_array)/len(l2_error_array))