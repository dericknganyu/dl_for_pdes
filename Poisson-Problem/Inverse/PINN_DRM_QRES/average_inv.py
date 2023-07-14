import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
import torchphysics as tp 
import pytorch_lightning as pl
import numpy as np
import torch
#import h5py

### Code to compute the inverse Poisson problem for PINN and QRES.
### We compute the average of multiple runs, to get the average perfomance of these methods.
average_exmaples = 100

### Define problem domain, scaling parameters and discrete coordinates
### Stays the same for all runs:
X = tp.spaces.R2('x') # input is 2D
U = tp.spaces.R1('u') # output for the interpolating network
F = tp.spaces.R1('f') # output for the f-network

scaling = 100
noise = 0.0 # Parameter for nosie 
steps = 8 # Resolution of point grid that will be used
xmin, xmax = 0, 1
ymin, ymax = 0, 1
Nx, Ny = 512, 512 
xx = np.linspace(xmin, xmax, Nx+1)
xx = np.array(np.meshgrid(xx, xx)).T
x_points = xx[::steps, ::steps].reshape(-1, 2).astype(np.float32)
xx = tp.spaces.Points(torch.tensor(xx.reshape(-1, 2).astype(np.float32)), X)
x_points = tp.spaces.Points(torch.tensor(x_points), X)

### PDE:
pde_sampler = tp.samplers.DataSampler(x_points)

def pde_residual(f, u, x):
    return tp.utils.laplacian(u, x) + f

### Load data and scale the input.
#path = 'poisson/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5'
#hf = h5py.File(path, "r")
#output_data = hf['test_in'][:average_exmaples, :, :]
#input_data = hf['test_out'][:average_exmaples, :, :]
#hf.close()
current_path = os.path.dirname(os.path.abspath(__file__))
output_data = np.load(current_path + "/f_data.npy")[:average_exmaples]
input_data = np.load(current_path + "/u_data.npy")[:average_exmaples, ::steps, ::steps]
### shapes:
### 65*65, 128*128, 257*257, 513*513
input_data = torch.tensor(input_data).reshape(average_exmaples, 65*65)
input_data *= scaling
output_data = torch.tensor(output_data)

### Define array to store error of each run
l2_error_array = np.zeros(average_exmaples)

for i in range(average_exmaples):
    print("current step:", i)
    ### Define model (or reset for next example)
    #model_u = tp.models.FCN(input_space=X, output_space=U, hidden=(50, 50, 50))
    model_u = tp.models.QRES(input_space=X, output_space=U, hidden=(36, 36, 36))
    #model_f = tp.models.FCN(input_space=X, output_space=F, hidden=(50, 50, 50))
    model_f = tp.models.QRES(input_space=X, output_space=F, hidden=(36, 36, 36))
    parallel_model = tp.models.Parallel(model_u, model_f)
    
    ### Set current data (and add noise)
    data = tp.spaces.Points(input_data[i].unsqueeze(-1), U)
    data += (noise * torch.max(data)) * torch.randn_like(data)
    ### Define conditions
    pde_cond = tp.conditions.PINNCondition(module=parallel_model, sampler=pde_sampler, 
                                           residual_fn=pde_residual)
    data_loader = tp.utils.PointsDataLoader((x_points, data), batch_size=50000)
    data_condition = tp.conditions.DataCondition(module=model_u,
                                                dataloader=data_loader,
                                                norm=2,  
                                                weight=10000) 
    
    ### Start training
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)
    solver = tp.solver.Solver(train_conditions=[pde_cond, data_condition], optimizer_setting=optim)
    trainer = pl.Trainer(gpus=1, max_steps=5000, logger=False, benchmark=True,
                         checkpoint_callback=False)         
    trainer.fit(solver)
    
    ### LBFGS
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.LBFGS, lr=0.2, 
                                optimizer_args={'max_iter': 2, 'history_size': 100})
    data_condition.use_full_dataset = True
    solver = tp.solver.Solver(train_conditions=[pde_cond, data_condition], optimizer_setting=optim)
    trainer = pl.Trainer(gpus=1, max_steps=10000, logger=False, benchmark=True,
                         checkpoint_callback=False)         
    trainer.fit(solver)

    ### Compute current error:
    model_f.to('cpu')
    f_out = model_f(xx)
    f_out = (f_out.as_tensor / scaling).detach().cpu().reshape(1, Nx+1, Nx+1)
    diff = np.abs(f_out[0] - output_data[i])
    l2_rel = torch.sqrt(torch.sum(diff**2))/torch.sqrt(torch.sum(output_data[i]**2))
    l2_error_array[i] = l2_rel
    print("In step ", i, "had error:", l2_rel)
    np.save(current_path + "/inv_l2_error_noise" + str(noise) + ".npy", l2_error_array)

    ### Save some solution:
    #if i == 0:
    #    np.save(current_path + "first_sol_noise" + str(noise) + ".npy", f_out)

print("Average error:", np.sum(l2_error_array)/len(l2_error_array))