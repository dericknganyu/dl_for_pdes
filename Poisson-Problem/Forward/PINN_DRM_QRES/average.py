import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import torchphysics as tp 
import pytorch_lightning as pl
import numpy as np
import torch
#import h5py

### Code to compute the forward Poisson problem for DRM, PINN and QRES.
### We compute the average of multiple runs, to get the average perfomance of these methods.
average_exmaples = 100

### Define problem domain, scaling parameters and discrete coordinates
### Stays the same for all runs:
X = tp.spaces.R2('x') # input is 2D
U = tp.spaces.R1('u') # output is 1D
square = tp.domains.Parallelogram(X, [0, 0], [1, 0], [0, 1])

scaling = 100
steps = 8 # resolution of point grid that will be used
xmin, xmax = 0, 1
ymin, ymax = 0, 1
Nx, Ny = 512, 512 
xx = np.linspace(xmin, xmax, Nx+1)
xx = np.array(np.meshgrid(xx, xx)).T
x_points = xx[::steps, ::steps].reshape(-1, 2).astype(np.float32)
xx = tp.spaces.Points(torch.tensor(xx.reshape(-1, 2).astype(np.float32)), X)
x_points = tp.spaces.Points(torch.tensor(x_points), X)
physic_sampler = tp.samplers.DataSampler(x_points)

f = torch.zeros(1) # rhs dummy

### Boundary conditions for DRM and PINN/QRES:
def bound_residual(u):
    return u**2
def bound_residual_pinn(u):
    return u

bound_sampler = tp.samplers.GridSampler(square.boundary, n_points=int(2052/steps)).make_static()

### PDE 
def energy_residual(u, x):
    return 1/2 * torch.sum(tp.utils.grad(u, x)**2, dim=1, keepdim=True) - f * u

def pde_residual(u, x):
    return tp.utils.laplacian(u, x) + f

### Load data and scale the rhs.
#path = '/poisson/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5'
#hf = h5py.File(path, "r")
#input_data = hf['test_in'][:average_exmaples, :, :]
#output_data = hf['test_out'][:average_exmaples, :, :]
#hf.close()
current_path = os.path.dirname(os.path.abspath(__file__))
input_data = np.load(current_path + "/f_data.npy")[:average_exmaples, ::steps, ::steps]
output_data = np.load(current_path + "/u_data.npy")[:average_exmaples]
input_data = torch.tensor(input_data).reshape(average_exmaples, 65*65) # considered shapes: 65*65, 128*128, 257*257, 513*513
input_data *= scaling
output_data = torch.tensor(output_data) # The expected output will be compared on the whole domain

### Define array to store error of each run
l2_error_array = np.zeros(average_exmaples)

for i in range(average_exmaples):
    print("current step:", i)
    ### Define model (or reset for next example)
    #model = tp.models.DeepRitzNet(input_space=X, output_space=U, depth=2, width=40)
    #model = tp.models.FCN(input_space=X, output_space=U, hidden=(50, 50, 50))
    model = tp.models.QRES(input_space=X, output_space=U, hidden=(36,36,36))
    
    ### Set current data
    f = input_data[i].unsqueeze(-1)
    f = f.to('cuda')
    
    ### Define training conditions
    ### DRM:
    #bound_cond = tp.conditions.DeepRitzCondition(module=model, sampler=bound_sampler, 
    #                                             integrand_fn=bound_residual, weight=10000)
    #pde_cond = tp.conditions.DeepRitzCondition(model, energy_sampler, energy_residual)
    ### PINN/QRES:
    bound_cond = tp.conditions.PINNCondition(module=model, sampler=bound_sampler, 
                                             residual_fn=bound_residual_pinn, weight=10000)
    pde_cond = tp.conditions.PINNCondition(module=model, sampler=physic_sampler, residual_fn=pde_residual)
   
    ### Start training
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)
    solver = tp.solver.Solver(train_conditions=[pde_cond, bound_cond], optimizer_setting=optim)
    trainer = pl.Trainer(gpus=1, max_steps=5000, logger=False, benchmark=True, checkpoint_callback=False)       
    trainer.fit(solver)
    
    ### Switch to LBFGS
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.LBFGS, lr=0.2, 
                                optimizer_args={'max_iter': 2, 'history_size': 100})
    solver = tp.solver.Solver(train_conditions=[pde_cond, bound_cond], optimizer_setting=optim)
    trainer = pl.Trainer(gpus=1, max_steps=8000, logger=False, benchmark=True,
                         checkpoint_callback=False) 
    trainer.fit(solver)

    ### Compute current error, using all domain points:
    model.to('cpu')
    u_out = model(xx)
    u_out = (u_out.as_tensor / scaling).detach().cpu().reshape(1, Nx+1, Nx+1)
    diff = np.abs(u_out[0] - output_data[i])
    l2_rel = torch.sqrt(torch.sum(diff**2))/torch.sqrt(torch.sum(output_data[i]**2))
    l2_error_array[i] = l2_rel
    print("In step ", i, "had error:", l2_rel)
    np.save(current_path + "/l2_error.npy", l2_error_array)

    ### Save some solution:
    #if i == 0:
    #    np.save(current_path + "first_sol.npy", u_out)

print("Average error:", np.sum(l2_error_array)/len(l2_error_array))