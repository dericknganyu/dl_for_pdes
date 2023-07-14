import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import torchphysics as tp 
import pytorch_lightning as pl
import numpy as np
import torch
import scipy.signal
#import h5py

### Code to compute the forward Darcy problem for DRM, PINN and QRES.
### We compute the average of multiple runs, to get the average perfomance of these methods.
average_exmaples = 100

### Define problem domain, scaling parameters and discrete coordinates
### Stays the same for all runs:
X = tp.spaces.R2('x') # input is 2D
U = tp.spaces.R1('u') # output is 1D
square = tp.domains.Parallelogram(X, [0, 0], [1, 0], [0, 1])

scaling = 0.01
steps = 1
xmin, xmax = 0, 1
ymin, ymax = 0, 1
Nx, Ny = 512, 512 
xx = np.linspace(xmin, xmax, Nx+1)
xx = np.array(np.meshgrid(xx, xx)).T
x_points = xx[::steps, ::steps]
xx = xx.reshape(-1, 2).astype(np.float32)
xx = tp.spaces.Points(torch.tensor(xx), X)

a = torch.zeros(1) # dummy

kernel = [[0, 1, 0], [1, 0, -1], [0, -1, 0]] # to compute where a has jumps

### Boundary conditions for DRM and PINN/QRES
def bound_residual(u):
    return u**2
def bound_residual_pinn(u):
    return u
bound_sampler = tp.samplers.GridSampler(square.boundary, n_points=int(2052/steps)).make_static()

### PDE 
def energy_residual(u, x):
    return 1/2 * a * torch.sum(tp.utils.grad(u, x)**2, dim=1, keepdim=True) - 1.0

def pde_residual(u, x):
    return a*tp.utils.laplacian(u, x) + 1.0

### Load data and scale
#path = 'darcy/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5'
#hf = h5py.File(path, "r")
#output_data = hf['test_out'][:average_exmaples, :, :]
#input_data = hf['test_in'][:average_exmaples, :, :]
#hf.close()
current_path = os.path.dirname(os.path.abspath(__file__))
input_data = np.load(current_path + "/f_data.npy")[:average_exmaples, ::steps, ::steps]
output_data = np.load(current_path + "/u_data.npy")[:average_exmaples]

input_data = scaling * input_data
output_data = torch.tensor(output_data)

### Define array to store error of each run
l2_error_array = np.zeros(average_exmaples)

for i in range(average_exmaples):
    print("current step:", i)
    ### Define model (or reset for next example)
    #model = tp.models.DeepRitzNet(input_space=X, output_space=U, depth=3, width=60)
    #model = tp.models.FCN(input_space=X, output_space=U, hidden=(50, 50, 50, 20))
    model = tp.models.QRES(input_space=X, output_space=U, hidden=(36, 36, 36, 15))
    
    ### Set current data
    a = input_data[i]

    ### Remove points where a has a jump (there the strong PDE is not well defined)
    conv_matrxi = scipy.signal.convolve2d(a, kernel, mode="same", boundary='symm')
    okay_p = (conv_matrxi == 0)
    a = torch.tensor(a[okay_p]).reshape(-1, 1)
    a = a.to('cuda')
    okay_points = x_points[okay_p].reshape(-1, 2).astype(np.float32)
    okay_points = tp.spaces.Points(torch.tensor(okay_points), X)
    pde_sampler = tp.samplers.DataSampler(okay_points)
    ### Or just use all points:
    # in_points = x_points.reshape(-1, 2).astype(np.float32)
    # pde_sampler = tp.samplers.DataSampler(tp.spaces.Points(torch.tensor(in_points), X))
    # a = torch.tensor(a).reshape(-1, 1)
    # a = a.to('cuda')

    ### Define training conditions
    ### DRM:
    #bound_cond = tp.conditions.DeepRitzCondition(module=model, sampler=bound_sampler, 
    #                                             integrand_fn=bound_residual, weight=100)
    #pde_cond = tp.conditions.DeepRitzCondition(model, pde_sampler, energy_residual)
    ### PINN/QRES
    bound_cond = tp.conditions.PINNCondition(module=model, sampler=bound_sampler, 
                                             residual_fn=bound_residual_pinn, weight=100)
    pde_cond = tp.conditions.PINNCondition(module=model, sampler=pde_sampler, residual_fn=pde_residual)
   
    ### Start training
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)
    solver = tp.solver.Solver(train_conditions=[pde_cond, bound_cond], optimizer_setting=optim)
    trainer = pl.Trainer(gpus=1, max_steps=5000, logger=False, benchmark=True,
                         checkpoint_callback=False, weights_summary=None, progress_bar_refresh_rate=0)       
    trainer.fit(solver)
    
    ### Switch to LBFGS
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.LBFGS, lr=0.25, 
                                optimizer_args={'max_iter': 2, 'history_size': 100})
    solver = tp.solver.Solver(train_conditions=[pde_cond, bound_cond], optimizer_setting=optim)
    trainer = pl.Trainer(gpus=1, max_steps=12000, logger=False, benchmark=True,
                         checkpoint_callback=False, weights_summary=None, progress_bar_refresh_rate=0) 
    trainer.fit(solver)

    ### Compute current error:
    model.to('cpu')
    u_out = model(xx)
    u_out = (u_out.as_tensor * scaling).detach().cpu().reshape(1, Nx+1, Nx+1)
    diff = np.abs(u_out[0] - output_data[i])
    l2_rel = torch.sqrt(torch.sum(diff**2))/torch.sqrt(torch.sum(output_data[i]**2))
    l2_error_array[i] = l2_rel
    print("In step ", i, "had error:", l2_rel)
    np.save(current_path + "/l2_error.npy", l2_error_array)

    ### Save some solution:
    #if i == 0:
    #    np.save(current_path + "first_sol.npy", u_out)

print("Average error:", np.sum(l2_error_array)/len(l2_error_array))