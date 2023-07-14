import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torchphysics as tp
import numpy as np
import torch
import pytorch_lightning as pl

### Computes the fluid temperature in the domain
### with the rotating rod. Is decoupled from the fluid flow, 
### which has to be computed before.

radius = 5.0
widht, height = 0.5, 4.0
t_0, t_end = 0.0, 1.0
a_0, a_end = np.pi, 2*np.pi # rotation speed
k_cool, k_hot = 270, 330 # temperature values
mu = 2.0 # viscosity
lamb = 0.5 # heat diffusion

def start_fn(t, a):
    return 1 - torch.exp(-a*t)
# dont want to jump from initial temperature to k_hot
def heat_up_fn(t, a):
    return k_cool + (k_hot - k_cool) * start_fn(t, a)

X = tp.spaces.R2('x') # space
T = tp.spaces.R1('t') # time
A = tp.spaces.R1('a') # rotation speed
U = tp.spaces.R2('u') # velocity
P = tp.spaces.R1('p') # pressure
K = tp.spaces.R1('k') # temperature#

def rotation_function(a, t):
    # rotate clockwise and use t*(1-e^-at) -> velocity has a smooth start up
    return - a * t * start_fn(t, a)

circle = tp.domains.Circle(X, [0.0, 0.0], radius)
rod = tp.domains.Parallelogram(X, [-widht, -height], [widht, -height], [-widht, height])
rod = tp.domains.Rotate.from_angles(rod, rotation_function)
omega = circle - rod
t_int = tp.domains.Interval(T, t_0, t_end)
a_int = tp.domains.Interval(A, a_0, a_end)


ac_fn = tp.models.AdaptiveActivationFunction(torch.nn.Tanh())
model_u = tp.models.Sequential(
    tp.models.NormalizationLayer(circle*t_int*a_int),
    tp.models.FCN(input_space=X*T*A, output_space=U, hidden=(100,100,100,100,100,100,100), 
                  activations=ac_fn)
)

# model for temperature
ac_fn_temp = tp.models.AdaptiveActivationFunction(torch.nn.Tanh())
model_k = tp.models.Sequential(
    tp.models.NormalizationLayer(circle*t_int*a_int),
    tp.models.FCN(input_space=X*T*A, output_space=K, hidden=(100,100,100,100,100,100,100), 
                  activations=ac_fn_temp)
)


## load model for fluid velocity:
model_u.load_state_dict(torch.load(os.path.join(sys.path[0], '../final_u.pt')))

# constrain for the velocity:
# outer boundary u = 0 and initialy u = 0
def constrain_fn_u(u, x, t, a):
    time_scale = start_fn(t, a)
    rot_speed = radius*a*torch.column_stack((torch.cos(-a*t*time_scale), 
                                            torch.sin(-a*t*time_scale)))
    rot_speed *= (1 - torch.exp(-a*t) + a*t*torch.exp(-a*t))
    distance = x[:, :1]**2 + x[:, 1:]**2
    u_con = rot_speed * u * (1 - distance / radius**2)
    return u_con
# constrain for the temperature:
# initialy k = k_cool and on outer boundary k = k_cool
def constrain_fn_k(k, x, t, a):
    distance = x[:, :1]**2 + x[:, 1:]**2
    k_con = (k_hot - k_cool) * start_fn(t, a) * k * (1.0 - distance/radius**2) + k_cool
    return k_con

# maybe train pde everywhere and ignore the hole 
# (should not matter for the physics but is more efficient in ADAM)
inner_sampler = tp.samplers.RandomUniformSampler(circle*(t_int*a_int), n_points=90000)
rod_sampler = tp.samplers.RandomUniformSampler(rod.boundary*(t_int*a_int), n_points=50000)


# heat equation
def heat_residual(k, x, t, a):
    u = model_u(tp.spaces.Points(torch.column_stack((x, t, a)), X*T*A))
    u = constrain_fn_u(u.as_tensor, x, t, a)
    k = constrain_fn_k(k, x, t, a)
    grad_k = tp.utils.grad(k, x, t) # first and second entry for x-gradient, last is time
    lap_k = tp.utils.laplacian(k, x, grad=grad_k[:, :2])
    conv = torch.sum(grad_k[:, :2]*u, dim=1).reshape(-1, 1)
    return - lamb * lap_k + conv + grad_k[:, 2:]

heat_condition = tp.conditions.PINNCondition(model_k, inner_sampler, heat_residual, name='heat_pde', 
                                             weight=0.1)

def heat_bc_residual(k, x, t, a):
    k = constrain_fn_k(k, x, t, a)
    return k - heat_up_fn(t, a)

bc_heat_condition = tp.conditions.PINNCondition(model_k, rod_sampler, heat_bc_residual, weight=50, 
                                                name='heat_bc')


model_u.to('cuda')
optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.0005) 
solver = tp.solver.Solver([bc_heat_condition],
                          optimizer_setting=optim)

trainer = pl.Trainer(gpus=1, # or None for CPU
                     max_steps=2500,
                     #logger=False,
                     benchmark=True,
                     checkpoint_callback=False)
trainer.fit(solver)
optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.01) 
solver = tp.solver.Solver([bc_heat_condition, heat_condition],
                          optimizer_setting=optim)

trainer = pl.Trainer(gpus=1, # or None for CPU
                     max_steps=10000,
                     #logger=False,
                     benchmark=True,
                     checkpoint_callback=False)
trainer.fit(solver)


optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001)
solver = tp.solver.Solver([bc_heat_condition, heat_condition],
                          optimizer_setting=optim)

trainer = pl.Trainer(gpus=1, # or None for CPU
                     max_steps=10000,
                     #logger=False,
                     benchmark=True,
                     checkpoint_callback=False)
trainer.fit(solver)

optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.0001)
solver = tp.solver.Solver([bc_heat_condition, heat_condition],
                          optimizer_setting=optim)

trainer = pl.Trainer(gpus=1, # or None for CPU
                     max_steps=10000,
                     #logger=False,
                     benchmark=True,
                     checkpoint_callback=False)
trainer.fit(solver)

optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.00001)
solver = tp.solver.Solver([bc_heat_condition, heat_condition],
                          optimizer_setting=optim)

trainer = pl.Trainer(gpus=1, # or None for CPU
                     max_steps=20000,
                     #logger=False,
                     benchmark=True,
                     checkpoint_callback=False)
trainer.fit(solver)

torch.save(model_k.state_dict(), 'final_temp.pt')


# # LBFGS 
# # first make all samplers static
# # for navier-stokes inner t = 20, x = 150
# #                   boundary t = 25, x = 300
# a_sampler = tp.samplers.GridSampler(a_int, n_points=18) \
#             + tp.samplers.GridSampler(a_int.boundary, n_points=2)
# t_sampler = tp.samplers.GridSampler(t_int, n_points=40) \
#             + tp.samplers.GridSampler(t_int.boundary, n_points=2)
# inner_sampler = tp.samplers.RandomUniformSampler(circle, n_points=200) \
#                 * (t_sampler * a_sampler)
# a_sampler = tp.samplers.RandomUniformSampler(a_int, n_points=48) \
#             + tp.samplers.GridSampler(a_int.boundary, n_points=2)
# rod_sampler = tp.samplers.GridSampler(rod.boundary, n_points=350) \
#                 * (t_sampler * a_sampler)
# inner_sampler = inner_sampler.make_static()
# rod_sampler = rod_sampler.make_static()
# heat_condition.sampler = inner_sampler
# bc_heat_condition.sampler = rod_sampler

# optim = tp.OptimizerSetting(optimizer_class=torch.optim.LBFGS, lr=0.5, 
#                             optimizer_args={'max_iter': 2, 'history_size': 100})
# solver = tp.solver.Solver([bc_heat_condition, heat_condition],
#                         optimizer_setting=optim)

# trainer = pl.Trainer(gpus=1, # or None for CPU
#                     max_steps=20000,
#                     #logger=False,
#                     benchmark=True,
#                     checkpoint_callback=False)
# trainer.fit(solver)
