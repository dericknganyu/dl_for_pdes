import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torchphysics as tp
import numpy as np
import torch
import pytorch_lightning as pl

### Computes the fluid velocity and pressure in the domain
### with the rotating rod.

radius = 5.0
widht, height = 0.5, 4.0
t_0, t_end = 0.0, 1.0
a_0, a_end = np.pi, 2*np.pi # rotation speed
k_cool, k_hot = 270, 330 # temperature values
mu = 2.0 # viscosity
lamb = 0.5 # heat diffusion

def start_fn(t, a):
    return 1 - torch.exp(-a*t)

X = tp.spaces.R2('x') # space
T = tp.spaces.R1('t') # time
A = tp.spaces.R1('a') # rotation speed
U = tp.spaces.R2('u') # velocity
P = tp.spaces.R1('p') # pressure

def rotation_function(a, t):
    # rotate clockwise and use t*(1-e^-at) -> velocity has a smooth start up
    return - a * t * start_fn(t, a)

circle = tp.domains.Circle(X, [0.0, 0.0], radius)
rod = tp.domains.Parallelogram(X, [-widht, -height], [widht, -height], [-widht, height])
rod = tp.domains.Rotate.from_angles(rod, rotation_function)
omega = circle - rod
t_int = tp.domains.Interval(T, t_0, t_end)
a_int = tp.domains.Interval(A, a_0, a_end)

model_u = tp.models.Sequential(
    tp.models.NormalizationLayer(circle*t_int*a_int),
    tp.models.FCN(input_space=X*T*A, output_space=U, hidden=(100,100,100,100,100,100,100))
)
model_p = tp.models.Sequential(
    tp.models.NormalizationLayer(circle*t_int*a_int),
    tp.models.FCN(input_space=X*T*A, output_space=P, hidden=(80,80,80,80,80,80))
)

model_u_p = tp.models.Parallel(model_u, model_p)

## load models:
#model_u.load_state_dict(torch.load(os.path.join(sys.path[0], '../new_u_net.pt')))
#model_p.load_state_dict(torch.load(os.path.join(sys.path[0], '../new_p_net.pt')))


# constrain for the velocity:
# outer boundary u = 0 and initialy u = 0
# and already use rod velocity 
def constrain_fn_u(u, x, t, a):
    time_scale = start_fn(t, a)
    rot_speed = radius*a*torch.column_stack((torch.cos(-a*t*time_scale), 
                                            torch.sin(-a*t*time_scale)))
    rot_speed *= (1 - torch.exp(-a*t) + a*t*torch.exp(-a*t))
    distance = x[:, :1]**2 + x[:, 1:]**2
    u_con = rot_speed * u * (1 - distance / radius**2)
    return u_con
# constrain for the pressure:
def constrain_fn_p(p, t, a):
    p_con = p * start_fn(t, a)
    return p_con

inner_sampler = tp.samplers.RandomUniformSampler(circle*(t_int*a_int), n_points=30000)
rod_sampler = tp.samplers.RandomUniformSampler(rod.boundary*(t_int*a_int), n_points=20000)

#tp.utils.scatter(X*T, rod_sampler)

def div_resiudal(u, x, t, a):
    u_con = constrain_fn_u(u, x, t, a)
    return tp.utils.div(u_con, x)

div_condition = tp.conditions.PINNCondition(model_u, inner_sampler, 
                                            div_resiudal, name='div_cond', weight=1)


# navier stokes
def ns_resiudal(u, p, x, t, a):
    u = constrain_fn_u(u, x, t, a)
    p = constrain_fn_p(p, t, a)
    grad_p = tp.utils.grad(p, x)
    u_t = tp.utils.jac(u, t).squeeze(-1)
    u_jac = tp.utils.jac(u, x)
    u_1 = tp.utils.laplacian(u[:, :1], x)
    u_2 = tp.utils.laplacian(u[:, 1:], x)
    u_lap = torch.column_stack((u_1, u_2))
    return u_t + torch.bmm(u_jac, u.unsqueeze(-1)).squeeze(-1) - mu*u_lap + grad_p

ns_condition = tp.conditions.PINNCondition(model_u_p, inner_sampler, 
                                            ns_resiudal, name='ns_cond', weight=0.05)

# points further away from the center have a greater radial velocity.
# To determine the speed at point x, we have to compute the distance to 
# the line (cos(-a*t*(1-e^-at)), sin(-a*t*(1-e^-at))) [orthogonal to rotating rod]
# -> normal vector (-sin(-a*t*(1-e^-at)), cos(-a*t*(1-e^-at))) is needed
def radial_velocity(x, t, a):
    time_scale = start_fn(t, a)
    normal_vector = torch.column_stack((-torch.sin(-a*t*time_scale), torch.cos(-a*t*time_scale)))
    distance = torch.sum(normal_vector*x, dim=1).reshape(-1, 1)
    angular_velocity = a*torch.column_stack((normal_vector[:, 1:], -normal_vector[:, :1]))
    # for velocity use chain rule on rotation function
    return distance * angular_velocity * (1 - torch.exp(-a*t) + a*t*torch.exp(-a*t))

def bc_velocity_residual(u, t, x, a):
    u = constrain_fn_u(u, x, t, a)
    expected_velocity = radial_velocity(x, t, a)
    return u - expected_velocity

bc_velocity_condition = tp.conditions.PINNCondition(model_u, rod_sampler, 
                                                    bc_velocity_residual, 
                                                    name='rot_cond', weight=20)


### Start multiple training runs:
optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.001) 
solver = tp.solver.Solver([bc_velocity_condition, ns_condition, div_condition],
                          optimizer_setting=optim)

trainer = pl.Trainer(gpus=1, # or None for CPU
                     max_steps=5000,
                     #logger=False,
                     benchmark=True,
                     checkpoint_callback=False)
trainer.fit(solver)

optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.0005)
solver = tp.solver.Solver([ns_condition, div_condition, bc_velocity_condition],
                          optimizer_setting=optim)

trainer = pl.Trainer(gpus=1, # or None for CPU
                     max_steps=10000,
                     #logger=False,
                     benchmark=True,
                     checkpoint_callback=False)
trainer.fit(solver)

torch.save(model_u.state_dict(), 'final_u.pt')
torch.save(model_p.state_dict(), 'final_p.pt')

optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.0001)
solver = tp.solver.Solver([ns_condition, div_condition, bc_velocity_condition],
                          optimizer_setting=optim)

trainer = pl.Trainer(gpus=1, # or None for CPU
                     max_steps=10000,
                     #logger=False,
                     benchmark=True,
                     checkpoint_callback=False)
trainer.fit(solver)

torch.save(model_u.state_dict(), 'final_u.pt')
torch.save(model_p.state_dict(), 'final_p.pt')

optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=0.00001)
solver = tp.solver.Solver([ns_condition, div_condition, bc_velocity_condition],
                          optimizer_setting=optim)

trainer = pl.Trainer(gpus=1, # or None for CPU
                     max_steps=20000,
                     #logger=False,
                     benchmark=True,
                     checkpoint_callback=False)
trainer.fit(solver)

torch.save(model_u.state_dict(), 'final_u.pt')
torch.save(model_p.state_dict(), 'final_p.pt')
