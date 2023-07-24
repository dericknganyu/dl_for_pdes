import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import numpy as np
import h5py

import torchphysics as tp

torch.set_default_tensor_type(torch.FloatTensor)

torch.manual_seed(1234)
np.random.seed(1234)


path = '/localdata/nick/poisson/new_aUP_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5'

hf = h5py.File(path, "r")

Input = hf['train_in']
Output_ = hf['train_out']
Nd_in = 65
Nd_out = 65
locaF_in = [8 * i for i in range(Nd_in)]
locaF_out = [8 * i for i in range(Nd_out)]
xx = np.linspace(0,1,Nd_out).astype(np.float32)
X1, X2 = np.meshgrid(xx,xx)
X = np.stack((X1,X2), axis=-1)

X = torch.flatten(torch.tensor(X), 0, 1)

dis_xx = np.linspace(0,1,Nd_in).astype(np.float32)
dis_X1, dis_X2 = np.meshgrid(dis_xx,dis_xx)
dis_X = np.stack((dis_X1,dis_X2), axis=-1)
dis_X = torch.flatten(torch.tensor(dis_X), 0, 1)

# data on the grid points
Input_data = Input[:,locaF_in,:]
Input_data = Input_data[:,:,locaF_in]
Input_data = 0.1*torch.Tensor(Input_data).unsqueeze(-1).float()#.cuda()

# compute the values at the edge that should not be considered in physics loss
print('max_in', torch.max(Input_data), torch.min(Input_data))
print(Input_data.shape)
conv_filter = torch.Tensor([[1., 1., 1.],[1., -8, 1.],[1., 1., 1.]]).unsqueeze(0).unsqueeze(0).cuda()
conv_input = Input_data.squeeze(-1).unsqueeze(1)
zs = torch.zeros(*conv_input.shape[:-2], conv_input.shape[-2]+2, conv_input.shape[-1]+2)
zs[..., 1:-1, 1:-1] = conv_input
zs[..., -1] = zs[..., -2]
zs[..., 0] = zs[..., 1]
zs[..., -1, :] = zs[..., -2, :]
zs[..., 0, :] = zs[..., 1, :]
conv2d_out = torch.nn.functional.conv2d(zs.cuda(), conv_filter)
conv2d_out = (conv2d_out == 0).float().moveaxis(1,-1).cuda()
print(conv2d_out.shape)
conv2d_out = torch.flatten(conv2d_out, 1, 2)


F_data = Input[:,locaF_out,:]
F_data = F_data[:,:,locaF_out]
F_data = 0.1*torch.Tensor(F_data).unsqueeze(-1).float().cuda()
F_data = torch.flatten(F_data, 1, 2)

noise_level = 0.1

Output = Output_[:,locaF_out,:]
Output = Output[:,:,locaF_out]
Output = 50 * torch.Tensor(Output).unsqueeze(-1).float().cuda()

noise = torch.randn_like(Output)
Output = Output + noise_level*noise*torch.norm(Output)/torch.norm(noise)


U_data = Output_[:,locaF_out,:]
U_data = U_data[:,:,locaF_out]
U_data = 50 * torch.Tensor(U_data).unsqueeze(-1).float().cuda()
U_data = U_data + noise_level*noise*torch.norm(Output)/torch.norm(noise)
U_data = torch.flatten(U_data, 1, 2)

# Spaces
T = tp.spaces.R2('t') # input variable
U = tp.spaces.R1('u') # output variable
F = tp.spaces.R1('f') # function space name
C = tp.spaces.R1('c') # dummy space used to cut some points


# Domains
T_int = tp.domains.Parallelogram(T, [0, 0], [1, 0], [0, 1])
Fn_space = tp.spaces.FunctionSpace(T_int, F)
dis_sampler = tp.samplers.DataSampler(tp.spaces.Points(X, T)).make_static()
boundary_sampler = tp.samplers.GridSampler(T_int.boundary, n_points = 8*65).make_static()


class Conv2dBranchNet(tp.models.BranchNet):
    def __init__(self, function_space, output_neurons,
                 discretization_sampler):
        super().__init__(function_space, discretization_sampler)
        self.sequential = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(25, 25)),
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2048, output_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(output_neurons, output_neurons),
        )

    def forward(self, x):
        x = x.as_tensor
        x = torch.swapaxes(x, -1, -3)
        self.current_out =  self._reshape_multidimensional_output(
            self.sequential(x))


class Sinus(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)

class FCTrunkNetSin(tp.models.TrunkNet):
    def __init__(self, input_space, output_neurons):
        super().__init__(input_space)
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(2, 128),
            Sinus(),
            torch.nn.Linear(128, 128),
            Sinus(),
            torch.nn.Linear(128, 128),
            Sinus(),
            torch.nn.Linear(128, 128),
            Sinus(),
            torch.nn.Linear(128, 128),
            Sinus(),
            torch.nn.Linear(128, 128),
            Sinus(),
            torch.nn.Linear(128, output_neurons),         
        )

    def forward(self, points):
        points = self._fix_points_order(points)
        return self._reshape_multidimensional_output(self.sequential(points.as_tensor))

class FCTrunkNet(tp.models.TrunkNet):
    def __init__(self, input_space, output_neurons):
        super().__init__(input_space)
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(2, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, output_neurons),         
        )

    def forward(self, points):
        points = self._fix_points_order(points)
        return self._reshape_multidimensional_output(self.sequential(points.as_tensor))



# Model

p = 128
solution_trunk_net = FCTrunkNetSin(T, output_neurons=p)
branch_net = Conv2dBranchNet(Fn_space, output_neurons=p,
                             discretization_sampler=dis_sampler)

interpolation_trunk_net = FCTrunkNet(T, output_neurons=p)
interpolation_trunk_net.finalize(U, p)

model = tp.models.DeepONet(solution_trunk_net, branch_net, output_space=U, output_neurons=p)


n = -1
branch_batch_size=116

dataloader = tp.utils.DeepONetDataLoader(branch_data=Output, trunk_data=X,
                                         output_data=torch.cat([F_data, U_data, conv2d_out], -1), branch_space=U,
                                         trunk_space=T, output_space=F*U*C,
                                         branch_batch_size=branch_batch_size, trunk_batch_size=100)


class PoissonCondition(tp.conditions.Condition):
    def __init__(self, model, interpolation_trunk_net, dataloader):
        super().__init__(name='poisson', weight=1.)
        self.dataloader = dataloader
        self.model = model
        self.interpolation_trunk_net = interpolation_trunk_net

    def forward(self, device='cpu', iteration=None):
        try:
            batch = next(self.iterator)
        except (StopIteration, AttributeError):
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        branch_in, trunk_in, out = batch
        branch_in, trunk_in, out = branch_in.to(device), trunk_in.to(device), out.to(device)
        out = out.as_tensor
        f = out[..., 0:1]
        u = out[..., 1:2]
        cut = out[..., 2:3]
        self.model.branch(branch_in)

        trunk_in._t = trunk_in._t.expand(*u.shape[:-1], 2)

        model_out = self.model(trunk_in).as_tensor

        trunk_in_coord, trunk_in = trunk_in.track_coord_gradients()

        # second trunk net with same branch net
        interpolation_trunk_out = self.interpolation_trunk_net(trunk_in)
        interpolation_out = torch.sum(interpolation_trunk_out * self.model.branch.current_out.unsqueeze(1), dim=-1)
        interpolation_loss = torch.mean((interpolation_out-u)**2)

        laplace = tp.utils.laplacian(0.02*interpolation_out, trunk_in_coord['t']) # output is scaled
        divergence = 10*model_out*laplace

        physics_loss = torch.mean((divergence+1)**2*cut)

        deviation = model_out-f
        data_loss = torch.mean(deviation**2)
        if (iteration % 400) == 0:
            print('relative error', torch.norm(deviation)/torch.norm(f))
            print('physics loss', physics_loss)
            print('interpolation_loss', interpolation_loss)
            print('data loss', data_loss)
            print('interpolation error', torch.norm(interpolation_out-u)/torch.norm(u))
            print(interpolation_out.shape)
        return 10.0*interpolation_loss + physics_loss + data_loss


import pytorch_lightning as pl

lrate = 0.00015
for i in range(30):
    lrate = 0.85 * lrate

    optim = tp.OptimizerSetting(optimizer_class=torch.optim.AdamW, lr=lrate)
    solver = tp.solver.Solver([PoissonCondition(model, interpolation_trunk_net, dataloader)],
                               val_conditions=(), optimizer_setting=optim)
    pytorch_total_params = sum(p.numel() for p in solver.parameters() if p.requires_grad)
    print('PARAMS', pytorch_total_params)
    trainer = pl.Trainer(gpus='-1' if torch.cuda.is_available() else None,
                         num_sanity_val_steps=0,
                         benchmark=True,
                         max_steps=55000,
                         logger=True,
                         checkpoint_callback=False,
                         #enable_progress_bar=True,
                         progress_bar_refresh_rate=20,
                         gradient_clip_val=3.0,
                         )
    trainer.fit(solver)
    if (i % 4) == 1:
        torch.save(model.state_dict(), f'pi-deepo-inverse-darcy-{i}-hybrid-65-noise10-after-adam.pt')
