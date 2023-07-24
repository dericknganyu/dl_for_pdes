import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import numpy as np
import h5py

import torchphysics as tp

torch.set_default_tensor_type(torch.FloatTensor)

torch.manual_seed(1234)
np.random.seed(1234)

path = '/localdata/nick/poisson/fUG_Square_TrainData=1024_TestData=5000_Resolution=513X513_Domain=[0,1]X[0,1].hdf5'
#
hf = h5py.File(path, "r")
#
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
Input_data = torch.Tensor(Input_data).unsqueeze(-1).float().cuda()

F_data = Input[:,locaF_out,:]
F_data = F_data[:,:,locaF_out]
F_data = torch.Tensor(F_data).unsqueeze(-1).float().cuda()
F_data = torch.flatten(F_data, 1, 2)

Output = Output_[:,locaF_out,:]
Output = Output[:,:,locaF_out]
Output = torch.Tensor(Output).unsqueeze(-1).float().cuda()


# Spaces
T = tp.spaces.R2('t') # input variable
U = tp.spaces.R1('u') # output variable
F = tp.spaces.R1('f') # function space name


# Domains
T_int = tp.domains.Parallelogram(T, [0, 0], [1, 0], [0, 1])
Fn_space = tp.spaces.FunctionSpace(T_int, F)
dis_sampler = tp.samplers.DataSampler(tp.spaces.Points(X, T)).make_static()
boundary_sampler = tp.samplers.GridSampler(T_int.boundary, n_points = 8*65).make_static()


class LinearBranchNet(tp.models.BranchNet):
    def __init__(self, function_space, output_neurons,
                 discretization_sampler, Nd_in):
        super().__init__(function_space, discretization_sampler)
        self.Nd_in = Nd_in
        self.sequential = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(65, 65)
                                       ),
            torch.nn.Flatten(),
            torch.nn.Linear(65 * 65, output_neurons, bias=False)
        )

    def forward(self, discrete_function_batch):
        self.current_out = self._reshape_multidimensional_output(
            self.sequential(discrete_function_batch.as_tensor.reshape(-1,1,self.Nd_in,self.Nd_in))
            )


p = 128
trunk_net = tp.models.FCTrunkNet(T, hidden=(128,128,128,128,128,128))
branch_net = LinearBranchNet(Fn_space, output_neurons=p,
                             discretization_sampler=dis_sampler, Nd_in=Nd_in)

model = tp.models.DeepONet(trunk_net, branch_net, output_space=U, output_neurons=p)
