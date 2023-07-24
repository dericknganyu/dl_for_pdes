import pytorch_lightning as pl

from poisson_setup import *


Output = torch.flatten(Output, 1, 2)  * 100


# Data loading
n = -1
branch_batch_size=128

dataloader = torch.utils.data.DataLoader(
    tp.utils.data.deeponet_dataloader.DeepONetDataset(
        branch_data_points=Input_data,
        trunk_data_points=X,
        out_data_points=torch.cat([F_data, Output], -1),
        branch_space=F,
        trunk_space=T,
        output_space=F*U,
        branch_batch_size=branch_batch_size,
        trunk_batch_size=100,
        shuffle_branch=False,
        shuffle_trunk=True),
    batch_size=None,
    shuffle=False,
    num_workers=0,
    pin_memory=False)


class PoissonCondition(tp.conditions.Condition):
    def __init__(self, model, dataloader):
        super().__init__(name='poisson', weight=1.)
        self.dataloader = dataloader
        self.model = model
    def forward(self, device='cpu', iteration=None):
        try:
            batch = next(self.iterator)
        except (StopIteration, AttributeError):
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        branch_in, trunk_in, out = batch
        branch_in, trunk_in, out = branch_in.to(device), trunk_in.to(device), out.to(device)
        self.model.branch(branch_in)
        f = out[..., 'f']
        u = out[..., 'u'].as_tensor
        trunk_in = tp.spaces.Points(trunk_in.as_tensor.expand(*f.shape, 2), trunk_in.space)
        trunk_in_coord, trunk_in = trunk_in.track_coord_gradients()
        #print("shape", trunk_in.as_tensor.shape)
        model_out = self.model(trunk_in).as_tensor
        laplace = tp.utils.laplacian(model_out, trunk_in_coord['t'])
        deviation = model_out-u
        data_loss = torch.mean(deviation**2)
        physics_loss = torch.mean((0.01*laplace+f.as_tensor)**2)
        if (iteration % 400) == 0:
            print('relative error', torch.norm(deviation)/torch.norm(u))
            print('physics loss', physics_loss)
        return torch.sqrt(data_loss) + physics_loss

class BoundaryCondition(tp.conditions.Condition):
    def __init__(self, model):
        super().__init__(name='boundary', weight=1.)
        self.model = model

    def forward(self, device='cpu', iteration=None):
        # since this is called after the other condition, we don't need to evaluate the branch
        trunk_in = boundary_sampler.sample_points(device=device)
        trunk_in = tp.spaces.Points(trunk_in.as_tensor.expand(branch_batch_size, trunk_in.shape[0], 2), trunk_in.space)
        #self.model.branch(branch_in, iteration_num=iteration, device=device)
        return torch.mean(self.model(trunk_in).as_tensor**2)


lrate = 0.0005
for i in range(30):
    lrate = 0.8 * lrate

    optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=lrate)
    solver = tp.solver.Solver([PoissonCondition(model, dataloader), BoundaryCondition(model)], val_conditions=(), optimizer_setting=optim)

    trainer = pl.Trainer(gpus='-1' if torch.cuda.is_available() else None,
                         num_sanity_val_steps=0,
                         benchmark=True,
                         max_steps=55000,
                         #logger=False,
                         checkpoint_callback=False,
                         #enable_progress_bar=True,
                         progress_bar_refresh_rate=20,
                         gradient_clip_val=5.0,
                         )
    trainer.fit(solver)
    if (i % 2) == 1:
        torch.save(model.state_dict(), f'pi-deepo-4-{i}-513-after-adam.pt')
