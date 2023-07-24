import pytorch_lightning as pl

from poisson_setup import *


noise_level = 0.1


Output = Output * 50
noise = torch.randn_like(Output)
Output = Output + noise_level*noise*torch.norm(Output)/torch.norm(noise)


U_data = Output_[:,locaF_out,:]
U_data = U_data[:,:,locaF_out]
U_data = torch.Tensor(U_data).unsqueeze(-1).float().cuda()
U_data = U_data + noise_level*noise*torch.norm(Output)/torch.norm(noise)
U_data = torch.flatten(U_data, 1, 2)
U_data = U_data * 50


interpolation_trunk_net = tp.models.FCTrunkNet(T, hidden=(128,128,128,128,128,128))
interpolation_trunk_net.finalize(U, p)

# data loading
n = -1
branch_batch_size=128

dataloader = torch.utils.data.DataLoader(
    tp.utils.data.deeponet_dataloader.DeepONetDataset(
        branch_data_points=Output,
        trunk_data_points=X,
        out_data_points=torch.cat([F_data, U_data], -1),
        branch_space=U,
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
        f = out[..., 'f']
        u = out[..., 'u']
        self.model.branch(branch_in)

        trunk_in._t = trunk_in._t.expand(*u.shape, 2)

        model_out = self.model(trunk_in).as_tensor

        #trunk_in = tp.spaces.Points(trunk_in, trunk_space)
        trunk_in_coord, trunk_in = trunk_in.track_coord_gradients()

        # second trunk net with same branch net
        interpolation_trunk_out = self.interpolation_trunk_net(trunk_in)
        interpolation_trunk_out = interpolation_trunk_out.unsqueeze(0) # shape = [1, trunk_n, dim, neurons]
        interpolation_out = torch.sum(interpolation_trunk_out * self.model.branch.current_out.unsqueeze(1), dim=-1)

        interpolation_loss = torch.mean((interpolation_out-u.as_tensor)**2)

        laplace = tp.utils.laplacian(interpolation_out, trunk_in_coord['t'])

        deviation = model_out-f.as_tensor
        data_loss = torch.mean(deviation**2)

        physics_loss = torch.mean((0.02*laplace+model_out)**2)
        if (iteration % 400) == 0:
            print('relative error', torch.norm(deviation)/torch.norm(out.as_tensor))
            print('physics loss', physics_loss)
            print('interpolation_loss', interpolation_loss)
            print('data loss', data_loss)
            print('interpolation error', torch.norm(interpolation_out-u.as_tensor)/torch.norm(u.as_tensor))
        return interpolation_loss + physics_loss + data_loss


lrate = 0.00015
for i in range(30):
    lrate = 0.85 * lrate

    optim = tp.OptimizerSetting(optimizer_class=torch.optim.AdamW, lr=lrate)
    solver = tp.solver.Solver([PoissonCondition(model, interpolation_trunk_net, dataloader)],
                               val_conditions=(), optimizer_setting=optim)

    trainer = pl.Trainer(gpus='-1' if torch.cuda.is_available() else None,
                         num_sanity_val_steps=0,
                         benchmark=True,
                         max_steps=55000,
                         #logger=False,
                         checkpoint_callback=False,
                         #enable_progress_bar=True,
                         progress_bar_refresh_rate=20,
                         gradient_clip_val=3.0,
                         )
    trainer.fit(solver)
    if (i % 4) == 1:
        torch.save(model.state_dict(), f'pi-deepo-inverse-{i}-65-noise-10-after-adam.pt')
