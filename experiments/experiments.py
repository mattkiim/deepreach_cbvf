import wandb
import torch
import os
import shutil
import time
import math
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.io as spio
from mpl_toolkits.axes_grid1 import make_axes_locatable

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
from datetime import datetime
from sklearn import svm 
from utils import diff_operators
from utils.error_evaluators import scenario_optimization, ValueThresholdValidator, MultiValidator, MLPConditionedValidator, target_fraction, MLP, MLPValidator, SliceSampleGenerator

class Experiment(ABC):
    def __init__(self, model, dataset, experiment_dir, use_wandb):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb
        self.N = self.dataset.dynamics.state_dim

    @abstractmethod
    def init_special(self):
        raise NotImplementedError

    def _load_checkpoint(self, epoch):
        if epoch == -1:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_final.pth')
            self.model.load_state_dict(torch.load(model_path))
        else:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_epoch_%04d.pth' % epoch)
            self.model.load_state_dict(torch.load(model_path)['model'])

    
    def trajectory_rollout(self, policy, dynamics, tMin, tMax, dt, scenario_batch_size, initial_states, tStart_generator=None):
        state_trajs = torch.zeros(scenario_batch_size, int((tMax-tMin)/dt) + 1, dynamics.state_dim)
        ctrl_trajs = torch.zeros(scenario_batch_size, int((tMax-tMin)/dt), dynamics.control_dim)

        state_trajs[:, 0, :] = initial_states

        # rollout
        for k in tqdm(range(int((tMax-tMin)/dt)), desc='Trajectory Propagation'):
            traj_time = tMax - k * dt
            traj_times = torch.full((scenario_batch_size, ), traj_time)

            # get the states at t=k
            traj_coords = torch.cat((traj_times.unsqueeze(-1), state_trajs[:, k]), dim=-1)

            traj_policy_results = policy({'coords': dynamics.coord_to_input(traj_coords.cuda())}) # learned costate/gradient
            traj_dvs = dynamics.io_to_dv(traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach()

            # optimal control based on the policy's output
            ctrl_trajs[:, k] = dynamics.optimal_control(traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())

            state_trajs[:, k+1] = dynamics.equivalent_wrapped_state(
                state_trajs[:, k].cuda() + dt * dynamics.dsdt(state_trajs[:, k].cuda(), ctrl_trajs[:, k].cuda()).cuda()
            ).cpu()

        return state_trajs, ctrl_trajs


    def initialize_states_with_general_displacement(self, state_dim, theta, displacement):
        initial_states = torch.zeros(state_dim)
            
        x_start = -displacement * torch.cos(theta)  # x-coordinate displaced opposite to the orientation
        y_start = -displacement * torch.sin(theta)  # y-coordinate displaced opposite to the orientation

        initial_states[0] = x_start  # x position
        initial_states[1] = y_start  # y position

        initial_states[2] = theta  # orientation angle

        return initial_states
        

    def validate(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        fig = plt.figure(figsize=(5*len(times), 5*len(zs)))
        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

                with torch.no_grad():
                    model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                    # TODO: do something similar to this? add the optimal trajectory plot to the dataset?
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
                
                ax = fig.add_subplot(len(times), len(zs), (j+1) + i*len(zs))
                ax.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))
                s = ax.imshow(1*(values.detach().cpu().numpy().reshape(x_resolution, y_resolution).T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(s, cax=cax)
                # fig.colorbar(s) 

                # TODO: plot optimal trajectories
                optimal_traj = self.compute_optimal_trajectory(times[i], xys, zs[j], plot_config, coords, values)
                ax.plot(optimal_traj[:, 0], optimal_traj[:, 1], color='green', linestyle='-', marker='o')


        fig.savefig(save_path)
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'val_plot': wandb.Image(fig),
            })
        plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)
    
    def validateND(self, epoch, save_path, x_resolution, y_resolution, z_resolution, time_resolution, plot_value=True):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        # z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        # zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)
        Xg, Yg = torch.meshgrid(xs, ys)
        
        ## Plot Set and Value Fn

        fig_set = plt.figure(figsize=(5*len(times), 3*5*1))
        fig_val = plt.figure(figsize=(5*len(times), 3*5*1))

        for i in range(3*len(times)):
            
            ax_set = fig_set.add_subplot(3, len(times), 1+i)
            ax_val = fig_val.add_subplot(3, len(times), 1+i, projection='3d')
            ax_set.set_title('t = %0.2f' % (times[i % len(times)]))
            ax_val.set_title('t = %0.2f' % (times[i % len(times)]))

            ## Define Grid Slice to Plot

            coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
            coords[:, 0] = times[i % len(times)]
            coords[:, 1:] = torch.tensor(plot_config['state_slices']) # initialized to zero (nothing else to set!)

            if i < len(times): # xN - xi plane
                ax_set.set_xlabel("xN"); ax_set.set_ylabel("xi")
                ax_val.set_xlabel("xN"); ax_val.set_ylabel("xi")
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]

            elif i < 2*len(times): # xi - xj plane
                ax_set.set_xlabel("xi"); ax_set.set_ylabel("xj")
                ax_val.set_xlabel("xi"); ax_val.set_ylabel("xj")
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['z_axis_idx']] = xys[:, 1]

            else: # xN - (xi = xj) plane
                ax_set.set_xlabel("xN"); ax_set.set_ylabel("xi=xj")
                ax_val.set_xlabel("xN"); ax_val.set_ylabel("xi=xj")
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 2:] = (xys[:, 1] * torch.ones(self.N-1, xys.size()[0])).t()

            with torch.no_grad():
                model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())
            
            learned_value = values.detach().cpu().numpy().reshape(x_resolution, y_resolution)

            ## Plot Zero-level Set of Learned Value

            s = ax_set.imshow(1*(learned_value.T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
            divider = make_axes_locatable(ax_set)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig_set.colorbar(s, cax=cax)

            ## Plot Ground-Truth Zero-Level Contour

            n_grid_plane_pts = int(self.dataset.n_grid_pts/3)
            n_grid_len = int(n_grid_plane_pts ** 0.5)
            pix_start = (i // len(times)) * n_grid_plane_pts
            tix_start = (i % len(times)) * self.dataset.n_grid_pts
            ix = pix_start + tix_start
            Vg = self.dataset.values_DP_grid[ix:ix+n_grid_plane_pts][:n_grid_len**2].reshape(n_grid_len, n_grid_len)
            # Vg = self.dataset.values_DP_grid[ix:ix+n_grid_plane_pts]
            
            ax_set.contour(self.dataset.X1g, self.dataset.X2g, Vg.cpu(), [0.])

            ## Plot the Linear Ground-Truth (ideal warm-start) Zero-Level Contour

            Vg = self.dataset.values_DP_linear_grid[ix:ix+n_grid_plane_pts][:n_grid_len**2].reshape(n_grid_len, n_grid_len)
            # Vg = self.dataset.values_DP_linear_grid[ix:ix+n_grid_plane_pts]

            ax_set.contour(self.dataset.X1g, self.dataset.X2g, Vg.cpu(), [0.], colors='gold', linestyles='dashed')

            ## Plot 3D Value Fn

            if plot_value:
                if learned_value.min() > 0:
                    RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                elif learned_value.max() < 0:
                    RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                else:
                    n_bins_high = int(256 * (learned_value.max()/(learned_value.max() - learned_value.min())) // 1)
                    RdWh = matplotlib.colors.LinearSegmentedColormap.from_list('RdWh', [(1,0,0), (1,0,0), (1,0.5,0.5), (1,1,1)])
                    WhBl = matplotlib.colors.LinearSegmentedColormap.from_list('WhBl', [(1,1,1), (0.5,0.5,1), (0,0,1), (0,0,1)])
                    colors = np.vstack((RdWh(np.linspace(0., 1, 256-n_bins_high)), WhBl(np.linspace(0., 1, n_bins_high))))
                    RdWhBl_vscaled = matplotlib.colors.LinearSegmentedColormap.from_list('RdWhBl_vscaled', colors)
                
                ax_val.view_init(elev=15, azim=-60)
                surf = ax_val.plot_surface(Xg, Yg, learned_value, cmap=RdWhBl_vscaled) #cmap='bwr_r')
                fig_val.colorbar(surf, ax=ax_val, fraction=0.02, pad=0.1)
                ax_val.set_zlim(-max(ax_val.get_zlim()[1]/5, 0.5))
                ax_val.contour(Xg, Yg, learned_value, zdir='z', offset=ax_val.get_zlim()[0], cmap=RdWh, levels=[0.]) #cmap='bwr_r')

        fig_set.savefig(save_path)
        if plot_value: fig_val.savefig(save_path.split('_epoch')[0] + '_Vfn' + save_path.split('_epoch')[1])
        if self.use_wandb:
            log_dict_plot = {'step': epoch,
                        'val_plot': wandb.Image(fig_set),} # (silly) legacy name
            if plot_value: log_dict_plot['val_fn_plot'] = wandb.Image(fig_val)
            wandb.log(log_dict_plot)
        plt.close()

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def validate2(self, epoch, epochs, save_path, x_resolution, y_resolution, z_resolution, time_resolution):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()

        state_test_range = self.dataset.dynamics.state_test_range()
        x_min, x_max = state_test_range[plot_config['x_axis_idx']]
        y_min, y_max = state_test_range[plot_config['y_axis_idx']]
        z_min, z_max = state_test_range[plot_config['z_axis_idx']]

        times = torch.linspace(0, self.dataset.tMax, time_resolution)
        xs = torch.linspace(x_min, x_max, x_resolution)
        ys = torch.linspace(y_min, y_max, y_resolution)
        zs = torch.linspace(z_min, z_max, z_resolution)
        xys = torch.cartesian_prod(xs, ys)

        # Generate separate file names for 3D and 2D plots based on the provided save_path
        save_path_3d = os.path.splitext(save_path)[0] + '_3d.png'  # Add '_3d' before the file extension
        save_path_2d = os.path.splitext(save_path)[0] + '_2d.png'  # Add '_2d' before the file extension

        # Create '2d' and '3d' directories under the save_path if they don't exist
        dir_2d = os.path.join(os.path.dirname(save_path), '2d')
        dir_3d = os.path.join(os.path.dirname(save_path), '3d')
        os.makedirs(dir_2d, exist_ok=True)  # Create the '2d' folder
        os.makedirs(dir_3d, exist_ok=True)  # Create the '3d' folder

        # Create two separate figures: one for 3D plots and one for 2D heatmaps
        fig_3d = plt.figure(figsize=(5*len(times), 5*len(zs)))  # Adjust the figure to have extra column for min plot
        fig_2d = plt.figure(figsize=(5*len(times), 5*len(zs))) # Adjust the figure to have extra column for min plot

        # Array to store minimum values for each x, y pair across all z slices
        min_values = np.full((x_resolution, y_resolution), np.inf)
        

        for i in range(len(times)):
            for j in range(len(zs)):
                coords = torch.zeros(x_resolution*y_resolution, self.dataset.dynamics.state_dim + 1)
                coords[:, 0] = times[i]
                coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                coords[:, 1 + plot_config['z_axis_idx']] = zs[j]

                with torch.no_grad():
                    model_results = self.model({'coords': self.dataset.dynamics.coord_to_input(coords.cuda())})
                    values = self.dataset.dynamics.io_to_value(
                        model_results['model_in'].detach(), 
                        model_results['model_out'].squeeze(dim=-1).detach()
                    )

                # Reshape 'values' to 2D for both 2D and 3D plotting
                values_reshaped = values.detach().cpu().numpy().reshape(x_resolution, y_resolution)

                # Update min_values to track the minimum value across all z-slices for each x, y pair
                min_values = np.minimum(min_values, values_reshaped)

                # 3D Plot
                ax_3d = fig_3d.add_subplot(len(times), len(zs), (j+1) + i*(len(zs)), projection='3d')  # Create a 3D subplot
                X, Y = np.meshgrid(xs, ys)  # Generate grid for x and y
                ax_3d.plot_surface(X, Y, values_reshaped.T, cmap='viridis', edgecolor='none')  # Plot the 3D surface
                ax_3d.set_title(f'3D: t = {times[i]:.2f}, {plot_config["state_labels"][plot_config["z_axis_idx"]]} = {zs[j]:.2f}')
                ax_3d.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))
                ax_3d.set_xlabel('X-axis')
                ax_3d.set_ylabel('Y-axis')
                ax_3d.set_zlabel('Values')

                ax_3d.set_xlim([-1.5, 1.5])
                ax_3d.set_ylim([-1.5, 1.5])
                ax_3d.set_zlim([-1.5, 1.5])

                ax_3d.view_init(elev=20, azim=120)


                # 2D Heatmap Plot
                ax_2d = fig_2d.add_subplot(len(times), len(zs), (j+1) + i*(len(zs)))  # Create a 2D subplot
                s = ax_2d.imshow(1*(values_reshaped.T <= 0), cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))  # Plot the 2D heatmap
                ax_2d.set_title('t = %0.2f, %s = %0.2f' % (times[i], plot_config['state_labels'][plot_config['z_axis_idx']], zs[j]))
                fig_2d.colorbar(s, ax=ax_2d)  # Add colorbar for the 2D plot


                scenario_batch_size = 1
                initial_states = self.initialize_states_with_general_displacement(
                    state_dim=self.dataset.dynamics.state_dim, 
                    theta=zs[j],  # Use the zs tensor for orientations
                    displacement=0.75  # Displace by 0.5
                )


                if epoch == epochs: # and j == 2
                    state_trajs, ctrl_trajs = self.trajectory_rollout(
                        policy=self.model, 
                        dynamics=self.dataset.dynamics,
                        tMin=self.dataset.tMin,
                        tMax=times[i],
                        dt=max(times) / epochs * 10,
                        scenario_batch_size=scenario_batch_size, # not sure what I should set this to... 
                        initial_states=initial_states
                    )

                    # plot rolled-out trajectories on the 2D plot
                    for k in range(state_trajs.shape[0]):
                        x_traj = state_trajs[k, :, plot_config['x_axis_idx']].cpu().numpy() 
                        y_traj = state_trajs[k, :, plot_config['y_axis_idx']].cpu().numpy() 
                        ax_2d.plot(x_traj, y_traj, color='white', lw=1.0, label=f'Trajectory {k + 1}')
                        ax_2d.scatter(x_traj[0], y_traj[0], color='green', s=100, zorder=5, label=f'Start {k+1}' if k == 0 else "")
                        ax_2d.scatter(x_traj[-1], y_traj[-1], color='red', s=100, zorder=5, label=f'End {k+1}' if k == 0 else "")

                        
            # # After processing all z-slices for the current time step, add the minimum z-values plot
            # # Add minimum z-values as the last column for the 3D figure
            # ax_min_3d = fig_3d.add_subplot(len(times), len(zs) + 1, (len(zs)+1) + i*(len(zs)+1), projection='3d')
            # ax_min_3d.plot_surface(X, Y, min_values.T, cmap='plasma', edgecolor='none')
            # ax_min_3d.set_title(f'3D: t = {times[i]:.2f}, {plot_config["state_labels"][plot_config["z_axis_idx"]]} = min')  # \u03B8 is the Unicode for theta (θ)
            # ax_min_3d.set_xlabel('X-axis')
            # ax_min_3d.set_ylabel('Y-axis')
            # ax_min_3d.set_zlabel('Min Values')

            # # Add minimum z-values as the last column for the 2D figure
            # ax_min_2d = fig_2d.add_subplot(len(times), len(zs) + 1, (len(zs)+1) + i*(len(zs)+1))
            # s_min = ax_min_2d.imshow(min_values.T, cmap='plasma', origin='lower', extent=(-1., 1., -1., 1.))
            # ax_min_2d.set_title(f'2D: t = {times[i]:.2f}, {plot_config["state_labels"][plot_config["z_axis_idx"]]} = min')  # \u03B8 is the Unicode for theta (θ)
            # fig_2d.colorbar(s_min, ax=ax_min_2d)

        # Adjust the spacing between subplots (wspace for width, hspace for height)
        fig_3d.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust the spacing for 3D plots
        fig_2d.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust the spacing for 2D plots

        # Save the figures to their respective folders
        fig_3d.savefig(os.path.join(dir_3d, os.path.basename(save_path_3d)))  # Save the 3D plot figure
        fig_2d.savefig(os.path.join(dir_2d, os.path.basename(save_path_2d)))  # Save the 2D heatmap figure

        # Optionally log the figures using wandb if enabled
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                '3D_val_plot': wandb.Image(fig_3d),
                '2D_val_plot': wandb.Image(fig_2d),
            })

        # Close the figures
        plt.close(fig_3d)
        plt.close(fig_2d)

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def train(
            self, batch_size, epochs, lr, 
            steps_til_summary, epochs_til_checkpoint, 
            loss_fn, clip_grad, use_lbfgs, adjust_relative_grads, 
            val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution,
            use_CSL, CSL_lr, CSL_dt, epochs_til_CSL, num_CSL_samples, CSL_loss_frac_cutoff, max_CSL_epochs, CSL_loss_weight, CSL_batch_size
        ):
        was_eval = not self.model.training
        self.model.train()
        self.model.requires_grad_(True)

        train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)

        optim = torch.optim.Adam(lr=lr, params=self.model.parameters())

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=2, T_mult=2, eta_min=1e-9)
        
        # copy settings from Raissi et al. (2019) and here 
        # https://github.com/maziarraissi/PINNs
        if use_lbfgs:
            optim = torch.optim.LBFGS(lr=lr, params=self.model.parameters(), max_iter=50000, max_eval=50000,
                                    history_size=50, line_search_fn='strong_wolfe')

        training_dir = os.path.join(self.experiment_dir, 'training')
        
        summaries_dir = os.path.join(training_dir, 'summaries')
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)

        checkpoints_dir = os.path.join(training_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)

        total_steps = 0

        if adjust_relative_grads:
            new_weight = 1

        with tqdm(total=len(train_dataloader) * epochs) as pbar:
            train_losses = []
            last_CSL_epoch = -1
            for epoch in range(0, epochs):
                if self.dataset.pretrain: # skip CSL
                    last_CSL_epoch = epoch
                time_interval_length = (self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)
                CSL_tMax = self.dataset.tMin + int(time_interval_length/CSL_dt)*CSL_dt
                
                # self-supervised learning
                for step, (model_input, gt) in enumerate(train_dataloader):
                    start_time = time.time()
                
                    model_input = {key: value.cuda() for key, value in model_input.items()}
                    gt = {key: value.cuda() for key, value in gt.items()}

                    model_results = self.model({'coords': model_input['model_coords']})

                    states = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())[..., 1:]
                    values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1))
                    dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))
                    boundary_values = gt['boundary_values']
                    if self.dataset.dynamics.loss_type == 'brat_hjivi':
                        reach_values = gt['reach_values']
                        avoid_values = gt['avoid_values']
                    dirichlet_masks = gt['dirichlet_masks']

                    if self.dataset.dynamics.loss_type == 'brt_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'])
                    elif self.dataset.dynamics.loss_type == 'brat_hjivi':
                        losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, reach_values, avoid_values, dirichlet_masks, model_results['model_out'])
                    else:
                        raise NotImplementedError
                    
                    if use_lbfgs:
                        def closure():
                            optim.zero_grad()
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                train_loss += loss.mean() 
                            train_loss.backward()
                            return train_loss
                        optim.step(closure)

                    # Adjust the relative magnitude of the losses if required
                    if self.dataset.dynamics.deepreach_model in ['vanilla', 'diff'] and adjust_relative_grads:
                        if losses['diff_constraint_hom'] > 0.01:
                            params = OrderedDict(self.model.named_parameters())
                            # Gradients with respect to the PDE loss
                            optim.zero_grad()
                            losses['diff_constraint_hom'].backward(retain_graph=True)
                            grads_PDE = []
                            for key, param in params.items():
                                grads_PDE.append(param.grad.view(-1))
                            grads_PDE = torch.cat(grads_PDE)

                            # Gradients with respect to the boundary loss
                            optim.zero_grad()
                            losses['dirichlet'].backward(retain_graph=True)
                            grads_dirichlet = []
                            for key, param in params.items():
                                grads_dirichlet.append(param.grad.view(-1))
                            grads_dirichlet = torch.cat(grads_dirichlet)

                            num = torch.mean(torch.abs(grads_PDE))
                            den = torch.mean(torch.abs(grads_dirichlet))
                            new_weight = 0.9*new_weight + 0.1*num/den
                            losses['dirichlet'] = new_weight*losses['dirichlet']
                        writer.add_scalar('weight_scaling', new_weight, total_steps) # TODO: make sure this is off

                    # import ipdb; ipdb.set_trace()

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_name == 'dirichlet':
                            writer.add_scalar(loss_name, single_loss/new_weight, total_steps)
                        else:
                            writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                    train_losses.append(train_loss.item())
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                    if not total_steps % steps_til_summary:
                        torch.save(self.model.state_dict(),
                                os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(model, model_input, gt, model_output, writer, total_steps)

                    if not use_lbfgs:
                        optim.zero_grad()
                        train_loss.backward()

                        if clip_grad:
                            if isinstance(clip_grad, bool):
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

                        optim.step()

                    pbar.update(1)

                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                        if self.use_wandb:
                            wandb.log({
                                'step': epoch,
                                'train_loss': train_loss,
                                'pde_loss': losses['diff_constraint_hom'],
                            })
                    
                    total_steps += 1

                scheduler.step()

                if not (epoch+1) % epochs_til_checkpoint:
                    # Saving the optimizer state is important to produce consistent results
                    checkpoint = { 
                        'epoch': epoch+1,
                        'model': self.model.state_dict(),
                        'optimizer': optim.state_dict()}
                    torch.save(checkpoint,
                        os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % (epoch+1)))
                    np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
                        np.array(train_losses))

                    # self.validate(
                    #     epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
                    #     x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)
                    self.validate2(
                        epoch=epoch+1, epochs=epochs, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)), x_resolution = val_x_resolution, y_resolution = val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)


        if was_eval:
            self.model.eval()
            self.model.requires_grad_(False)

    def test(self, current_time, last_checkpoint, checkpoint_dt, dt, num_scenarios, num_violations, set_type, control_type, data_step, checkpoint_toload=None):
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        testing_dir = os.path.join(self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
        if os.path.exists(testing_dir):
            overwrite = input("The testing directory %s already exists. Overwrite? (y/n)"%testing_dir)
            if not (overwrite == 'y'):
                print('Exiting.')
                quit()
            shutil.rmtree(testing_dir)
        os.makedirs(testing_dir)

        if checkpoint_toload is None:
            print('running cross-checkpoint testing')

            for i in tqdm(range(sidelen), desc='Checkpoint'):
                self._load_checkpoint(epoch=checkpoints[i])
                raise NotImplementedError

        else:
            print('running specific-checkpoint testing')
            self._load_checkpoint(checkpoint_toload)

            model = self.model
            dataset = self.dataset
            dynamics = dataset.dynamics
            raise NotImplementedError

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

class DeepReach(Experiment):
    def init_special(self):
        pass