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
from scipy.stats import beta as beta_dist
from mpl_toolkits.axes_grid1 import make_axes_locatable

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from collections import OrderedDict
from datetime import datetime
from sklearn import svm 
from utils import diff_operators
from utils.error_evaluators_zy import scenario_optimization, ValueThresholdValidator, MultiValidator, MLPConditionedValidator, target_fraction, MLP, MLPValidator, SliceSampleGenerator
import seaborn as sns
import itertools

class Experiment(ABC):
    def __init__(self, model, dataset, experiment_dir, use_wandb, rollout):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb
        self.N = self.dataset.dynamics.state_dim
        self.rollout = rollout

    @abstractmethod
    def init_special(self):
        raise NotImplementedError

    def _load_checkpoint(self, epoch):
        if epoch == -1:
            model_path = os.path.join(self.experiment_dir, 'training', 'checkpoints', 'model_final.pth')
            self.model.load_state_dict(torch.load(model_path)['model']) # FIXME: manually copied mkims last epoch into a model file
            # self.model.load_state_dict(torch.load(model_path)) # should be this
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
            # TODO: move all to cuda 
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
                # coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                coords[:, 1:-1] = torch.tensor(plot_config['state_slices'])

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

                if self.rollout:
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
            for epoch in range(0, epochs):                
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
                        # losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out']) 
                        # print(states.shape)
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
        if data_step in ["plot_basic_recovery", 'run_basic_recovery', 'plot_hists', 'run_robust_recovery', 'plot_robust_recovery']:
            testing_dir = self.experiment_dir
        else:
            testing_dir = os.path.join(
                self.experiment_dir, 'testing_%s' % current_time.strftime('%m_%d_%Y_%H_%M'))
            if os.path.exists(testing_dir):
                overwrite = input(
                    "The testing directory %s already exists. Overwrite? (y/n)" % testing_dir)
                if not (overwrite == 'y'):
                    print('Exiting.')
                    quit()
                shutil.rmtree(testing_dir)
            os.makedirs(testing_dir)

        if checkpoint_toload is None:
            print('running cross-checkpoint testing')

            # checkpoint x simulation_time square matrices
            sidelen = 10
            assert (last_checkpoint /
                    checkpoint_dt) % sidelen == 0, 'checkpoints cannot be even divided by sidelen'
            BRT_volumes_matrix = np.zeros((sidelen, sidelen))
            BRT_errors_matrix = np.zeros((sidelen, sidelen))
            BRT_error_rates_matrix = np.zeros((sidelen, sidelen))
            BRT_error_region_fracs_matrix = np.zeros((sidelen, sidelen))

            exBRT_volumes_matrix = np.zeros((sidelen, sidelen))
            exBRT_errors_matrix = np.zeros((sidelen, sidelen))
            exBRT_error_rates_matrix = np.zeros((sidelen, sidelen))
            exBRT_error_region_fracs_matrix = np.zeros((sidelen, sidelen))

            checkpoints = np.linspace(0, last_checkpoint, num=sidelen+1)[1:]
            checkpoints[-1] = -1
            times = np.linspace(self.dataset.tMin,
                                self.dataset.tMax, num=sidelen+1)[1:]
            print('constructing matrices for')
            print('checkpoints:', checkpoints)
            print('times:', times)
            for i in tqdm(range(sidelen), desc='Checkpoint'):
                self._load_checkpoint(epoch=checkpoints[i])
                for j in tqdm(range(sidelen), desc='Simulation Time', leave=False):
                    # get BRT volume, error, error rate, error region fraction
                    results = scenario_optimization(
                        model=self.model, dynamics=self.dataset.dynamics, tMin=self.dataset.tMin, t=times[
                            j], dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000),
                        sample_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=0.0),
                        violation_validator=ValueThresholdValidator(
                            v_min=0.0, v_max=float('inf')),
                        max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
                    BRT_volumes_matrix[i, j] = results['valid_sample_fraction']
                    if results['maxed_scenarios']:
                        BRT_errors_matrix[i,
                                          j] = results['max_violation_error']
                        BRT_error_rates_matrix[i,
                                               j] = results['violation_rate']
                        BRT_error_region_fracs_matrix[i, j] = target_fraction(
                            model=self.model, dynamics=self.dataset.dynamics, t=times[j],
                            sample_validator=ValueThresholdValidator(
                                v_min=float('-inf'), v_max=0.0),
                            target_validator=ValueThresholdValidator(
                                v_min=-results['max_violation_error'], v_max=0.0),
                            num_samples=num_scenarios, batch_size=min(10*num_scenarios, 1000000))
                    else:
                        BRT_errors_matrix[i, j] = np.NaN
                        BRT_error_rates_matrix[i, j] = np.NaN
                        BRT_error_region_fracs_matrix[i, j] = np.NaN

                    # get exBRT error, error rate, error region fraction
                    results = scenario_optimization(
                        model=self.model, dynamics=self.dataset.dynamics, tMin=self.dataset.tMin, t=times[
                            j], dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(num_scenarios, 100000), sample_batch_size=min(10*num_scenarios, 1000000),
                        sample_validator=ValueThresholdValidator(
                            v_min=0.0, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=0.0),
                        max_scenarios=num_scenarios, max_samples=1000*num_scenarios)
                    exBRT_volumes_matrix[i,
                                         j] = results['valid_sample_fraction']
                    if results['maxed_scenarios']:
                        exBRT_errors_matrix[i,
                                            j] = results['max_violation_error']
                        exBRT_error_rates_matrix[i,
                                                 j] = results['violation_rate']
                        exBRT_error_region_fracs_matrix[i, j] = target_fraction(
                            model=self.model, dynamics=self.dataset.dynamics, t=times[j],
                            sample_validator=ValueThresholdValidator(
                                v_min=0.0, v_max=float('inf')),
                            target_validator=ValueThresholdValidator(
                                v_min=0.0, v_max=results['max_violation_error']),
                            num_samples=num_scenarios, batch_size=min(10*num_scenarios, 1000000))
                    else:
                        exBRT_errors_matrix[i, j] = np.NaN
                        exBRT_error_rates_matrix[i, j] = np.NaN
                        exBRT_error_region_fracs_matrix[i, j] = np.NaN

            # save the matrices
            matrices = {
                'BRT_volumes_matrix': BRT_volumes_matrix,
                'BRT_errors_matrix': BRT_errors_matrix,
                'BRT_error_rates_matrix': BRT_error_rates_matrix,
                'BRT_error_region_fracs_matrix': BRT_error_region_fracs_matrix,
                'exBRT_volumes_matrix': exBRT_volumes_matrix,
                'exBRT_errors_matrix': exBRT_errors_matrix,
                'exBRT_error_rates_matrix': exBRT_error_rates_matrix,
                'exBRT_error_region_fracs_matrix': exBRT_error_region_fracs_matrix,
            }
            for name, arr in matrices.items():
                with open(os.path.join(testing_dir, f'{name}.npy'), 'wb') as f:
                    np.save(f, arr)

            # plot the matrices
            matrices = {
                'BRT_volumes_matrix': [
                    BRT_volumes_matrix, 'BRT Fractions of Test State Space'
                ],
                'BRT_errors_matrix': [
                    BRT_errors_matrix, 'BRT Errors'
                ],
                'BRT_error_rates_matrix': [
                    BRT_error_rates_matrix, 'BRT Error Rates'
                ],
                'BRT_error_region_fracs_matrix': [
                    BRT_error_region_fracs_matrix, 'BRT Error Region Fractions'
                ],
                'exBRT_volumes_matrix': [
                    exBRT_volumes_matrix, 'exBRT Fractions of Test State Space'
                ],
                'exBRT_errors_matrix': [
                    exBRT_errors_matrix, 'exBRT Errors'
                ],
                'exBRT_error_rates_matrix': [
                    exBRT_error_rates_matrix, 'exBRT Error Rates'
                ],
                'exBRT_error_region_fracs_matrix': [
                    exBRT_error_region_fracs_matrix, 'exBRT Error Region Fractions'
                ],
            }
            for name, data in matrices.items():
                cmap = matplotlib.cm.get_cmap('Reds')
                cmap.set_bad(color='blue')
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks(range(sidelen))
                ax.set_yticks(range(sidelen))
                ax.set_xticklabels(np.round_(times, decimals=2))
                ax.set_yticklabels(np.linspace(
                    0, last_checkpoint, num=sidelen+1)[1:])
                plt.xlabel('Simulation Time')
                plt.ylabel('Checkpoint')
                ax.imshow(data[0], cmap=cmap)
                plt.title(data[1])
                for (y, x), label in np.ndenumerate(data[0]):
                    plt.text(x, y, '%.7f' %
                             label, ha='center', va='center', fontsize=4)
                plt.savefig(os.path.join(testing_dir, name + '.png'), dpi=600)
                plt.clf()
                # log version
                cmap = matplotlib.cm.get_cmap('Reds')
                cmap.set_bad(color='blue')
                fig, ax = plt.subplots(1, 1)
                ax.set_xticks(range(sidelen))
                ax.set_yticks(range(sidelen))
                ax.set_xticklabels(np.round_(times, decimals=2))
                ax.set_yticklabels(np.linspace(
                    0, last_checkpoint, num=sidelen+1)[1:])
                plt.xlabel('Simulation Time')
                plt.ylabel('Checkpoint')
                new_matrix = np.log(data[0])
                ax.imshow(new_matrix, cmap=cmap)
                plt.title('(Log) ' + data[1])
                for (y, x), label in np.ndenumerate(new_matrix):
                    plt.text(x, y, '%.7f' %
                             label, ha='center', va='center', fontsize=4)
                plt.savefig(os.path.join(
                    testing_dir, name + '_log' + '.png'), dpi=600)
                plt.clf()

        else:
            print('running specific-checkpoint testing')
            self._load_checkpoint(checkpoint_toload)

            model = self.model
            dataset = self.dataset
            dynamics = dataset.dynamics

            if data_step == 'plot_violations':
                # plot violations on slice
                plot_config = dynamics.plot_config()
                slices = plot_config['state_slices']
                slices[plot_config['x_axis_idx']] = None
                slices[plot_config['y_axis_idx']] = None
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=100000, sample_batch_size=100000,
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=slices),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=100000, max_samples=1000000)
                plt.title('violations for slice = %s' %
                          plot_config['state_slices'], fontsize=8)
                plt.scatter(results['states'][..., plot_config['x_axis_idx']][~results['violations']], results['states']
                            [...,  plot_config['y_axis_idx']][~results['violations']], s=0.05, color=(0, 0, 1), marker='o')
                plt.scatter(results['states'][..., plot_config['x_axis_idx']][results['violations']], results['states']
                            [...,  plot_config['y_axis_idx']][results['violations']], s=0.05, color=(1, 0, 0), marker='o')
                x_min, x_max = dynamics.state_test_range()[
                    plot_config['x_axis_idx']]
                y_min, y_max = dynamics.state_test_range()[
                    plot_config['y_axis_idx']]
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.savefig(os.path.join(
                    testing_dir, f'violations.png'), dpi=800)
                plt.clf()

                # plot distribution of violations over state variables
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=100000, sample_batch_size=100000,
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=100000, max_samples=1000000)
                for i in range(dynamics.state_dim):
                    plt.title('violations over %s' %
                              plot_config['state_labels'][i])
                    plt.scatter(results['states'][..., i][~results['violations']], results['values']
                                [~results['violations']], s=0.05, color=(0, 0, 1), marker='o')
                    plt.scatter(results['states'][..., i][results['violations']], results['values']
                                [results['violations']], s=0.05, color=(1, 0, 0), marker='o')
                    plt.savefig(os.path.join(
                        testing_dir, f'violations_over_state_dim_{i}.png'), dpi=800)
                    plt.clf()
            if data_step == 'plot_hists':
                logs = {}

                # rollout samples all over the state space
                beta = 1e-16
                epsilon = 1e-3
                N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))
                M = 5

                logs['beta'] = beta
                logs['epsilon'] = epsilon
                logs['N'] = N
                logs['M'] = M

                delta_level = float(
                    'inf') if dynamics.set_mode == 'reach' else float('-inf')

                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=N, max_samples=1000*min(N, 10000))

                sns.set_style('whitegrid')
                costs_ = results['costs'].cpu().numpy()
                values_ = results['values'].cpu().numpy()
                unsafe_cost_safe_value_indeces = np.argwhere(
                    np.logical_and(costs_ < 0, values_ >= 0))
                fig1 = plt.figure()
                positive_values_with_negative_cost = values_[
                    unsafe_cost_safe_value_indeces]
                unsafe_trajs = results['batch_state_trajs'].cpu().numpy()[
                    unsafe_cost_safe_value_indeces, ...]
                outlier_states = results['states'][unsafe_cost_safe_value_indeces, ...]
                torch.set_printoptions(
                    precision=2, threshold=10_000, sci_mode=False)
                print(results['batch_state_trajs'].shape)
                print(outlier_states.shape)
                print(outlier_states[:200, ...])

                (vs, bins, patches) = plt.hist(
                    positive_values_with_negative_cost, bins=200)
                plt.title("Learned values for actually unsafe states marked as safe\n%s" %
                          dynamics.deepReach_model)
                fig1.savefig(os.path.join(
                    testing_dir, f'value distribution.png'), dpi=800)
                print("save path: ", os.path.join(
                    testing_dir, f'value distribution.png'))
                plt.close(fig1)

                # fig2=plt.figure()
                # unsafe_cost_value_indeces=np.argwhere(costs_<0)
                # plt.hist(values_[unsafe_cost_value_indeces],bins=200)
                # plt.title("Learned values for actually unsafe states marked as safe \n %s"%dynamics.deepReach_model)
                # fig2.savefig(os.path.join(testing_dir, f'all value distribution.png'), dpi=800)
                # plt.close(fig2)
                np.save(os.path.join(testing_dir, f'state_traj'),
                        unsafe_trajs)
                np.save(os.path.join(testing_dir, f'safe_traj'),
                        results['batch_state_trajs'].cpu().numpy()[costs_ < 0])

                np.save(os.path.join(testing_dir, f'value_data'),
                        positive_values_with_negative_cost)
                np.save(os.path.join(testing_dir, f'bins'), bins)
                np.save(os.path.join(testing_dir, f'vs'), vs)

            if data_step == 'plot_robust_recovery':
                epsilons=-np.load(os.path.join(testing_dir, f'epsilons.npy'))+1
                deltas=np.load(os.path.join(testing_dir, f'deltas.npy'))
                target_eps=0.01
                delta_level=deltas[np.argmin(np.abs(epsilons-target_eps))]
                fig,values_slices = self.plot_recovery_fig(
                    dataset, dynamics, model, delta_level)
                fig.savefig(os.path.join(
                    testing_dir, f'robust_BRTs_1e-2.png'), dpi=800)
                np.save(os.path.join(testing_dir, f'values_slices'),values_slices)

            if data_step == 'run_robust_recovery':
                logs = {}
                # rollout samples all over the state space
                beta_ = 1e-10
                N = 300000
                logs['beta_'] = beta_
                logs['N'] = N
                delta_level = float(
                    'inf') if dynamics.set_mode == 'reach' else float('-inf')

                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=N, max_samples=1000*min(N, 10000))

                sns.set_style('whitegrid')
                costs_ = results['costs'].cpu().numpy()
                values_ = results['values'].cpu().numpy()
                unsafe_cost_safe_value_indeces = np.argwhere(
                    np.logical_and(costs_ < 0, values_ >= 0))

                print("k max: ", unsafe_cost_safe_value_indeces.shape[0])

                # determine delta_level_max
                delta_level_max = np.max(
                    values_[unsafe_cost_safe_value_indeces])
                print("delta_level_max: ", delta_level_max)

                # for each delta level, determine (1) the corresponding volume;
                # (2) k and and corresponding epsilon
                ks = []
                epsilons = []
                volumes = []

                for delta_level_ in np.arange(0, delta_level_max, delta_level_max/100):
                    k = int(np.argwhere(np.logical_and(
                        costs_ < 0, values_ >= delta_level_)).shape[0])
                    eps = beta_dist.ppf(beta_,  N-k, k+1)
                    volume = values_[values_ >= delta_level_].shape[0]/values_.shape[0]
                    
                    ks.append(k)
                    epsilons.append(eps)
                    volumes.append(volume)

                # plot epsilon volume graph
                fig1, ax1 = plt.subplots()
                color = 'tab:red'
                ax1.set_xlabel('volumes')
                ax1.set_ylabel('epsilons', color=color)
                ax1.plot(volumes, epsilons, color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()

                color = 'tab:blue'
                ax2.set_ylabel('number of outliers', color=color)
                ax2.plot(volumes, ks, color=color)
                ax2.tick_params(axis='y', labelcolor=color)

                plt.title("beta_=1e-10, N =3e6")
                fig1.savefig(os.path.join(
                    testing_dir, f'robust_verification_results.png'), dpi=800)
                plt.close(fig1)
                np.save(os.path.join(testing_dir, f'epsilons'),
                        epsilons)
                np.save(os.path.join(testing_dir, f'volumes'),
                        volumes)
                np.save(os.path.join(testing_dir, f'deltas'),
                        np.arange(0, delta_level_max, delta_level_max/100))
                np.save(os.path.join(testing_dir, f'ks'),
                        ks)
                
            if data_step == 'run_basic_recovery':
                logs = {}

                # 0. explicit statement of probabilistic guarantees, N, \beta, \epsilon
                beta = 1e-16
                epsilon = 1e-3
                N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))
                M = 5

                logs['beta'] = beta
                logs['epsilon'] = epsilon
                logs['N'] = N
                logs['M'] = M

                # 1. execute algorithm for tMax
                # record state/learned_value/violation for each while loop iteration
                delta_level = float(
                    'inf') if dynamics.set_mode == 'reach' else float('-inf')
                algorithm_iters = []
                for i in range(M):
                    print('algorithm iter', str(i))
                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float(
                            '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=N, max_samples=1000*min(N, 10000))
                    if not results['maxed_scenarios']:
                        delta_level = float(
                            '-inf') if dynamics.set_mode == 'reach' else float('inf')
                        break
                    algorithm_iters.append(
                        {
                            'states': results['states'],
                            'values': results['values'],
                            'violations': results['violations']
                        }
                    )
                    if results['violation_rate'] == 0:
                        break
                    violation_levels = results['values'][results['violations']]
                    delta_level_arg = np.argmin(
                        violation_levels) if dynamics.set_mode == 'reach' else np.argmax(violation_levels)
                    delta_level = violation_levels[delta_level_arg].item()

                    print('violation_rate:', str(results['violation_rate']))
                    print('delta_level:', str(delta_level))
                    print('valid_sample_fraction:', str(
                        results['valid_sample_fraction'].item()))
                    sns.set_style('whitegrid')
                    # density_plot=sns.kdeplot(results['costs'].cpu().numpy(), bw=0.5)
                    # density_plot=sns.displot(results['costs'].cpu().numpy(), x="cost function")
                    # fig1 = density_plot.get_figure()
                    fig1 = plt.figure()
                    plt.hist(results['costs'].cpu().numpy(), bins=200)
                    fig1.savefig(os.path.join(
                        testing_dir, f'cost distribution.png'), dpi=800)
                    plt.close(fig1)
                    fig2 = plt.figure()
                    plt.hist(results['costs'].cpu().numpy() -
                             results['values'].cpu().numpy(), bins=200)
                    fig2.savefig(os.path.join(
                        testing_dir, f'diff distribution.png'), dpi=800)
                    plt.close(fig1)

                logs['algorithm_iters'] = algorithm_iters
                logs['delta_level'] = delta_level

                # 2. record solution volume, recovered volume
                S = 1000000
                logs['S'] = S
                logs['learned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=0.0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                logs['recovered_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000)
                ).item()

                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['theoretically_recoverable_volume'] = 1 - \
                        results['violation_rate']
                else:
                    logs['theoretically_recoverable_volume'] = 0

                print('learned_volume', str(logs['learned_volume']))
                print('recovered_volume', str(logs['recovered_volume']))
                print('theoretically_recoverable_volume', str(
                    logs['theoretically_recoverable_volume']))

                # 3. validate theoretical guarantees via mass sampling
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['recovered_violation_rate'] = results['violation_rate']
                else:
                    logs['recovered_violation_rate'] = 0
                print('recovered_violation_rate', str(
                    logs['recovered_violation_rate']))

                with open(os.path.join(testing_dir, 'basic_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'plot_basic_recovery':
                with open(os.path.join(self.experiment_dir, 'basic_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 0.
                print('N:', str(logs['N']))
                print('M:', str(logs['M']))
                print('beta:', str(logs['beta']))
                print('epsilon:', str(logs['epsilon']))
                print('S:', str(logs['S']))
                print('delta level', str(logs['delta_level']))
                delta_level = logs['delta_level']
                print('learned volume', str(logs['learned_volume']))
                print('recovered volume', str(logs['recovered_volume']))
                print('theoretically recoverable volume', str(
                    logs['theoretically_recoverable_volume']))
                print('recovered violation rate', str(
                    logs['recovered_violation_rate']))

                fig = self.plot_recovery_fig(
                    dataset, dynamics, model, delta_level)

                plt.tight_layout()
                fig.savefig(os.path.join(
                    testing_dir, f'basic_BRTs.png'), dpi=800)

            if data_step == 'collect_samples':
                logs = {}

                # 1. record 10M state, learned value, violation
                P = int(1e7)
                logs['P'] = P
                print('collecting training samples')
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(P, 100000), sample_batch_size=10*min(P, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=P, max_samples=1000*min(P, 10000))
                logs['training_samples'] = {
                    'states': results['states'],
                    'values': results['values'],
                    'violations': results['violations'],
                }
                with open(os.path.join(testing_dir, 'sample_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'train_binner':
                with open(os.path.join(self.experiment_dir, 'sample_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 1. train MLP predictor
                # plot validation of MLP predictor
                def validate_predictor(predictor, epoch):
                    print('validating predictor at epoch', str(epoch))
                    predictor.eval()

                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=100000, sample_batch_size=100000,
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(
                            v_min=float('-inf'), v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=100000, max_samples=1000000)

                    inputs = torch.cat(
                        (results['states'], results['values'][..., None]), dim=-1)
                    preds = torch.sigmoid(
                        predictor(inputs.cuda())).detach().cpu().numpy()

                    plt.title(f'Predictor Validation at Epoch {epoch}')
                    plt.ylabel('Value')
                    plt.xlabel('Prediction')
                    plt.scatter(preds[~results['violations']], results['values']
                                [~results['violations']], color='blue', label='nonviolations', alpha=0.1)
                    plt.scatter(preds[results['violations']], results['values']
                                [results['violations']], color='red', label='violations', alpha=0.1)
                    plt.legend()
                    plt.savefig(os.path.join(
                        testing_dir, f'predictor_validation_at_epoch_{epoch}.png'), dpi=800)
                    plt.clf()

                    predictor.train()

                print('training predictor')
                violation_scale = 5
                violation_weight = 1.5
                states = logs['training_samples']['states']
                values = logs['training_samples']['values']
                violations = logs['training_samples']['violations']
                violation_strengths = torch.where(violations, (torch.max(
                    values[violations]) - values) if dynamics.set_mode == 'reach' else (values - torch.min(values[violations])), torch.tensor([0.0])).cuda()
                violation_scales = torch.exp(
                    violation_scale * violation_strengths / torch.max(violation_strengths))

                plt.title(f'Violation Scales')
                plt.ylabel('Frequency')
                plt.xlabel('Scale')
                plt.hist(violation_scales.cpu().numpy(), range=(0, 10))
                plt.savefig(os.path.join(
                    testing_dir, f'violation_scales.png'), dpi=800)
                plt.clf()

                inputs = torch.cat((states, values[..., None]), dim=-1).cuda()
                outputs = 1.0*violations.cuda()
                # outputs = violation_strengths / torch.max(violation_strengths)

                predictor = MLP(input_size=dynamics.state_dim+1)
                predictor.cuda()
                predictor.train()

                lr = 0.00005
                lr_decay = 0.2
                decay_patience = 20
                decay_threshold = 1e-12
                opt = torch.optim.Adam(predictor.parameters(), lr=lr)
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, factor=lr_decay, patience=decay_patience, threshold=decay_threshold)

                pos_weight = violation_weight * \
                    ((outputs <= 0).sum() / (outputs > 0).sum())

                n_epochs = 1000
                batch_size = 100000
                for epoch in range(n_epochs):
                    idxs = torch.randperm(len(outputs))
                    for batch in range(math.ceil(len(outputs) / batch_size)):
                        batch_idxs = idxs[batch *
                                          batch_size: (batch+1)*batch_size]

                        BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss(
                            weight=violation_scales[batch_idxs], pos_weight=pos_weight)
                        loss = BCEWithLogitsLoss(
                            predictor(inputs[batch_idxs]).squeeze(dim=-1), outputs[batch_idxs])
                        # MSELoss = torch.nn.MSELoss()
                        # loss = MSELoss(predictor(inputs[batch_idxs]).squeeze(dim=-1), outputs[batch_idxs])
                        loss.backward()
                        opt.step()
                    print(f'Epoch {epoch}: loss: {loss.item()}')
                    sched.step(loss.item())

                    if (epoch+1) % 100 == 0:
                        torch.save(predictor.state_dict(), os.path.join(
                            testing_dir, f'predictor_at_epoch_{epoch}.pth'))
                        validate_predictor(predictor, epoch)
                logs['violation_scale'] = violation_scale
                logs['violation_weight'] = violation_weight
                logs['n_epochs'] = n_epochs
                logs['lr'] = lr
                logs['lr_decay'] = lr_decay
                logs['decay_patience'] = decay_patience
                logs['decay_threshold'] = decay_threshold

                with open(os.path.join(testing_dir, 'train_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'run_binned_recovery':
                logs = {}

                # 0. explicit statement of probabilistic guarantees, N, \beta, \epsilon
                beta = 1e-16
                epsilon = 1e-3
                N = int(math.ceil((2/epsilon)*(np.log(1/beta)+1)))
                M = 5

                logs['beta'] = beta
                logs['epsilon'] = epsilon
                logs['N'] = N
                logs['M'] = M

                # 1. execute algorithm for each bin of MLP predictor
                epoch = 699
                logs['epoch'] = epoch
                predictor = MLP(input_size=dynamics.state_dim+1)
                predictor.load_state_dict(torch.load(os.path.join(
                    self.experiment_dir, f'predictor_at_epoch_{epoch}.pth')))
                predictor.cuda()
                predictor.train()

                bins = [0, 0.8, 0.85, 0.9, 0.95, 1]
                logs['bins'] = bins

                binned_delta_levels = []
                for i in range(len(bins)-1):
                    print('bin', str(i))
                    binned_delta_level = float(
                        'inf') if dynamics.set_mode == 'reach' else float('-inf')
                    for j in range(M):
                        print('algorithm iter', str(j))
                        results = scenario_optimization(
                            model=model, dynamics=dynamics,
                            tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                            set_type=set_type, control_type=control_type,
                            scenario_batch_size=min(N, 100000), sample_batch_size=10*min(N, 10000),
                            sample_generator=SliceSampleGenerator(
                                dynamics=dynamics, slices=[None]*dynamics.state_dim),
                            sample_validator=MultiValidator([
                                MLPValidator(
                                    mlp=predictor, o_min=bins[i], o_max=bins[i+1], model=model, dynamics=dynamics),
                                ValueThresholdValidator(v_min=float(
                                    '-inf'), v_max=binned_delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=binned_delta_level, v_max=float('inf')),
                            ]),
                            violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                                'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                            max_scenarios=N, max_samples=100000*min(N, 10000))
                        if not results['maxed_scenarios']:
                            binned_delta_level = float(
                                '-inf') if dynamics.set_mode == 'reach' else float('inf')
                            break
                        if results['violation_rate'] == 0:
                            break
                        violation_levels = results['values'][results['violations']]
                        binned_delta_level_arg = np.argmin(
                            violation_levels) if dynamics.set_mode == 'reach' else np.argmax(violation_levels)
                        binned_delta_level = violation_levels[binned_delta_level_arg].item(
                        )
                        print('violation_rate:', str(
                            results['violation_rate']))
                        print('binned_delta_level:', str(binned_delta_level))
                        print('valid_sample_fraction:', str(
                            results['valid_sample_fraction'].item()))
                    binned_delta_levels.append(binned_delta_level)
                logs['binned_delta_levels'] = binned_delta_levels

                # 2. record solution volume, auto-binned recovered volume
                S = 1000000
                logs['S'] = S
                logs['learned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=ValueThresholdValidator(v_min=float(
                        '-inf'), v_max=0.0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                logs['binned_volume'] = target_fraction(
                    model=model, dynamics=dynamics, t=dataset.tMax,
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    target_validator=MLPConditionedValidator(
                        mlp=predictor,
                        o_levels=bins,
                        v_levels=[[float('-inf'), binned_delta_level] if dynamics.set_mode == 'reach' else [
                            binned_delta_level, float('inf')] for binned_delta_level in binned_delta_levels],
                        model=model,
                        dynamics=dynamics,
                    ),
                    num_samples=S,
                    batch_size=min(S, 1000000),
                ).item()
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=ValueThresholdValidator(
                        v_min=float('-inf'), v_max=float('inf')),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['theoretically_recoverable_volume'] = 1 - \
                        results['violation_rate']
                else:
                    logs['theoretically_recoverable_volume'] = 0
                print('learned_volume', str(logs['learned_volume']))
                print('binned_volume', str(logs['binned_volume']))
                print('theoretically_recoverable_volume', str(
                    logs['theoretically_recoverable_volume']))

                # 3. validate theoretical guarantees via mass sampling
                results = scenario_optimization(
                    model=model, dynamics=dynamics,
                    tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                    set_type=set_type, control_type=control_type,
                    scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                    sample_generator=SliceSampleGenerator(
                        dynamics=dynamics, slices=[None]*dynamics.state_dim),
                    sample_validator=MLPConditionedValidator(
                        mlp=predictor,
                        o_levels=bins,
                        v_levels=[[float('-inf'), binned_delta_level] if dynamics.set_mode == 'reach' else [
                            binned_delta_level, float('inf')] for binned_delta_level in binned_delta_levels],
                        model=model,
                        dynamics=dynamics,
                    ),
                    violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                        'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                    max_scenarios=S, max_samples=1000*min(S, 10000))
                if results['maxed_scenarios']:
                    logs['binned_violation_rate'] = results['violation_rate']
                else:
                    logs['binned_violation_rate'] = 0
                print('binned_violation_rate', str(
                    logs['binned_violation_rate']))

                with open(os.path.join(testing_dir, 'binned_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

            if data_step == 'plot_binned_recovery':
                with open(os.path.join(self.experiment_dir, 'binned_logs.pickle'), 'rb') as f:
                    logs = pickle.load(f)

                # 0.
                print('N:', str(logs['N']))
                print('M:', str(logs['M']))
                print('beta:', str(logs['beta']))
                print('epsilon:', str(logs['epsilon']))
                # print('P:', str(logs['P']))
                print('S:', str(logs['S']))
                print('bins', str(logs['bins']))
                bins = logs['bins']
                print('binned delta levels', str(logs['binned_delta_levels']))
                binned_delta_levels = logs['binned_delta_levels']
                print('learned volume', str(logs['learned_volume']))
                print('binned volume', str(logs['binned_volume']))
                print('theoretically recoverable volume', str(
                    logs['theoretically_recoverable_volume']))
                print('binned violation rate', str(
                    logs['binned_violation_rate']))

                epoch = logs['epoch']
                predictor = MLP(input_size=dynamics.state_dim+1)
                predictor.load_state_dict(torch.load(os.path.join(
                    self.experiment_dir, f'predictor_at_epoch_{epoch}.pth')))
                predictor.cuda()
                predictor.eval()

                # 1. for ground truth slices (if available), record (higher-res) grid of learned values and MLP predictions
                # plot (with ground truth) learned BRTs, auto-binned recovered BRTs
                # plot MLP predictor bins
                z_res = 5
                plot_config = dataset.dynamics.plot_config()
                if os.path.exists(os.path.join(self.experiment_dir, 'ground_truth.mat')):
                    ground_truth = spio.loadmat(os.path.join(
                        self.experiment_dir, 'ground_truth.mat'))
                    if 'gmat' in ground_truth:
                        ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
                        ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
                        ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
                        ground_truth_values = ground_truth['data']
                        ground_truth_ts = np.linspace(
                            0, 1, ground_truth_values.shape[3])
                    elif 'g' in ground_truth:
                        ground_truth_xs = ground_truth['g']['vs'][0,
                                                                  0][0][0][:, 0]
                        ground_truth_ys = ground_truth['g']['vs'][0,
                                                                  0][1][0][:, 0]
                        ground_truth_zs = ground_truth['g']['vs'][0,
                                                                  0][2][0][:, 0]
                        ground_truth_ts = ground_truth['tau'][0]
                        ground_truth_values = ground_truth['data']

                    # idxs to plot
                    x_idxs = np.linspace(
                        0, len(ground_truth_xs)-1, len(ground_truth_xs)).astype(dtype=int)
                    y_idxs = np.linspace(
                        0, len(ground_truth_ys)-1, len(ground_truth_ys)).astype(dtype=int)
                    z_idxs = np.linspace(
                        0, len(ground_truth_zs)-1, z_res).astype(dtype=int)
                    t_idxs = np.array(
                        [len(ground_truth_ts)-1]).astype(dtype=int)

                    # indexed ground truth to plot
                    ground_truth_xs = ground_truth_xs[x_idxs]
                    ground_truth_ys = ground_truth_ys[y_idxs]
                    ground_truth_zs = ground_truth_zs[z_idxs]
                    ground_truth_ts = ground_truth_ts[t_idxs]
                    ground_truth_values = ground_truth_values[
                        x_idxs[:, None, None, None],
                        y_idxs[None, :, None, None],
                        z_idxs[None, None, :, None],
                        t_idxs[None, None, None, :]
                    ]
                    ground_truth_grids = ground_truth_values
                    xs = ground_truth_xs
                    ys = ground_truth_ys
                    zs = ground_truth_zs

                else:
                    ground_truth_grids = None
                    resolution = 512
                    xs = np.linspace(*dynamics.state_test_range()
                                     [plot_config['x_axis_idx']], resolution)
                    ys = np.linspace(*dynamics.state_test_range()
                                     [plot_config['y_axis_idx']], resolution)
                    zs = np.linspace(*dynamics.state_test_range()
                                     [plot_config['z_axis_idx']], z_res)

                xys = torch.cartesian_prod(torch.tensor(xs), torch.tensor(ys))
                value_grids = np.zeros((len(zs), len(xs), len(ys)))
                prediction_grids = np.zeros((len(zs), len(xs), len(ys)))
                for i in range(len(zs)):
                    coords = torch.zeros(
                        xys.shape[0], dataset.dynamics.state_dim + 1)
                    coords[:, 0] = dataset.tMax
                    coords[:, 1:] = torch.tensor(plot_config['state_slices'])
                    coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
                    coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
                    coords[:, 1 + plot_config['z_axis_idx']] = zs[i]

                    model_results = model(
                        {'coords': dataset.dynamics.coord_to_input(coords.cuda())})
                    values = dataset.dynamics.io_to_value(model_results['model_in'].detach(
                    ), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
                    value_grids[i] = values.reshape(len(xs), len(ys))

                    inputs = torch.cat(
                        (coords[..., 1:], values[:, None]), dim=-1)
                    outputs = torch.sigmoid(
                        predictor(inputs.cuda()).cpu().squeeze(dim=-1))
                    prediction_grids[i] = outputs.reshape(
                        len(xs), len(ys)).detach().cpu()

                def overlay_ground_truth(image, z_idx):
                    thickness = max(0, image.shape[0] // 120 - 1)
                    ground_truth_grid = ground_truth_grids[:, :, z_idx, 0]
                    ground_truth_BRTs = ground_truth_grid < 0
                    for x in range(ground_truth_BRTs.shape[0]):
                        for y in range(ground_truth_BRTs.shape[1]):
                            if not ground_truth_BRTs[x, y]:
                                continue
                            neighbors = [
                                (x, y+1),
                                (x, y-1),
                                (x+1, y+1),
                                (x+1, y),
                                (x+1, y-1),
                                (x-1, y+1),
                                (x-1, y),
                                (x-1, y-1),
                            ]
                            for neighbor in neighbors:
                                if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_BRTs.shape[0] and neighbor[1] < ground_truth_BRTs.shape[1]:
                                    if not ground_truth_BRTs[neighbor]:
                                        image[x-thickness:x+1+thickness, y-thickness:y +
                                              1+thickness] = np.array([50, 50, 50])
                                        break

                def overlay_border(image, set, color):
                    thickness = max(0, image.shape[0] // 120 - 1)
                    for x in range(set.shape[0]):
                        for y in range(set.shape[1]):
                            if not set[x, y]:
                                continue
                            neighbors = [
                                (x, y+1),
                                (x, y-1),
                                (x+1, y+1),
                                (x+1, y),
                                (x+1, y-1),
                                (x-1, y+1),
                                (x-1, y),
                                (x-1, y-1),
                            ]
                            for neighbor in neighbors:
                                if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < set.shape[0] and neighbor[1] < set.shape[1]:
                                    if not set[neighbor]:
                                        image[x-thickness:x+1, y -
                                              thickness:y+1+thickness] = color
                                        break

                fig = plt.figure()
                fig.suptitle(plot_config['state_slices'], fontsize=8)
                for i in range(len(zs)):
                    values = value_grids[i]
                    predictions = prediction_grids[i]

                    # learned BRT and bin-recovered BRT
                    ax = fig.add_subplot(1, len(zs), (i+1))
                    ax.set_title('%s = %0.2f' % (
                        plot_config['state_labels'][plot_config['z_axis_idx']], zs[i]), fontsize=8)

                    image = np.full((*values.shape, 3), 255, dtype=int)
                    BRT = values < 0
                    bin_recovered_BRT = np.full(values.shape, 0, dtype=bool)
                    # per bin, set accordingly
                    for j in range(len(bins)-1):
                        mask = (
                            (predictions >= bins[j])*(predictions < bins[j+1]))
                        binned_delta_level = binned_delta_levels[j]
                        bin_recovered_BRT[mask *
                                          (values < binned_delta_level)] = True

                    if dynamics.set_mode == 'reach':
                        image[BRT] = np.array([252, 227, 152])
                        overlay_border(image, BRT, np.array([249, 188, 6]))
                        image[bin_recovered_BRT] = np.array([155, 241, 249])
                        overlay_border(image, bin_recovered_BRT,
                                       np.array([15, 223, 240]))
                        if ground_truth_grids is not None:
                            overlay_ground_truth(image, i)
                    else:
                        image[bin_recovered_BRT] = np.array([155, 241, 249])
                        image[BRT] = np.array([252, 227, 152])
                        overlay_border(image, BRT, np.array([249, 188, 6]))
                        # overlay recovered border over learned BRT
                        overlay_border(image, bin_recovered_BRT,
                                       np.array([15, 223, 240]))
                        if ground_truth_grids is not None:
                            overlay_ground_truth(image, i)

                    ax.imshow(image.transpose(1, 0, 2),
                              origin='lower', extent=(-1., 1., -1., 1.))
                    ax.set_xlabel(
                        plot_config['state_labels'][plot_config['x_axis_idx']])
                    ax.set_ylabel(
                        plot_config['state_labels'][plot_config['y_axis_idx']])
                    ax.set_xticks([-1, 1])
                    ax.set_yticks([-1, 1])
                    ax.tick_params(labelsize=6)
                    if i != 0:
                        ax.set_yticks([])
                plt.tight_layout()
                fig.savefig(os.path.join(
                    testing_dir, f'binned_BRTs.png'), dpi=800)

            if data_step == 'plot_cost_function':
                if os.path.exists(os.path.join(self.experiment_dir, 'cost_logs.pickle')):
                    with open(os.path.join(self.experiment_dir, 'cost_logs.pickle'), 'rb') as f:
                        logs = pickle.load(f)

                else:
                    with open(os.path.join(self.experiment_dir, 'basic_logs.pickle'), 'rb') as f:
                        logs = pickle.load(f)

                    S = logs['S']
                    delta_level = logs['delta_level']
                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float(
                            '-inf'), v_max=0.0) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=0.0, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=S, max_samples=1000*min(S, 10000))
                    if results['maxed_scenarios']:
                        logs['learned_costs'] = results['costs']
                    else:
                        logs['learned_costs'] = None

                    results = scenario_optimization(
                        model=model, dynamics=dynamics,
                        tMin=dataset.tMin, tMax=dataset.tMax, dt=dt,
                        set_type=set_type, control_type=control_type,
                        scenario_batch_size=min(S, 100000), sample_batch_size=10*min(S, 10000),
                        sample_generator=SliceSampleGenerator(
                            dynamics=dynamics, slices=[None]*dynamics.state_dim),
                        sample_validator=ValueThresholdValidator(v_min=float(
                            '-inf'), v_max=delta_level) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=delta_level, v_max=float('inf')),
                        violation_validator=ValueThresholdValidator(v_min=0.0, v_max=float(
                            'inf')) if dynamics.set_mode == 'reach' else ValueThresholdValidator(v_min=float('-inf'), v_max=0.0),
                        max_scenarios=S, max_samples=1000*min(S, 10000))
                    if results['maxed_scenarios']:
                        logs['recovered_costs'] = results['costs']
                    else:
                        logs['recovered_costs'] = None

                if logs['learned_costs'] is not None and logs['recovered_costs'] is not None:
                    plt.title(f'Trajectory Costs')
                    plt.ylabel('Frequency')
                    plt.xlabel('Cost')
                    plt.hist(logs['learned_costs'], color=(
                        247/255, 187/255, 8/255), alpha=0.5)
                    plt.hist(logs['recovered_costs'], color=(
                        14/255, 222/255, 241/255), alpha=0.5)
                    plt.axvline(x=0, linestyle='--', color='black')
                    plt.savefig(os.path.join(
                        testing_dir, f'cost_function.png'), dpi=800)

                with open(os.path.join(testing_dir, 'cost_logs.pickle'), 'wb') as f:
                    pickle.dump(logs, f)

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def plot_recovery_fig(self, dataset, dynamics, model, delta_level):
        # 1. for ground truth slices (if available), record (higher-res) grid of learned values
        # plot (with ground truth) learned BRTs, recovered BRTs
        z_res = 5
        plot_config = dataset.dynamics.plot_config()
        print(dataset.dynamics)
        if os.path.exists(os.path.join(self.experiment_dir, 'ground_truth.mat')):
            ground_truth = spio.loadmat(os.path.join(
                self.experiment_dir, 'ground_truth.mat'))
            if 'gmat' in ground_truth:
                ground_truth_xs = ground_truth['gmat'][..., 0][:, 0, 0]
                ground_truth_ys = ground_truth['gmat'][..., 1][0, :, 0]
                ground_truth_zs = ground_truth['gmat'][..., 2][0, 0, :]
                ground_truth_values = ground_truth['data']
                ground_truth_ts = np.linspace(
                    0, 1, ground_truth_values.shape[3])

            elif 'g' in ground_truth:
                ground_truth_xs = ground_truth['g']['vs'][0, 0][0][0][:, 0]
                ground_truth_ys = ground_truth['g']['vs'][0, 0][1][0][:, 0]
                ground_truth_zs = ground_truth['g']['vs'][0, 0][2][0][:, 0]
                ground_truth_ts = ground_truth['tau'][0]
                ground_truth_values = ground_truth['data']

            # idxs to plot
            x_idxs = np.linspace(0, len(ground_truth_xs)-1,
                                 len(ground_truth_xs)).astype(dtype=int)
            y_idxs = np.linspace(0, len(ground_truth_ys)-1,
                                 len(ground_truth_ys)).astype(dtype=int)
            z_idxs = np.linspace(0, len(ground_truth_zs) -
                                 1, z_res).astype(dtype=int)
            t_idxs = np.array([len(ground_truth_ts)-1]).astype(dtype=int)

            # indexed ground truth to plot
            ground_truth_xs = ground_truth_xs[x_idxs]
            ground_truth_ys = ground_truth_ys[y_idxs]
            ground_truth_zs = ground_truth_zs[z_idxs]
            ground_truth_ts = ground_truth_ts[t_idxs]
            ground_truth_values = ground_truth_values[
                x_idxs[:, None, None, None],
                y_idxs[None, :, None, None],
                z_idxs[None, None, :, None],
                t_idxs[None, None, None, :]
            ]
            ground_truth_grids = ground_truth_values

            xs = ground_truth_xs
            ys = ground_truth_ys
            zs = ground_truth_zs
        else:
            ground_truth_grids = None
            resolution = 512
            xs = np.linspace(*dynamics.state_test_range()
                             [plot_config['x_axis_idx']], resolution)
            ys = np.linspace(*dynamics.state_test_range()
                             [plot_config['y_axis_idx']], resolution)
            zs = np.linspace(*dynamics.state_test_range()
                             [plot_config['z_axis_idx']], z_res)

        xys = torch.cartesian_prod(torch.tensor(xs), torch.tensor(ys))
        value_grids = np.zeros((len(zs), len(xs), len(ys)))
        for i in range(len(zs)):
            coords = torch.zeros(xys.shape[0], dataset.dynamics.state_dim + 1)
            coords[:, 0] = dataset.tMax
            print(torch.tensor(plot_config['state_slices']).shape)
            coords[:, 1:] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
            if dataset.dynamics.state_dim > 2:
                coords[:, 1 + plot_config['z_axis_idx']] = zs[i]

            model_results = model(
                {'coords': dataset.dynamics.coord_to_input(coords.cuda())})
            values = dataset.dynamics.io_to_value(model_results['model_in'].detach(
            ), model_results['model_out'].detach().squeeze(dim=-1)).detach().cpu()
            value_grids[i] = values.reshape(len(xs), len(ys))

        fig = plt.figure()
        fig.suptitle(plot_config['state_slices'], fontsize=8)
        x_min, x_max = dataset.dynamics.state_test_range()[
            plot_config['x_axis_idx']]
        y_min, y_max = dataset.dynamics.state_test_range()[
            plot_config['y_axis_idx']]

        for i in range(len(zs)):
            values = value_grids[i]

            # learned BRT and recovered BRT
            ax = fig.add_subplot(1, len(zs), (i+1))
            ax.set_title('%s = %0.2f' % (
                plot_config['state_labels'][plot_config['z_axis_idx']], zs[i]), fontsize=8)

            image = np.full((*values.shape, 3), 255, dtype=int)
            BRT = values < 0
            recovered_BRT = values < delta_level

            if dynamics.set_mode == 'reach':
                image[BRT] = np.array([252, 227, 152])
                self.overlay_border(image, BRT, np.array([249, 188, 6]))
                image[recovered_BRT] = np.array([155, 241, 249])
                self.overlay_border(image, recovered_BRT,
                                    np.array([15, 223, 240]))
                if ground_truth_grids is not None:
                    self.overlay_ground_truth(image, i, ground_truth_grids)
            else:
                image[recovered_BRT] = np.array([155, 241, 249])
                image[BRT] = np.array([252, 227, 152])
                self.overlay_border(image, BRT, np.array([249, 188, 6]))
                # overlay recovered border over learned BRT
                self.overlay_border(image, recovered_BRT,
                                    np.array([15, 223, 240]))
                if ground_truth_grids is not None:
                    self.overlay_ground_truth(image, i, ground_truth_grids)

            ax.imshow(image.transpose(1, 0, 2), origin='lower',
                      extent=(x_min, x_max, y_min, y_max))

            ax.set_xlabel(plot_config['state_labels']
                          [plot_config['x_axis_idx']])
            ax.set_ylabel(plot_config['state_labels']
                          [plot_config['y_axis_idx']])
            ax.set_xticks([x_min, x_max])
            ax.set_yticks([y_min, y_max])
            ax.tick_params(labelsize=6)
            if i != 0:
                ax.set_yticks([])
        return fig, value_grids

    def overlay_ground_truth(self, image, z_idx, ground_truth_grids):
        thickness = max(0, image.shape[0] // 120 - 1)
        ground_truth_grid = ground_truth_grids[:, :, z_idx, 0]
        ground_truth_brts = ground_truth_grid < 0
        for x in range(ground_truth_brts.shape[0]):
            for y in range(ground_truth_brts.shape[1]):
                if not ground_truth_brts[x, y]:
                    continue
                neighbors = [
                    (x, y+1),
                    (x, y-1),
                    (x+1, y+1),
                    (x+1, y),
                    (x+1, y-1),
                    (x-1, y+1),
                    (x-1, y),
                    (x-1, y-1),
                ]
                for neighbor in neighbors:
                    if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < ground_truth_brts.shape[0] and neighbor[1] < ground_truth_brts.shape[1]:
                        if not ground_truth_brts[neighbor]:
                            image[x-thickness:x+1, y-thickness:y +
                                  1+thickness] = np.array([50, 50, 50])
                            break

    def overlay_border(self, image, set, color):
        thickness = max(0, image.shape[0] // 120 - 1)
        for x in range(set.shape[0]):
            for y in range(set.shape[1]):
                if not set[x, y]:
                    continue
                neighbors = [
                    (x, y+1),
                    (x, y-1),
                    (x+1, y+1),
                    (x+1, y),
                    (x+1, y-1),
                    (x-1, y+1),
                    (x-1, y),
                    (x-1, y-1),
                ]
                for neighbor in neighbors:
                    if neighbor[0] >= 0 and neighbor[1] >= 0 and neighbor[0] < set.shape[0] and neighbor[1] < set.shape[1]:
                        if not set[neighbor]:
                            image[x-thickness:x+1, y -
                                  thickness:y+1+thickness] = color
                            break

    # def post_training_CSL(self,loss_fn, clip_grad, use_lbfgs,
    #     val_x_resolution, val_y_resolution, val_z_resolution, val_time_resolution,
    #     use_CSL, CSL_dt, epochs_til_CSL, num_CSL_samples, CSL_loss_frac_cutoff, max_CSL_epochs, CSL_loss_weight, CSL_batch_size):

    #             last_CSL_epoch = epoch

    #             # generate CSL datasets
    #             self.model.eval()

    #             CSL_dataset = scenario_optimization(
    #                 model=self.model, dynamics=self.dataset.dynamics,
    #                 tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
    #                 set_type="BRT", control_type="value",  # TODO: implement option for BRS too
    #                 scenario_batch_size=min(num_CSL_samples, 100000), sample_batch_size=min(10*num_CSL_samples, 1000000),
    #                 sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[
    #                                                         None]*self.dataset.dynamics.state_dim),
    #                 sample_validator=ValueThresholdValidator(
    #                     v_min=float('-inf'), v_max=float('inf')),
    #                 violation_validator=ValueThresholdValidator(
    #                     v_min=0.0, v_max=0.0),
    #                 max_scenarios=num_CSL_samples, tStart_generator=lambda n: torch.zeros(
    #                     n).uniform_(self.dataset.tMin, CSL_tMax)
    #             )
    #             CSL_coords = torch.cat(
    #                 (CSL_dataset['times'].unsqueeze(-1), CSL_dataset['states']), dim=-1)
    #             CSL_costs = CSL_dataset['costs']

    #             num_CSL_val_samples = int(0.1*num_CSL_samples)
    #             CSL_val_dataset = scenario_optimization(
    #                 model=self.model, dynamics=self.dataset.dynamics,
    #                 tMin=self.dataset.tMin, tMax=CSL_tMax, dt=CSL_dt,
    #                 set_type="BRT", control_type="value",  # TODO: implement option for BRS too
    #                 scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
    #                 sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[
    #                                                         None]*self.dataset.dynamics.state_dim),
    #                 sample_validator=ValueThresholdValidator(
    #                     v_min=float('-inf'), v_max=float('inf')),
    #                 violation_validator=ValueThresholdValidator(
    #                     v_min=0.0, v_max=0.0),
    #                 max_scenarios=num_CSL_val_samples, tStart_generator=lambda n: torch.zeros(
    #                     n).uniform_(self.dataset.tMin, CSL_tMax)
    #             )
    #             CSL_val_coords = torch.cat(
    #                 (CSL_val_dataset['times'].unsqueeze(-1), CSL_val_dataset['states']), dim=-1)
    #             CSL_val_costs = CSL_val_dataset['costs']

    #             CSL_val_tMax_dataset = scenario_optimization(
    #                 model=self.model, dynamics=self.dataset.dynamics,
    #                 tMin=self.dataset.tMin, tMax=self.dataset.tMax, dt=CSL_dt,
    #                 set_type="BRT", control_type="value",  # TODO: implement option for BRS too
    #                 scenario_batch_size=min(num_CSL_val_samples, 100000), sample_batch_size=min(10*num_CSL_val_samples, 1000000),
    #                 sample_generator=SliceSampleGenerator(dynamics=self.dataset.dynamics, slices=[
    #                                                         None]*self.dataset.dynamics.state_dim),
    #                 sample_validator=ValueThresholdValidator(
    #                     v_min=float('-inf'), v_max=float('inf')),
    #                 violation_validator=ValueThresholdValidator(
    #                     v_min=0.0, v_max=0.0),
    #                 # no tStart_generator, since I want all tMax times
    #                 max_scenarios=num_CSL_val_samples
    #             )
    #             CSL_val_tMax_coords = torch.cat(
    #                 (CSL_val_tMax_dataset['times'].unsqueeze(-1), CSL_val_tMax_dataset['states']), dim=-1)
    #             CSL_val_tMax_costs = CSL_val_tMax_dataset['costs']

    #             self.model.train()

    #             # CSL optimizer
    #             CSL_optim = torch.optim.Adam(
    #                 lr=csl_lr, params=self.model.parameters())

    #             # initial CSL val loss
    #             CSL_val_results = self.model(
    #                 {'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.cuda())})
    #             CSL_val_preds = self.dataset.dynamics.io_to_value(
    #                 CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
    #             CSL_val_errors = CSL_val_preds - CSL_val_costs.cuda()
    #             CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))
    #             CSL_initial_val_loss = CSL_val_loss
    #             wandb.log({
    #                 "step": epoch,
    #                 "CSL_val_loss": CSL_val_loss.item()
    #             })

    #             # initial self-supervised learning (SSL) val loss
    #             # right now, just took code from dataio.py and the SSL training loop above; TODO: refactor all this for cleaner modular code
    #             CSL_val_states = CSL_val_coords[..., 1:].cuda()
    #             CSL_val_dvs = self.dataset.dynamics.io_to_dv(
    #                 CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
    #             CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(
    #                 CSL_val_states)
    #             if self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                 CSL_val_reach_values = self.dataset.dynamics.reach_fn(
    #                     CSL_val_states)
    #                 CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(
    #                     CSL_val_states)
    #             # assumes time unit in dataset (model) is same as real time units
    #             CSL_val_dirichlet_masks = CSL_val_coords[:, 0].cuda(
    #             ) == self.dataset.tMin
    #             if self.dataset.dynamics.loss_type == 'brt_hjivi':
    #                 SSL_val_losses = loss_fn(
    #                     CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
    #             elif self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                 SSL_val_losses = loss_fn(CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:],
    #                                             CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
    #             else:
    #                 NotImplementedError
    #             # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
    #             SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean()
    #             wandb.log({
    #                 "step": epoch,
    #                 "SSL_val_loss": SSL_val_loss.item()
    #             })

    #             # CSL training loop
    #             for CSL_epoch in tqdm(range(max_CSL_epochs)):
    #                 CSL_idxs = torch.randperm(num_CSL_samples)
    #                 for CSL_batch in range(math.ceil(num_CSL_samples/CSL_batch_size)):
    #                     CSL_batch_idxs = CSL_idxs[CSL_batch *
    #                                                 CSL_batch_size:(CSL_batch+1)*CSL_batch_size]
    #                     CSL_batch_coords = CSL_coords[CSL_batch_idxs]

    #                     CSL_batch_results = self.model(
    #                         {'coords': self.dataset.dynamics.coord_to_input(CSL_batch_coords.cuda())})
    #                     CSL_batch_preds = self.dataset.dynamics.io_to_value(
    #                         CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
    #                     CSL_batch_costs = CSL_costs[CSL_batch_idxs].cuda()
    #                     CSL_batch_errors = CSL_batch_preds - CSL_batch_costs
    #                     CSL_batch_loss = CSL_loss_weight * \
    #                         torch.mean(torch.pow(CSL_batch_errors, 2))

    #                     CSL_batch_states = CSL_batch_coords[..., 1:].cuda()
    #                     CSL_batch_dvs = self.dataset.dynamics.io_to_dv(
    #                         CSL_batch_results['model_in'], CSL_batch_results['model_out'].squeeze(dim=-1))
    #                     CSL_batch_boundary_values = self.dataset.dynamics.boundary_fn(
    #                         CSL_batch_states)
    #                     if self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                         CSL_batch_reach_values = self.dataset.dynamics.reach_fn(
    #                             CSL_batch_states)
    #                         CSL_batch_avoid_values = self.dataset.dynamics.avoid_fn(
    #                             CSL_batch_states)
    #                     # assumes time unit in dataset (model) is same as real time units
    #                     CSL_batch_dirichlet_masks = CSL_batch_coords[:, 0].cuda(
    #                     ) == self.dataset.tMin
    #                     if self.dataset.dynamics.loss_type == 'brt_hjivi':
    #                         SSL_batch_losses = loss_fn(
    #                             CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_dirichlet_masks)
    #                     elif self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                         SSL_batch_losses = loss_fn(
    #                             CSL_batch_states, CSL_batch_preds, CSL_batch_dvs[..., 0], CSL_batch_dvs[..., 1:], CSL_batch_boundary_values, CSL_batch_reach_values, CSL_batch_avoid_values, CSL_batch_dirichlet_masks)
    #                     else:
    #                         NotImplementedError
    #                     # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_batch_dirichlet_masks == False))
    #                     SSL_batch_loss = SSL_batch_losses['diff_constraint_hom'].mean(
    #                     )

    #                     CSL_optim.zero_grad()
    #                     SSL_batch_loss.backward(retain_graph=True)
    #                     # no adjust_relative_grads, because I assume even with adjustment, the diff_constraint_hom remains unaffected and the only other loss (dirichlet) is zero
    #                     if (not use_lbfgs) and clip_grad:
    #                         if isinstance(clip_grad, bool):
    #                             torch.nn.utils.clip_grad_norm_(
    #                                 self.model.parameters(), max_norm=1.)
    #                         else:
    #                             torch.nn.utils.clip_grad_norm_(
    #                                 self.model.parameters(), max_norm=clip_grad)
    #                     CSL_batch_loss.backward()
    #                     CSL_optim.step()

    #                 # evaluate on CSL_val_dataset
    #                 CSL_val_results = self.model(
    #                     {'coords': self.dataset.dynamics.coord_to_input(CSL_val_coords.cuda())})
    #                 CSL_val_preds = self.dataset.dynamics.io_to_value(
    #                     CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
    #                 CSL_val_errors = CSL_val_preds - CSL_val_costs.cuda()
    #                 CSL_val_loss = torch.mean(torch.pow(CSL_val_errors, 2))

    #                 CSL_val_states = CSL_val_coords[..., 1:].cuda()
    #                 CSL_val_dvs = self.dataset.dynamics.io_to_dv(
    #                     CSL_val_results['model_in'], CSL_val_results['model_out'].squeeze(dim=-1))
    #                 CSL_val_boundary_values = self.dataset.dynamics.boundary_fn(
    #                     CSL_val_states)
    #                 if self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                     CSL_val_reach_values = self.dataset.dynamics.reach_fn(
    #                         CSL_val_states)
    #                     CSL_val_avoid_values = self.dataset.dynamics.avoid_fn(
    #                         CSL_val_states)
    #                 # assumes time unit in dataset (model) is same as real time units
    #                 CSL_val_dirichlet_masks = CSL_val_coords[:, 0].cuda(
    #                 ) == self.dataset.tMin
    #                 if self.dataset.dynamics.loss_type == 'brt_hjivi':
    #                     SSL_val_losses = loss_fn(
    #                         CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_dirichlet_masks)
    #                 elif self.dataset.dynamics.loss_type == 'brat_hjivi':
    #                     SSL_val_losses = loss_fn(
    #                         CSL_val_states, CSL_val_preds, CSL_val_dvs[..., 0], CSL_val_dvs[..., 1:], CSL_val_boundary_values, CSL_val_reach_values, CSL_val_avoid_values, CSL_val_dirichlet_masks)
    #                 else:
    #                     raise NotImplementedError
    #                 # I assume there is no dirichlet (boundary) loss here, because I do not ever explicitly generate source samples at tMin (i.e. torch.all(CSL_val_dirichlet_masks == False))
    #                 SSL_val_loss = SSL_val_losses['diff_constraint_hom'].mean(
    #                 )

    #                 CSL_val_tMax_results = self.model(
    #                     {'coords': self.dataset.dynamics.coord_to_input(CSL_val_tMax_coords.cuda())})
    #                 CSL_val_tMax_preds = self.dataset.dynamics.io_to_value(
    #                     CSL_val_tMax_results['model_in'], CSL_val_tMax_results['model_out'].squeeze(dim=-1))
    #                 CSL_val_tMax_errors = CSL_val_tMax_preds - CSL_val_tMax_costs.cuda()
    #                 CSL_val_tMax_loss = torch.mean(
    #                     torch.pow(CSL_val_tMax_errors, 2))

    #                 # log CSL losses, recovered_safe_set_fracs
    #                 if self.dataset.dynamics.set_mode == 'reach':
    #                     CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_batch_costs.cuda() < 0) / len(CSL_batch_preds)
    #                     if len(CSL_batch_preds[CSL_batch_costs.cuda() > 0]) > 0:
    #                         CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds < torch.min(
    #                             CSL_batch_preds[CSL_batch_costs.cuda() > 0])) / len(CSL_batch_preds)
    #                     else:
    #                         CSL_train_batch_recovered_safe_set_frac = torch.FloatTensor(
    #                             1)
    #                     CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_val_costs.cuda() < 0) / len(CSL_val_preds)
    #                     CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds < torch.min(
    #                         CSL_val_preds[CSL_val_costs.cuda() > 0])) / len(CSL_val_preds)
    #                     CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_val_tMax_costs.cuda() < 0) / len(CSL_val_tMax_preds)
    #                     CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds < torch.min(
    #                         CSL_val_tMax_preds[CSL_val_tMax_costs.cuda() > 0])) / len(CSL_val_tMax_preds)
    #                 elif self.dataset.dynamics.set_mode == 'avoid':
    #                     CSL_train_batch_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_batch_costs.cuda() > 0) / len(CSL_batch_preds)
    #                     # if len(CSL_batch_preds[CSL_batch_costs.cuda() < 0])>0:
    #                     #     CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds > torch.max(CSL_batch_preds[CSL_batch_costs.cuda() < 0])) / len(CSL_batch_preds)
    #                     # else:
    #                     #     CSL_train_batch_recovered_safe_set_frac=torch.FloatTensor(1)
    #                     CSL_train_batch_recovered_safe_set_frac = torch.sum(CSL_batch_preds > torch.max(
    #                         CSL_batch_preds[CSL_batch_costs.cuda() < 0])) / len(CSL_batch_preds)

    #                     CSL_val_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_val_costs.cuda() > 0) / len(CSL_val_preds)
    #                     CSL_val_recovered_safe_set_frac = torch.sum(CSL_val_preds > torch.max(
    #                         CSL_val_preds[CSL_val_costs.cuda() < 0])) / len(CSL_val_preds)
    #                     CSL_val_tMax_theoretically_recoverable_safe_set_frac = torch.sum(
    #                         CSL_val_tMax_costs.cuda() > 0) / len(CSL_val_tMax_preds)
    #                     CSL_val_tMax_recovered_safe_set_frac = torch.sum(CSL_val_tMax_preds > torch.max(
    #                         CSL_val_tMax_preds[CSL_val_tMax_costs.cuda() < 0])) / len(CSL_val_tMax_preds)

    #                     wandb.log({
    #                         "step": epoch+(CSL_epoch+1)*int(0.5*epochs_til_CSL/max_CSL_epochs),
    #                         "CSL_train_batch_loss": CSL_batch_loss.item(),
    #                         "SSL_train_batch_loss": SSL_batch_loss.item(),
    #                         "CSL_val_loss": CSL_val_loss.item(),
    #                         "SSL_val_loss": SSL_val_loss.item(),
    #                         "CSL_val_tMax_loss": CSL_val_tMax_loss.item(),
    #                         "CSL_train_batch_theoretically_recoverable_safe_set_frac": CSL_train_batch_theoretically_recoverable_safe_set_frac.item(),
    #                         "CSL_val_theoretically_recoverable_safe_set_frac": CSL_val_theoretically_recoverable_safe_set_frac.item(),
    #                         "CSL_val_tMax_theoretically_recoverable_safe_set_frac": CSL_val_tMax_theoretically_recoverable_safe_set_frac.item(),
    #                         "CSL_train_batch_recovered_safe_set_frac": CSL_train_batch_recovered_safe_set_frac.item(),
    #                         "CSL_val_recovered_safe_set_frac": CSL_val_recovered_safe_set_frac.item(),
    #                         "CSL_val_tMax_recovered_safe_set_frac": CSL_val_tMax_recovered_safe_set_frac.item(),
    #                     })

    #             if not (epoch+1) % epochs_til_checkpoint:
    #                 # Saving the optimizer state is important to produce consistent results
    #                 checkpoint = {
    #                     'epoch': epoch+1,
    #                     'model': self.model.state_dict(),
    #                     'optimizer': optim.state_dict()}
    #                 torch.save(checkpoint,
    #                            os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % (epoch+1)))
    #                 np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % (epoch+1)),
    #                            np.array(train_losses))
    #                 self.validate(
    #                     epoch=epoch+1, save_path=os.path.join(checkpoints_dir, 'BRS_validation_plot_epoch_%04d.png' % (epoch+1)),
    #                     x_resolution=val_x_resolution, y_resolution=val_y_resolution, z_resolution=val_z_resolution, time_resolution=val_time_resolution)

    #     torch.save(checkpoint, os.path.join(
    #         checkpoints_dir, 'model_CSL.pth'))


class DeepReach(Experiment):
    def init_special(self):
        pass