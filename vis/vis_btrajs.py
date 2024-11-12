import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from utils import modules, dataio, losses
from dynamics.dynamics import Dubins3D_P
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.transforms as transforms
import configargparse

p = configargparse.ArgumentParser()
p.add_argument('-p', '--plot', default=False, required=True, help='Plot (T) or (F).')
p.add_argument('-ss', '--seeds', type=int, default=-1, required=False, help='Seeds.')
p.add_argument('-s', '--seed', type=int, default=0, required=False, help='Seed.')
p.add_argument('-t', '--time', type=int, default=1, required=False, help='Time.')
opt = p.parse_args()

# Initialize the dynamics object
dynamics = Dubins3D_P(goalR=0.25, velocity=1.0, omega_max=3.14, angle_alpha_factor=1.2, set_mode='avoid', freeze_model=False)

# Load the trained model
epochs = 280000
model_name = "dubins3d_t1_g1_2"
model_path = f"/home/ubuntu/deepreach_cbvf/runs/{model_name}/training/checkpoints/model_epoch_{epochs}.pth"
checkpoint = torch.load(model_path)

# Instantiate the model
model = modules.SingleBVPNet(
    in_features=dynamics.input_dim,
    out_features=1,
    type='sine',
    mode='mlp',
    final_layer_factor=1,
    hidden_features=512,
    num_hidden_layers=3
)
model.load_state_dict(checkpoint['model'])
model = model.cuda()  # Move model to GPU
model.eval()

# Function to initialize states with general displacement
def initialize_states_with_general_displacement(state_dim, theta, displacement, gamma):
    initial_states = torch.zeros(state_dim)
    
    x_start = -displacement * torch.cos(theta)  # x-coordinate displaced opposite to the orientation
    y_start = -displacement * torch.sin(theta)  # y-coordinate displaced opposite to the orientation

    initial_states[0] = x_start  # x position
    initial_states[1] = y_start  # y position
    initial_states[2] = theta  # orientation angle
    initial_states[3] = gamma

    return initial_states

def sample_boundaries(state_dim, model, gamma): # TODO: rename
    initial_conditions = [] 

    for i in range(200):
        coords = sample_boundary(state_dim, model, gamma) 
        initial_conditions.append(coords) 

    initial_conditions_tensor = torch.stack(initial_conditions, dim=0)

    return initial_conditions_tensor

def sample_boundary(state_dim, model, gamma, epsilon=0.1, seed=0):
    torch.manual_seed(seed)
    
    value = epsilon + 1
    while not (value < epsilon and value > 0):
        with torch.no_grad():
            time = 1
            x = 2 * torch.rand(1) - 1
            y = 2 * torch.rand(1) - 1
            theta = 2 * torch.rand(1) * torch.pi - torch.pi
            
            coords = torch.zeros(1, state_dim + 1)
            # print(coords.shape)
            coords[:, 0] = time
            coords[:, 1] = x
            coords[:, 2] = y
            coords[:, 3] = theta
            coords[:, 4] = gamma

            model_input = dynamics.coord_to_input(coords.cuda())
            model_results = model({'coords': model_input})
            values = dynamics.io_to_value(model_results['model_in'].detach(),
                                            model_results['model_out'].squeeze(dim=-1).detach())
            # print(values)
            value = values.item()
            # print(value)
    return coords

# Function to rollout trajectories
def trajectory_rollout(policy, dynamics, tMin, tMax, dt, scenario_batch_size, initial_states):
    state_trajs = torch.zeros(scenario_batch_size, int((tMax - tMin) / dt) + 1, dynamics.state_dim)
    ctrl_trajs = torch.zeros(scenario_batch_size, int((tMax - tMin) / dt), dynamics.control_dim)

    state_trajs[:, 0, :] = initial_states[:, :, 1:]

    # rollout
    for k in tqdm(range(int((tMax - tMin) / dt)), desc='Trajectory Propagation'):
        traj_time = tMax - k * dt
        traj_times = torch.full((scenario_batch_size,), traj_time)

        # get the states at t=k
        traj_coords = torch.cat((traj_times.unsqueeze(-1), state_trajs[:, k]), dim=-1)

        traj_policy_results = policy({'coords': dynamics.coord_to_input(traj_coords.cuda())})  # learned costate/gradient
        traj_dvs = dynamics.io_to_dv(traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach()

        # optimal control based on the policy's output
        ctrl_trajs[:, k] = dynamics.optimal_control(traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())

        state_trajs[:, k + 1] = dynamics.equivalent_wrapped_state(
            state_trajs[:, k].cuda() + dt * dynamics.dsdt(state_trajs[:, k].cuda(), ctrl_trajs[:, k].cuda()).cuda()
        ).cpu()

    return state_trajs, ctrl_trajs


def visualize_value_function(model, dynamics, save_path, x_resolution=100, y_resolution=100, z_resolution=3, time_resolution=3, seed=0):
    plot_config = dynamics.plot_config()

    state_test_range = dynamics.state_test_range()
    x_min, x_max = state_test_range[plot_config['x_axis_idx']]
    y_min, y_max = state_test_range[plot_config['y_axis_idx']]
    z_min, z_max = state_test_range[plot_config['z_axis_idx']]
    
    times = torch.linspace(0, 1, time_resolution)
    xs = torch.linspace(x_min, x_max, x_resolution)
    ys = torch.linspace(y_min, y_max, y_resolution)
    zs = torch.linspace(z_min, z_max, z_resolution) # theta
    gammas = torch.linspace(0, 1, z_resolution)  # gamma
    xys = torch.cartesian_prod(xs, ys)
    
    
    fig_2d = plt.figure(figsize=(5 * len(times), 5 * len(zs)))
    car_image_path = 'car.png'  # Replace with the path to your car image
    car_image = plt.imread(car_image_path)

    for i, time in enumerate(times):
        for j, gamma in enumerate(gammas):
            coords = torch.zeros(x_resolution * y_resolution, dynamics.state_dim + 1)
            coords[:, 0] = time
            coords[:, 1:-1] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
            coords[:, 1 + plot_config['z_axis_idx']] = 0 # angle
            coords[:, -1] = gamma  # Assign gamma as the last value

            with torch.no_grad():
                model_input = dynamics.coord_to_input(coords.cuda())
                model_results = model({'coords': model_input})
                values = dynamics.io_to_value(model_results['model_in'].detach(),
                                              model_results['model_out'].squeeze(dim=-1).detach())
                
                # print(model_input.shape)
                # print(values.shape)
                # print(f"Value range: min = {values.min().item()}, max = {values.max().item()}")


            # Reshape values for plotting
            values_reshaped = values.cpu().numpy().reshape(x_resolution, y_resolution)

            # 2D Heatmap Plot
            ax_2d = fig_2d.add_subplot(len(times)+1, len(gammas), (j+1) + i * len(gammas))
            s = ax_2d.imshow(1*(values_reshaped.T <= 0), cmap='bwr', origin='lower', extent=[-1, 1, -1, 1])
            ax_2d.set_title(f'2D: t={time:.2f}, gamma={gamma:.2f}')
            fig_2d.colorbar(s, ax=ax_2d)

            ax_2d.contour(xs, ys, values_reshaped.T, levels=[0], colors='black')


            # Initialize states      
            # initial_states = initialize_states_with_general_displacement(
            #     state_dim=dynamics.state_dim, 
            #     theta=torch.tensor(0),  # TODO: theta.item() to make it dynamic
            #     displacement=0.75,
            #     gamma=gamma
            # )

            initial_states = sample_boundary(
                dynamics.state_dim,
                model, 
                gamma, 
                seed=seed
            )
            # print(initial_states.shape)
            # quit()
            
            # Rollout the trajectories
            state_trajs, ctrl_trajs = trajectory_rollout(
                policy=model,
                dynamics=dynamics,
                tMin=0,
                tMax=times[i],
                dt=(max(times) / epochs) * 1000,
                scenario_batch_size=1,
                initial_states=initial_states.unsqueeze(0) 
            )
            # if time == 1:
                # np.save(f"dr_traj_gamma_{gamma}.npy", state_trajs)
                # if gamma == 1: quit()
            
            # Plot the trajectories
            for k in range(state_trajs.shape[0]):
                x_traj = state_trajs[k, :, plot_config['x_axis_idx']].cpu().numpy()
                y_traj = state_trajs[k, :, plot_config['y_axis_idx']].cpu().numpy()
                plt.plot(x_traj, y_traj, lw=2, label=f'Trajectory {k + 1}, θ={0:.2f}')
                plt.scatter(x_traj[0], y_traj[0], color='green', s=50, zorder=5, label='Start' if k == 0 else "")
                plt.scatter(x_traj[-1], y_traj[-1], color='red', s=50, zorder=5, label='End' if k == 0 else "")


    # Save plots
    save_path_2d = os.path.splitext(save_path)[0] + model_name + '/value_function_trajs.png'
    os.makedirs(os.path.dirname(save_path_2d), exist_ok=True)
    
    fig_2d.savefig(save_path_2d)

    plt.close(fig_2d)

# seeds = None
# if opt.seeds == -1: seeds = [opt.seed]
# else: seeds = range(opt.seeds)

# norms_list = []

# true_bcs_min_list = []
# true_bcs_list = []

# neur_bcs_min_list = []
# neur_bcs_list = []

# SAVE_PATH = '/home/ubuntu/deepreach_cbvf/runs/'

# for seed in seeds: 
#     visualize_value_function(model, dynamics, seed=seed)

# Call the visualization function
visualize_value_function(model, dynamics, save_path='/home/ubuntu/deepreach_cbvf/runs/', seed=2)
