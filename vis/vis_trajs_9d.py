import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from utils import modules, dataio, losses
from dynamics.dynamics import Dubins3D_P, MultiVehicleCollision_P
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.transforms as transforms
import configargparse


p = configargparse.ArgumentParser()
p.add_argument('-r', '--radius', type=float, default=1.0, required=True, help='Radius of circle.')
p.add_argument('-p', '--plot', default=False, required=True, help='Plot (T) or (F).')
p.add_argument('-ss', '--seeds', type=int, default=-1, required=False, help='Seeds.')
p.add_argument('-s', '--seed', type=int, default=0, required=False, help='Seed.')
p.add_argument('-t', '--time', type=int, default=1, required=False, help='Time.')
p.add_argument('-g', '--gamma', type=float, default=1, required=False, help='Gamma.')

opt = p.parse_args()

# Initialize the dynamics object
dynamics = MultiVehicleCollision_P()

# Load the trained model
epochs = 500000
tMax = opt.time
gammaMax = opt.gamma
model_name = f"mvc_t{tMax}_g{gammaMax}"
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


def initialize_states_circle(state_dim, radius, gamma, seed=0, max_deviation=np.radians(10)):
    initial_states = torch.zeros(state_dim)
    
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    # Randomly select three angles on the circumference
    angles = torch.rand(3) * 2 * np.pi  # Three random angles between 0 and 2π

    # Calculate positions for each vehicle based on the randomly chosen angles
    position_A = (radius * torch.cos(angles[0]), radius * torch.sin(angles[0]))  # Vehicle A
    position_B = (radius * torch.cos(angles[1]), radius * torch.sin(angles[1]))  # Vehicle B
    position_C = (radius * torch.cos(angles[2]), radius * torch.sin(angles[2]))  # Vehicle C

    # Generate a random deviation within the range [-max_deviation, max_deviation]
    deviation_A = (torch.rand(1).item() - 0.5) * 2 * max_deviation
    deviation_B = (torch.rand(1).item() - 0.5) * 2 * max_deviation
    deviation_C = (torch.rand(1).item() - 0.5) * 2 * max_deviation

    # Set the ego vehicle's position and orientation with a slight deviation
    initial_states[0] = position_A[0]  # Ego x position
    initial_states[1] = position_A[1]  # Ego y position
    initial_states[6] = torch.atan2(-position_A[1], -position_A[0]) + deviation_A  # Orientation with deviation

    # Set Agent 1's position and orientation with a slight deviation
    initial_states[2] = position_B[0]  # Agent 1 x position
    initial_states[3] = position_B[1]  # Agent 1 y position
    initial_states[7] = torch.atan2(-position_B[1], -position_B[0]) + deviation_B  # Orientation with deviation

    # Set Agent 2's position and orientation with a slight deviation
    initial_states[4] = position_C[0]  # Agent 2 x position
    initial_states[5] = position_C[1]  # Agent 2 y position
    initial_states[8] = torch.atan2(-position_C[1], -position_C[0]) + deviation_C  # Orientation with deviation
    
    # Set gamma for the ego vehicle
    initial_states[9] = gamma  # Gamma value

    return initial_states

def initialize_states_random(state_dim, gamma, position_range=[-1, 1], orientation_range=[-1, 1], seed=0):
    """
    Initializes the states of three vehicles with completely random positions and orientations.

    Parameters:
    - state_dim: The dimensionality of the state vector.
    - position_range: A tuple (min_val, max_val) specifying the range for x and y positions.
    - orientation_range: A tuple (min_angle, max_angle) specifying the range for orientations (in radians).
    - gamma_range: A tuple (min_gamma, max_gamma) specifying the range for the gamma value.
    - seed: Seed for reproducibility.
    """
    torch.manual_seed(seed)
    
    initial_states = torch.zeros(state_dim)
    
    # x and y
    initial_states[0] = (torch.rand(1).item() * (position_range[1] - position_range[0]) + position_range[0])  
    initial_states[1] = (torch.rand(1).item() * (position_range[1] - position_range[0]) + position_range[0]) 
    initial_states[2] = (torch.rand(1).item() * (position_range[1] - position_range[0]) + position_range[0])  
    initial_states[3] = (torch.rand(1).item() * (position_range[1] - position_range[0]) + position_range[0]) 
    initial_states[4] = (torch.rand(1).item() * (position_range[1] - position_range[0]) + position_range[0]) 
    initial_states[5] = (torch.rand(1).item() * (position_range[1] - position_range[0]) + position_range[0])

    # theta
    initial_states[6] = (torch.rand(1).item() * (orientation_range[1] - orientation_range[0]) + orientation_range[0])  
    initial_states[7] = (torch.rand(1).item() * (orientation_range[1] - orientation_range[0]) + orientation_range[0]) 
    initial_states[8] = (torch.rand(1).item() * (orientation_range[1] - orientation_range[0]) + orientation_range[0]) 

    # gamma
    initial_states[9] = gamma 

    return initial_states

def initialize_states_circle_2(state_dim, radius, gamma, seed=0, max_deviation=np.radians(10), min_distance=0.5):
    initial_states = torch.zeros(state_dim)
    
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    valid_positions = False  # Flag to check if we have a valid position setup
    
    while not valid_positions:
        # Randomly select three angles on the circumference
        angles = torch.rand(3) * 2 * np.pi  # Three random angles between 0 and 2π

        # Calculate positions for each vehicle based on the randomly chosen angles
        position_A = (radius * torch.cos(angles[0]), radius * torch.sin(angles[0]))  # Vehicle A
        position_B = (radius * torch.cos(angles[1]), radius * torch.sin(angles[1]))  # Vehicle B
        position_C = (radius * torch.cos(angles[2]), radius * torch.sin(angles[2]))  # Vehicle C

        # Calculate pairwise distances
        distance_AB = torch.norm(torch.tensor(position_A) - torch.tensor(position_B))
        distance_AC = torch.norm(torch.tensor(position_A) - torch.tensor(position_C))
        distance_BC = torch.norm(torch.tensor(position_B) - torch.tensor(position_C))

        # Check if all distances are greater than min_distance
        if distance_AB >= min_distance and distance_AC >= min_distance and distance_BC >= min_distance:
            valid_positions = True

    # Generate a random deviation within the range [-max_deviation, max_deviation]
    deviation_A = (torch.rand(1).item() - 0.5) * 2 * max_deviation
    deviation_B = (torch.rand(1).item() - 0.5) * 2 * max_deviation
    deviation_C = (torch.rand(1).item() - 0.5) * 2 * max_deviation

    # Set the ego vehicle's position and orientation with a slight deviation
    initial_states[0] = position_A[0]  # Ego x position
    initial_states[1] = position_A[1]  # Ego y position
    initial_states[6] = torch.atan2(-position_A[1], -position_A[0]) + deviation_A  # Orientation with deviation

    # Set Agent 1's position and orientation with a slight deviation
    initial_states[2] = position_B[0]  # Agent 1 x position
    initial_states[3] = position_B[1]  # Agent 1 y position
    initial_states[7] = torch.atan2(-position_B[1], -position_B[0]) + deviation_B  # Orientation with deviation

    # Set Agent 2's position and orientation with a slight deviation
    initial_states[4] = position_C[0]  # Agent 2 x position
    initial_states[5] = position_C[1]  # Agent 2 y position
    initial_states[8] = torch.atan2(-position_C[1], -position_C[0]) + deviation_C  # Orientation with deviation
    
    # Set gamma for the ego vehicle
    initial_states[9] = gamma  # Gamma value

    return initial_states

def trajectory_rollout(policy, dynamics, tMin, tMax, dt, scenario_batch_size, initial_states):
    
    state_trajs = torch.zeros(scenario_batch_size, int((tMax - tMin) / dt) + 1, dynamics.state_dim)
    ctrl_trajs = torch.zeros(scenario_batch_size, int((tMax - tMin) / dt), dynamics.control_dim)

    state_trajs[:, 0, :] = initial_states

    for k in tqdm(range(int((tMax - tMin) / dt)), desc='Trajectory Propagation'):
        traj_time = tMax - k * dt
        traj_times = torch.full((scenario_batch_size,), traj_time)

        traj_coords = torch.cat((traj_times.unsqueeze(-1), state_trajs[:, k]), dim=-1)

        traj_policy_results = policy({'coords': dynamics.coord_to_input(traj_coords.cuda())})  # learned costate/gradient
        traj_dvs = dynamics.io_to_dv(traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach()

        # optimal control 
        ctrl_trajs[:, k] = dynamics.optimal_control(traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())

        state_trajs[:, k + 1] = dynamics.equivalent_wrapped_state(
            state_trajs[:, k].cuda() + dt * dynamics.dsdt(state_trajs[:, k].cuda(), ctrl_trajs[:, k].cuda()).cuda()
        ).cpu()

    return state_trajs, ctrl_trajs

def boundary_fn(state):
    xy1 = state[:, :, :2]
    xy2 = state[:, :, 2:4]
    xy3 = state[:, :, 4:6]

    distance_12 = torch.norm(xy1 - xy2, dim=-1)  
    distance_13 = torch.norm(xy1 - xy3, dim=-1)  
    distance_23 = torch.norm(xy2 - xy3, dim=-1) 

    min_distance = torch.min(torch.min(distance_12, distance_13), distance_23) - 0.25

    return min_distance

def neural_boundary_fn(state, tMax):
        batch_size, time_steps = state.shape[:-1]

        time_coords = torch.linspace(0, tMax, steps=time_steps).view(1, time_steps, 1)  # Shape: (1, time_steps, 1)
        time_coords = time_coords.expand(batch_size, time_steps, 1)
        coords = torch.cat((time_coords, state), dim=-1)

        model_input = dynamics.coord_to_input(coords.cuda())
        model_results = model({'coords': model_input})
        values = dynamics.io_to_value(model_results['model_in'].detach(),
                                        model_results['model_out'].squeeze(dim=-1).detach())
        return values

def visualize_value_function(model, dynamics, save_path, radius=1, x_resolution=100, y_resolution=100, z_resolution=3, time_resolution=3, seed=0, plot=False):
    ax_size = 2 * np.ceil(radius)
    plot_config = dynamics.plot_config()

    state_test_range = dynamics.state_test_range()
    x_min, x_max = state_test_range[plot_config['x_axis_idx']]
    y_min, y_max = state_test_range[plot_config['y_axis_idx']]
    z_min, z_max = state_test_range[plot_config['z_axis_idx']]
    
    times = torch.linspace(0, tMax, time_resolution)
    xs = torch.linspace(x_min, x_max, x_resolution)
    ys = torch.linspace(y_min, y_max, y_resolution)
    zs = torch.linspace(z_min, z_max, z_resolution) # theta
    gammas = torch.linspace(0, gammaMax, z_resolution)  # gamma
    xys = torch.cartesian_prod(xs, ys)
    
    
    fig_2d = plt.figure(figsize=(5 * len(times), 5 * len(zs)))
    car_image_path = 'car.png'
    car_image = plt.imread(car_image_path)

    true_bcs_min = []
    true_bcs = []

    neur_bcs_min = []
    neur_bcs = []


    for i, time in enumerate(times):
        norms = []
        for j, gamma in enumerate(gammas):
            if plot:
                ax_2d = fig_2d.add_subplot(len(times)+1, len(gammas), (j+1) + i * len(gammas))
                ax_2d.set_aspect('equal')
                ax_2d.set_xlim(-ax_size, ax_size) 
                ax_2d.set_ylim(-ax_size, ax_size)
                ax_2d.set_title(f'2D: t={time:.2f}, gamma={gamma:.2f}')

                if i == 0:  # Only add the circle for the time=0 plots
                    circle = plt.Circle((0, 0), radius, color='blue', fill=False, linestyle='--', linewidth=1.5)
                    ax_2d.add_artist(circle)


            # Initialize states      
            # initial_states = initialize_states_random(
            #     state_dim=dynamics.state_dim, 
            #     gamma=gamma,
            #     seed=seed
            # )

            initial_states = initialize_states_circle_2(
                state_dim=dynamics.state_dim,
                radius=radius,
                gamma=gamma,
                seed=seed
            )
            
            # Rollout the trajectories
            state_trajs, ctrl_trajs = trajectory_rollout(
                policy=model,
                dynamics=dynamics,
                tMin=0,
                tMax=times[i],
                dt=0.01,
                scenario_batch_size=1,
                initial_states=initial_states.unsqueeze(0) 
            )

            t1 = state_trajs[:, :, :3]
            t2 = state_trajs[:, :, 3:6]
            t3 = state_trajs[:, :, 6:9]
            norm = np.linalg.norm(t1 - t2) + np.linalg.norm(t2 - t3) + np.linalg.norm(t1 - t3)
            norms.append(norm)

            bc_true = boundary_fn(state_trajs) # TODO: rollout and plug into the actual model to get neural-approx. boundary condition
            bc_neur = neural_boundary_fn(state_trajs, times[i])
            # print(bc_true, bc_neur)
            
            if i == len(times)-1: 
                true_bcs_min.append(torch.min(bc_true).item())
                true_bcs.append(bc_true)

                neur_bcs_min.append(torch.min(bc_neur).item())
                neur_bcs.append(bc_neur)
            # Plot trajectories
            if plot: 
                for k in range(state_trajs.shape[0]):
                    for vehicle_index in range(3):
                        x_traj = state_trajs[k, :, plot_config['x_axis_idx'] + vehicle_index * 2].cpu().numpy()
                        y_traj = state_trajs[k, :, plot_config['y_axis_idx'] + vehicle_index * 2].cpu().numpy()
                        ax_2d.plot(x_traj, y_traj, lw=2, label=f'Vehicle {vehicle_index + 1} Trajectory')
                        ax_2d.scatter(x_traj[0], y_traj[0], color='green', s=50, zorder=5, label='Start' if k == 0 else "")
                        ax_2d.scatter(x_traj[-1], y_traj[-1], color='red', s=50, zorder=5, label='End' if k == 0 else "")

    # Save plots
    save_path_2d = os.path.splitext(save_path)[0] + model_name + f'/plots/trajs/traj_seed_{seed}.png' # TODO remove all of these paths and just keep one
    os.makedirs(os.path.dirname(save_path_2d), exist_ok=True)
    
    fig_2d.savefig(save_path_2d)
    plt.close(fig_2d)

    return norms, true_bcs_min, true_bcs, neur_bcs_min, neur_bcs, gammas



seeds = None
if opt.seeds == -1: seeds = [opt.seed]
else: seeds = range(opt.seeds)

norms_list = []

true_bcs_min_list = []
true_bcs_list = []

neur_bcs_min_list = []
neur_bcs_list = []

SAVE_PATH = '/home/ubuntu/deepreach_cbvf/runs/'

for seed in seeds: 
    norms, true_bcs_min, true_bcs, neur_bcs_min, neur_bcs, gammas = visualize_value_function(model, dynamics, radius=opt.radius, save_path=SAVE_PATH, seed=seed, plot=opt.plot)
    norms_list.append(norms)

    true_bcs_min_list.append(true_bcs_min)
    true_bcs_list.append(true_bcs)

    neur_bcs_min_list.append(neur_bcs_min)
    neur_bcs_list.append(neur_bcs)

output_file_path = os.path.splitext(SAVE_PATH)[0] + model_name + '/plots/metrics.txt'



# Write the norms and distances to a text file
with open(output_file_path, 'w') as f:
    for i, seed in enumerate(seeds):

        f.write(f"Seed {seed}:\n")
        f.write(f"Norms: {norms_list[i]}\n")
        f.write(f"True Min BC: {true_bcs_min_list[i]}\n")
        f.write(f"Neur Min BC: {neur_bcs_min_list[i]}\n")
        f.write("\n")
    # TODO: print how many were safe per gamma (if MIN BC >= 0, safe. otherwise, unsafe)

print(f"Results saved to {output_file_path}")


# Plot the minimum approx distances for each gamma value
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
  # Assumed gamma values for labeling, adjust as needed

if opt.plot:
    for idx, ax in enumerate(axs):
        ax.set_title(f'Gamma = {gammas[idx]}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Minimum Distance')
        
        # Plot each rollout in the corresponding gamma plot
        for seed_idx, seed_bcs in enumerate(neur_bcs_list):
            rollout_data = seed_bcs[idx].cpu().numpy().flatten()  # Get the rollout data for the current gamma
            ax.plot(rollout_data, label=f'Seed {seeds[seed_idx]}')
        
        ax.grid(True)

    # Save the plot with subplots showing minimum distances by gamma
    plot_output_path = os.path.splitext(SAVE_PATH)[0] + model_name + '/plots/min_dist_by_gamma.png'
    plt.savefig(plot_output_path, format='png', dpi=300)
    plt.close(fig)

    print(f"Min distance plot saved to {plot_output_path}")

