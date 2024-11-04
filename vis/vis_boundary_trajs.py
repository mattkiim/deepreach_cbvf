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

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

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

def vf_eval(xys, plot_config, dynamics, model, gamma, time, x_resolution=100, y_resolution=100):

    coords = torch.zeros(x_resolution * y_resolution, dynamics.state_dim + 1)
    coords[:, 0] = time
    coords[:, 1:-1] = torch.tensor(plot_config['state_slices'])
    coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
    coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
    coords[:, 1 + plot_config['z_axis_idx']] = 0 # angle
    coords[:, -1] = gamma  # Assign gamma as the last value

    # Evaluate the value function
    with torch.no_grad():
        model_input = dynamics.coord_to_input(coords.cuda())
        model_results = model({'coords': model_input})
        values = dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1).detach())

    return values


def vf_safe(xys, vf):
    safe_indices = torch.where(vf >= 0)[0].to(xys.device) # Non-negative values indicate safe areas
    print(safe_indices)
    safe_points = xys[safe_indices]

    return safe_points


def initialize_states(xys, vf, state_dim, theta, gamma):
    zls = vf_safe(xys, vf)
    initial_states = zls[0]
    print(initial_states)

    # initial_states = torch.zeros(state_dim)
    
    # x_start = torch.rand(1).item() * 2 - 1 
    # y_start = torch.rand(1).item() * 2 - 1 
    # theta = torch.rand(1).item() * 2 * torch.pi 

    # initial_states[0] = x_start
    # initial_states[1] = y_start
    # initial_states[2] = theta

    # initial_states[3] = gamma

    return initial_states

# Function to rollout trajectories
def trajectory_rollout(policy, dynamics, tMin, tMax, dt, scenario_batch_size, initial_states):
    state_trajs = torch.zeros(scenario_batch_size, int((tMax - tMin) / dt) + 1, dynamics.state_dim)
    ctrl_trajs = torch.zeros(scenario_batch_size, int((tMax - tMin) / dt), dynamics.control_dim)

    state_trajs[:, 0, :] = initial_states

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


def visualize_value_function(model, dynamics, save_path, x_resolution=100, y_resolution=100, z_resolution=3, time_resolution=3):
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

    for j, gamma in enumerate(gammas):
        initial_state_seed = torch.rand(1).item()
        
        
        for i, time in enumerate(times):
            values = vf_eval(xys, plot_config, dynamics, model, gamma, time)

            initial_states = initialize_states(
                xys,
                values,
                state_dim=dynamics.state_dim, 
                theta=torch.tensor(0), 
                gamma=gamma
            )

            # Reshape values for plotting
            values_reshaped = values.cpu().numpy().reshape(x_resolution, y_resolution)

            # 2D Heatmap Plot
            ax_2d = fig_2d.add_subplot(len(times)+1, len(gammas), (j+1) + i * len(gammas))
            s = ax_2d.imshow(1*(values_reshaped.T <= 0), cmap='bwr', origin='lower', extent=[-1, 1, -1, 1])
            ax_2d.set_title(f'2D: t={time:.2f}, gamma={gamma:.2f}')
            fig_2d.colorbar(s, ax=ax_2d)

            ax_2d.contour(xs, ys, values_reshaped.T, levels=[0], colors='black')

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
            
            # Plot the trajectories
            for k in range(state_trajs.shape[0]):
                x_traj = state_trajs[k, :, plot_config['x_axis_idx']].cpu().numpy()
                y_traj = state_trajs[k, :, plot_config['y_axis_idx']].cpu().numpy()
                plt.plot(x_traj, y_traj, lw=2, label=f'Trajectory {k + 1}, θ={0:.2f}')
                plt.scatter(x_traj[0], y_traj[0], color='green', s=50, zorder=5, label='Start' if k == 0 else "")
                plt.scatter(x_traj[-1], y_traj[-1], color='red', s=50, zorder=5, label='End' if k == 0 else "")
                

                # start_theta = state_trajs[k, 0, 2].item()  # Assuming theta is at index 2
                # end_theta = state_trajs[k, -1, 2].item()

                # imagebox_start = OffsetImage(car_image, zoom=0.005)  # Adjust zoom for size
                # imagebox_end = OffsetImage(car_image, zoom=0.005)    # Adjust zoom for size

                # ab_start = AnnotationBbox(imagebox_start, (x_traj[0], y_traj[0]), frameon=False)
                # ab_end = AnnotationBbox(imagebox_end, (x_traj[-1], y_traj[-1]), frameon=False)

                # ab_start.set_transform(ab_start.get_transform() + transforms.Affine2D().rotate_deg(np.degrees(start_theta)))
                # ab_end.set_transform(ab_end.get_transform() + transforms.Affine2D().rotate_deg(np.degrees(end_theta)))

                # # Add the images to the plot
                # ax_2d.add_artist(ab_start)
                # ax_2d.add_artist(ab_end)


    # Save plots
    save_path_2d = os.path.splitext(save_path)[0] + model_name + '/many_trajectories.png'
    os.makedirs(os.path.dirname(save_path_2d), exist_ok=True)
    
    fig_2d.savefig(save_path_2d)

    plt.close(fig_2d)

# Call the visualization function
visualize_value_function(model, dynamics, save_path='/home/ubuntu/deepreach_cbvf/runs/')
