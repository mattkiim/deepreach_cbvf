import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from utils import modules, dataio, losses
from dynamics.dynamics import Dubins3D_P
import matplotlib.gridspec as gridspec

# Close all figures
plt.close('all')

# Initialize the dynamics object
dynamics = Dubins3D_P(goalR=0.25, velocity=1.0, omega_max=3.14, angle_alpha_factor=1.2, set_mode='avoid', freeze_model=False)

# Load the trained model
epochs = 280000
model_path = f"/home/ubuntu/deepreach_cbvf/runs/dubins3d_t1_g1_2/training/checkpoints/model_epoch_{epochs}.pth"
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

# Visualization function
def visualize_value_function(model, dynamics, save_path, x_resolution=1000, y_resolution=1000, z_resolution=5, time_resolution=10):
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
    
    
    fig_3d = plt.figure(figsize=(5 * len(times), 5 * len(zs)))
    fig_2d = plt.figure(figsize=(5 * (len(times)), 5 * (len(zs)+5)))
    fig_level_sets = plt.figure(figsize=(5 * len(times), 5 * len(zs)))

    for i, time in enumerate(times):
        for j, gamma in enumerate(gammas):
            coords = torch.zeros(x_resolution * y_resolution, dynamics.state_dim + 1)
            coords[:, 0] = time
            coords[:, 1:-1] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
            coords[:, -1] = gamma  # Assign gamma as the last value

            with torch.no_grad():
                model_input = dynamics.coord_to_input(coords.cuda())
                model_results = model({'coords': model_input})
                values = dynamics.io_to_value(model_results['model_in'].detach(),
                                              model_results['model_out'].squeeze(dim=-1).detach())
                
                print(f"Value range: min = {values.min().item()}, max = {values.max().item()}")


            # Reshape values for plotting
            values_reshaped = values.cpu().numpy().reshape(x_resolution, y_resolution)

            # 3D Plot
            ax_3d = fig_3d.add_subplot(len(times), len(gammas), (j+1) + i * len(gammas), projection='3d')
            X, Y = np.meshgrid(xs, ys)
            ax_3d.plot_surface(X, Y, values_reshaped.T, cmap='viridis', edgecolor='none')
            ax_3d.set_title(f'3D: t={time:.2f}, gamma={gamma:.2f}')
            ax_3d.set_xlabel('X-axis')
            ax_3d.set_ylabel('Y-axis')
            ax_3d.set_zlabel('Values')

            # 2D Heatmap Plot
            ax_2d = fig_2d.add_subplot(len(times)+1, len(gammas), (j+1) + i * len(gammas))
            s = ax_2d.imshow(1*(values_reshaped.T <= 0), cmap='bwr', origin='lower', extent=[-1, 1, -1, 1])
            ax_2d.set_title(f'2D: t={time:.2f}, gamma={gamma:.2f}')
            fig_2d.colorbar(s, ax=ax_2d)

            ax_2d.contour(xs, ys, values_reshaped.T, levels=[0], colors='black')
            
            # New: Plot 2D level sets at various Z-slices (e.g., every 0.1)
            ax_level_set = fig_level_sets.add_subplot(len(times), len(gammas), (j+1) + i * len(gammas))
            ax_2d.set_aspect('equal')
            z_slices = torch.linspace(z_min, z_max, 10)  # Dividing into 0.1 intervals

            for z in z_slices:
                z_slice_values = np.where(values_reshaped.T == z.item(), values_reshaped.T, np.nan)
                ax_level_set.contour(xs, ys, z_slice_values, levels=[0], cmap='coolwarm')




    cmap = plt.get_cmap('cividis') 
    norm = mpl.colors.Normalize(vmin=0, vmax=len(times) - 1)

    for j, gamma in enumerate(gammas):
        ax_2d = fig_2d.add_subplot(len(times)+1, len(gammas), (j+1) + len(times) * len(gammas))
        ax_2d.set_facecolor('lightsteelblue')
        ax_2d.set_aspect('equal')
        for i, time in enumerate(times):
            coords = torch.zeros(x_resolution * y_resolution, dynamics.state_dim + 1)
            coords[:, 0] = time
            coords[:, 1:-1] = torch.tensor(plot_config['state_slices'])
            coords[:, 1 + plot_config['x_axis_idx']] = xys[:, 0]
            coords[:, 1 + plot_config['y_axis_idx']] = xys[:, 1]
            coords[:, -1] = gamma  # Assign gamma as the last value

            with torch.no_grad():
                model_input = dynamics.coord_to_input(coords.cuda())
                model_results = model({'coords': model_input})
                values = dynamics.io_to_value(model_results['model_in'].detach(),
                                              model_results['model_out'].squeeze(dim=-1).detach())
            
            values_reshaped = values.cpu().numpy().reshape(x_resolution, y_resolution)
            # ax_2d.contour(xs, ys, values_reshaped.T, levels=[0], colors=[plt.cm.viridis(i / len(times))])
            time_color = cmap(norm(i))  # Map time index to a color

            # Since the values are zero, create a contour based on time colors
            contour_plot = ax_2d.contour(xs, ys, values_reshaped.T, levels=[0], colors=[time_color])


        sm = mpl.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm = mpl.colors.Normalize(vmin=0, vmax=len(times) - 1))
        sm.set_array([])  # Required for creating a colorbar
        cbar = fig_2d.colorbar(sm, ax=ax_2d)
        # ax_2d.set_title(f'Overlay: gamma={gamma:.2f}')



    # Save plots
    save_path_3d = os.path.splitext(save_path)[0] + '_3d.png'
    save_path_2d = os.path.splitext(save_path)[0] + '_2d.png'
    save_path_level_set = os.path.splitext(save_path)[0] + '_level_sets.png'
    os.makedirs(os.path.dirname(save_path_2d), exist_ok=True)
    
    fig_3d.savefig(save_path_3d)
    fig_2d.savefig(save_path_2d)
    fig_level_sets.savefig(save_path_level_set)

    plt.close(fig_3d)
    plt.close(fig_2d)
    plt.close(fig_level_sets)

# Call the visualization function
visualize_value_function(model, dynamics, save_path='/home/ubuntu/deepreach_cbvf/value_function')
