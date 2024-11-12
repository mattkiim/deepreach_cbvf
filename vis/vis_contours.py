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
import matplotlib.gridspec as gridspec

# Close all figures
plt.close('all')

# Initialize the dynamics object
# dynamics = Dubins3D_P(goalR=0.25, velocity=1.0, omega_max=3.14, angle_alpha_factor=1.2, set_mode='avoid', freeze_model=False)
dynamics = MultiVehicleCollision_P()


# Load the trained model
# epochs = 280000
# model_path = f"/home/ubuntu/deepreach_cbvf/runs/dubins3d_t1_g1_2/training/checkpoints/model_epoch_{epochs}.pth"
# checkpoint = torch.load(model_path)

epochs = 500000
model_name = "mvc_t2_g1"
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

# Visualization function
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
                                              model_results['model_out'].squeeze(dim=-1).detach()) # TODO: save this
                
                print(f"Value range: min = {values.min().item()}, max = {values.max().item()}")


            # Reshape values for plotting
            values_reshaped = values.cpu().numpy().reshape(x_resolution, y_resolution)
            print(values_reshaped.shape)

            ax_level_set = fig_level_sets.add_subplot(len(times), len(gammas), (j+1) + i * len(gammas))
            ax_level_set.set_aspect('equal')
            z_axis_slices = np.linspace(values.min().item(), values.max().item(), 11)  # intervals of contour

            # Plot contours for each z slice
            cs = ax_level_set.contour(xs, ys, values_reshaped.T, levels=z_axis_slices, cmap='coolwarm')
            ax_level_set.set_title(f't={time:.2f}, gamma={gamma:.2f}')
            fig_level_sets.colorbar(cs, ax=ax_level_set)


    # Save plots
    save_path_level_set = os.path.splitext(save_path)[0] + '_level_sets.png'
    
    fig_level_sets.savefig(save_path_level_set)

    plt.close(fig_level_sets)

# Call the visualization function
visualize_value_function(model, dynamics, save_path='/home/ubuntu/deepreach_cbvf/value_function')
