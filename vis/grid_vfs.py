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
def visualize_value_function(model, dynamics, save_path, x_resolution=100, y_resolution=100, z_resolution=30, time_resolution=3):
    plot_config = dynamics.plot_config()

    state_test_range = dynamics.state_test_range()
    x_min, x_max = state_test_range[plot_config['x_axis_idx']]
    y_min, y_max = state_test_range[plot_config['y_axis_idx']]
    z_min, z_max = state_test_range[plot_config['z_axis_idx']]
    
    times = torch.linspace(0, 1, time_resolution)
    xs = torch.linspace(x_min, x_max, x_resolution)
    ys = torch.linspace(y_min, y_max, y_resolution)
    zs = torch.linspace(z_min, z_max, z_resolution) # theta
    xyzs = torch.cartesian_prod(xs, ys, zs)
    print(xyzs.shape)
    
    
    fig_3d = plt.figure(figsize=(5 * len(times), 5 * len(zs)))
    fig_2d = plt.figure(figsize=(5 * (len(times)), 5 * (len(zs)+5)))
    fig_level_sets = plt.figure(figsize=(5 * len(times), 5 * len(zs)))


    gammas = [0, 0.3, 0.5]
    for gamma in gammas:
        coords = torch.zeros(x_resolution * y_resolution * z_resolution, dynamics.state_dim + 1)
        coords[:, 0] = 1
        coords[:, 1:-1] = torch.tensor(plot_config['state_slices'])
        coords[:, 1 + plot_config['x_axis_idx']] = xyzs[:, 0]
        coords[:, 1 + plot_config['y_axis_idx']] = xyzs[:, 1]
        coords[:, 1 + plot_config['z_axis_idx']] = xyzs[:, 2]
        coords[:, -1] = gamma  # Assign gamma as the last value

        with torch.no_grad():
            model_input = dynamics.coord_to_input(coords.cuda())
            model_results = model({'coords': model_input})
            values = dynamics.io_to_value(model_results['model_in'].detach(),
                                            model_results['model_out'].squeeze(dim=-1).detach())
            
                # print(f"Value range: min = {values.min().item()}, max = {values.max().item()}")


            # Reshape values for plotting
            values_reshaped = values.cpu().numpy().reshape(x_resolution, y_resolution, z_resolution)
            print(values_reshaped.shape) # should be 100 x 100 x 30

            # Save the reshaped values to a .npy file
            np.save(f"dr_gamma_{gamma}.npy", values_reshaped)