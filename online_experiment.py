import sys
import os
import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress tracking
from utils import modules, dataio, losses
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from matplotlib.colors import ListedColormap
# Import dynamics correctly by adding its directory to sys.path
dynamics_path = "/home/ubuntu/deepreach_cbvf/dynamics"
sys.path.append(os.path.abspath(dynamics_path))
from dynamics import Dubins3D

# Close all figures
plt.close('all')
# Load the trained model
epochs = 125000
model_path = "/home/ubuntu/deepreach_cbvf/runs/dubins3d_t1_g0/training/checkpoints/model_epoch_125000.pth"
checkpoint = torch.load(model_path)

# Initialize the dynamics object
dynamics = Dubins3D(goalR=0, velocity=1.0, omega_max=3.14, angle_alpha_factor=1.2, set_mode='reach', freeze_model=False)

# Instantiate the model
model = modules.SingleBVPNet(
    in_features=dynamics.input_dim,
    out_features=1,
    type='sine',
    mode='mlp',
    final_layer_factor=1,
    hidden_features=512,
    num_hidden_layers=4
)
model.load_state_dict(checkpoint['model'])
model = model.cuda()  # Move model to GPU
model.eval()

# Define parameters
t0 = 0  # Initial time
tf = 10  # Final time
dt = 0.01  # Time step
#x0 = np.array([-1, 1, np.pi/2]) 
x0 = np.array([-1, 1, -2*np.pi/3])  # Initial state (replace with your initial state)
maxGamma = 1
omega_max = 1.0  # Define omega_max as a global variable

# Time span for the ODE solver
time_span = (t0, tf)

# Initialize lists to store trajectory and control signals
trajectory = [x0]
controls = []
deriv_traj = []
values_traj = []

def qp_controller(x, traj_dvs, v_inf, u_ref):
    """
    Function to compute the control input using a Quadratic Program (QP) controller.
    Args:
        x: Current state.
        traj_dvs: Gradient of value function from the model.
        v_inf: Value function from the model output.
        u_ref: Reference control.
    Returns:
        Optimal control input.
    """
    # Convert tensors to numpy arrays (move to CPU if necessary)
    DxV = traj_dvs.squeeze().cpu().numpy()  # Convert to CPU and numpy
    v_inf_value = v_inf.cpu().item() if isinstance(v_inf, torch.Tensor) else v_inf

    # Define the cost function for the QP
    def cost_function(u):
        return np.dot((u - u_ref).T, (u - u_ref))

    # Define the constraint for the QP
    def constraint_function(u, DxV, v_inf_value):
        g = dynamics.g(x)
        h = dynamics.h(x)
        return np.dot(DxV, g) + np.dot(DxV, h) * u + gamma * v_inf_value
    
    # Set bounds for the control input (assuming control limits are symmetric)
    bounds = Bounds([-dynamics.omega_max], [dynamics.omega_max])

    # Set up the constraint as a nonlinear constraint
    nonlinear_constraint = NonlinearConstraint(lambda u: constraint_function(u, DxV, v_inf_value), -np.inf, 0)
    
    # Initial guess for the control
    u0 = np.array([0.0])

    # Solve the QP problem
    result = minimize(cost_function, u0, method='SLSQP', bounds=bounds, constraints=[nonlinear_constraint])
    
    # Return the optimal control
    if result.success:
        return result.x
    else:
        raise ValueError("QP optimization failed")

# Add g and h methods to the Dubins3D class
def g(self, x):
    """
    Drift dynamics g(x).
    """
    v = self.velocity
    theta = x[2]
    return np.array([v * np.cos(theta), v * np.sin(theta), 0])

def h(self, x):
    """
    Control influence matrix h(x).
    """
    return np.array([0, 0, 1])

# Adding methods to the Dubins3D class dynamically
dynamics.g = g.__get__(dynamics)
dynamics.h = h.__get__(dynamics)

def feedback_control(t, x):
    """
    Function to compute the time derivative of the state, given the current state.
    The control is determined by the QP controller.
    """
    # Wrap the state to ensure the angle is within the correct range
    x = dynamics.equivalent_wrapped_state(torch.tensor(x, dtype=torch.float32)).numpy()

    # Add the current time as a feature along with the state
    traj_coords = torch.tensor([1] + list(x), dtype=torch.float32, requires_grad=True).unsqueeze(0).cuda()  # Ensure tensor is on GPU and requires grad

    # Use the policy to get controls for the current state
    #with torch.no_grad():
    traj_policy_results = model({'coords': dynamics.coord_to_input(traj_coords)})
    
    # Set the output of the model to require gradients to perform differentiation
    traj_dvs = dynamics.io_to_dv(
        traj_policy_results['model_in'],
        traj_policy_results['model_out'].squeeze(dim=-1)
    ).detach()
    #v_inf = traj_policy_results['model_out'].squeeze().item()  # Extract V_inf from model output
    v_inf = traj_policy_results['model_out'].squeeze()
    values = dynamics.io_to_value(
        traj_policy_results['model_in'].detach(), 
        traj_policy_results['model_out'].squeeze(dim=-1).detach()
    )
    # Store the value function along the trajectory
    deriv_traj.append(v_inf.item())
    values_traj.append(values.detach().cpu().numpy())

    # Reference control (could be set to zero or another reference value)
    u_ref = np.array([0.0])

    # Compute the optimal control using QP
    #ctrl=qp_controller(x, traj_dvs[..., 1:], v_inf, u_ref)
    #controls.append(ctrl)  # Record the control signal

    ctrl = dynamics.optimal_control(traj_coords[:, 1:], traj_dvs[..., 1:]).cpu().numpy().flatten()  # Move control to CPU
    controls.append(ctrl)  # Record the control signal
    return ctrl
    # Use the dynamics class to compute the control input and state derivative

def state_derivative(t, x, ctrl):
    state_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).cuda()  # Ensure tensor is on GPU
    wrapped_state_tensor = dynamics.equivalent_wrapped_state(state_tensor)
    dx_tensor = dynamics.dsdt(wrapped_state_tensor, torch.tensor(ctrl).unsqueeze(0), disturbance=None)
    dx = dx_tensor.squeeze(0).cpu().numpy()
    return dx

# Propagate the trajectory in a feedback fashion
for t in tqdm(np.arange(t0, tf, dt), desc='Trajectory Propagation'):
    # Solve the ODE from the current state to the next time step
    opt_ctrl = feedback_control(t, trajectory[-1])
    sol = solve_ivp(state_derivative, (t, t + dt), trajectory[-1], args=opt_ctrl, method='RK45', t_eval=[t + dt])
    
    # Append the new state to the trajectory
    wrapped_state = dynamics.equivalent_wrapped_state(torch.tensor(sol.y[:, -1], dtype=torch.float32)).numpy()
    trajectory.append(wrapped_state)

# Convert trajectory and controls to NumPy arrays for easier processing
trajectory = np.array(trajectory)
controls = np.array(controls)

time_points = np.arange(t0, tf, dt)

# Plot the level set at the minimum value of the function + 0.1
#plt.figure(figsize=(10, 5))
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
theta_values = torch.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x, y)
xys = torch.cartesian_prod(torch.tensor(x), torch.tensor(y))

# Prepare to compute the minimum value over theta for each (x, y) point
min_values = np.full((len(x), len(y)), np.inf)

for theta in theta_values:
    # Prepare coordinates for model evaluation
    coords = torch.zeros(xys.shape[0], 4)  # 4 dimensions: time, x, y, theta
    coords[:, 0] = 1.0  # Set time to 1.0
    coords[:, 1] = xys[:, 0]
    coords[:, 2] = xys[:, 1]
    coords[:, 3] = theta

    # Evaluate the model and update min_values
    with torch.no_grad():
        model_results = model({'coords': dynamics.coord_to_input(coords.cuda())})
        values = dynamics.io_to_value(model_results['model_in'], model_results['model_out'].squeeze(dim=-1).detach())
        values = values.cpu().numpy().reshape(len(x), len(y))
        min_values = np.minimum(min_values, values)

threshold_min = np.min(min_values)+0.1

# Plot the level set where the value is less than the threshold
# Define a custom colormap: white for background, light purple for level set
'''
cmap = ListedColormap(['white', 'plum'])
region = 1 * (min_values.T < threshold_min)
plt.imshow(region, cmap=cmap, alpha=0.5, origin='lower', extent=(-2, 2, -2, 2))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory in x-y Plane with Level Set')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show(block=False)'''

#plt.contourf(X, Y, 1 * (min_values.T < threshold_min), levels=[-0.5, 0.5], cmap='bwr', origin='lower', extent=(-2, 2, -2, 2), alpha=0.5)

# Create a large figure with four subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot the trajectories in x and y
axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], 'r', label='Trajectory (x, y)')
initial_heading = x0[2]
axs[0, 0].scatter(trajectory[0, 0], trajectory[0, 1], color='black', marker=(3, 0, np.degrees(initial_heading) - 90), s=100, label='Initial State')
axs[0, 0].scatter(trajectory[-1, 0], trajectory[-1, 1], color='black', marker='*', s=100, label='End State')
#axs[0, 0].add_patch(plt.Circle((0, 0), 0.1, color='black', alpha=0.5))
cmap = ListedColormap(['white', 'plum'])
region = 1 * (min_values.T < threshold_min)
axs[0, 0].imshow(region, cmap=cmap, alpha=0.5, origin='lower', extent=(-2, 2, -2, 2))
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('y')
axs[0, 0].set_title('Trajectory in x-y Plane with Level Set')
axs[0, 0].legend()
axs[0, 0].grid()
axs[0, 0].axis('equal')

# Plot the control signals over time
axs[0, 1].plot(time_points, controls, label='Control Signals')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Control Signal')
axs[0, 1].set_title('Control Signals Over Time')
axs[0, 1].legend()
axs[0, 1].grid()

# Plot the value function along the trajectory
axs[1, 0].plot(time_points, values_traj, label='Value Function (V_inf)')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Value Function (V_inf)')
axs[1, 0].set_title('Value Function Along the Trajectory')
axs[1, 0].legend()
axs[1, 0].grid()

# Plot the gradient along the trajectory
axs[1, 1].plot(time_points, deriv_traj, label='Gradient (DxV)')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Gradient (DxV)')
axs[1, 1].set_title('Gradient Along the Trajectory')
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.show(block=False)

'''
# Plot the initial state with a triangle pointing in the initial heading direction
initial_heading = x0[2]
plt.scatter(trajectory[0, 0], trajectory[0, 1], color='black', marker=(3, 0, np.degrees(initial_heading) - 90), s=100, label='Initial State')

plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='black', marker='*', s=100, label='End State')
plt.gca().add_patch(plt.Circle((0, 0), 0.1, color='black', alpha=0.5))  # Plot a solid disk at the origin with radius 0.1
'''
'''
# Plot the value function along the trajectory
plt.figure(figsize=(10, 5))
plt.plot(time_points, values_traj, label='Value Function (V)')
plt.xlabel('Time (s)')
plt.ylabel('Value Function (V_inf)')
plt.title('Value Function Along the Trajectory')
plt.legend()
plt.grid()
plt.show(block=False)

# Plot the gradient along the trajectory
plt.figure(figsize=(10, 5))
plt.plot(time_points, deriv_traj, label='Gradient (DxV)')
plt.xlabel('Time (s)')
plt.ylabel('Gradient (DxV)')
plt.title('Gradient Along the Trajectory')
plt.legend()
plt.grid()
plt.show(block=False)
'''
# Plot x, y, theta vs time
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot x over time
axs[0].plot(np.arange(t0, tf + dt, dt), trajectory[:, 0], label='x')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('x')
axs[0].set_title('x vs Time')
axs[0].legend()
axs[0].grid()

# Plot y over time
axs[1].plot(np.arange(t0, tf + dt, dt), trajectory[:, 1], label='y')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('y')
axs[1].set_title('y vs Time')
axs[1].legend()
axs[1].grid()

# Plot theta over time
axs[2].plot(np.arange(t0, tf + dt, dt), trajectory[:, 2], label='Theta (θ)')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Theta (θ)')
axs[2].set_title('Theta (θ) vs Time')
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show(block=True)

# Plot the control signals over time
'''plt.figure(figsize=(10, 5))
plt.plot(time_points, controls, label='Control Signals')
plt.xlabel('Time (s)')
plt.ylabel('Control Signal')
plt.title('Control Signals Over Time')
plt.legend()
plt.grid()
plt.show(block=True)'''