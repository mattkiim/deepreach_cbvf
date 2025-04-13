import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import torch
from tqdm import tqdm
from utils import modules
from dynamics.dynamics import Dubins3D_P, MultiVehicleCollision_P
import configargparse

import gurobipy as gp
from gurobipy import GRB

p = configargparse.ArgumentParser()
p.add_argument('-t', '--time', type=int, default=1, required=False, help='Time.')
p.add_argument('-g', '--gamma', type=int, default=1, required=False, help='Gamma.')

opt = p.parse_args()

z_res = 3
dynamics = MultiVehicleCollision_P()

# load initial states
initial_states = torch.tensor(np.load("boundary_initials5000.npy"))[:, :-1].unsqueeze(1)
n_initials = initial_states.shape[0]

# load model
epochs = 300000
tMax = opt.time
gammaMax = 1
model_name = f"mvc_t3_g{gammaMax}"
model_path = f"../runs/{model_name}/training/checkpoints/model_epoch_{epochs}.pth"
checkpoint = torch.load(model_path)



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
model = model.cuda()
model.eval()

confusion_matrix = {
    "tp": [0]*z_res,
    "fp": [0]*z_res,
    "tn": [0]*z_res,
    "fn": [0]*z_res
}

deltas = [0., 0., 0.]
# deltas = [0.08517936766147613, 0.14283565998077394, 0.17612344861030577] # optimal


def u_nominal(x):
    """
    A simple nominal controller for three Dubins vehicles stacked in one state.
    Each vehicle has (x, y, theta) with fixed forward speed and an angular rate as control.
    
    We steer each vehicle toward the origin (0, 0).
    """
    # x shape: [batch_size, 9], grouped as:
    #  vehicle1: (x0, y0, theta0) at indices 0,1,6
    #  vehicle2: (x1, y1, theta1) at indices 2,3,7
    #  vehicle3: (x2, y2, theta2) at indices 4,5,8

    batch_size = x.shape[0]
    u = torch.zeros(batch_size, 3, device=x.device, dtype=x.dtype)
    # u will be [batch_size, 3], each column is the angular velocity for vehicle i.

    # gain for heading control
    k_p = 1.0  

    for i in range(3):
        # Indices for (x_i, y_i, theta_i)
        x_idx = 2 * i      # 0->(x0,y0), 1->(x1,y1), 2->(x2,y2)
        y_idx = 2 * i + 1
        theta_idx = 6 + i  # angles start at 6,7,8

        x_i = x[:, x_idx]       # shape [batch_size]
        y_i = x[:, y_idx]       # shape [batch_size]
        theta_i = x[:, theta_idx]

        desired_heading = torch.atan2(-y_i, -x_i)

        heading_error = desired_heading - theta_i
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))

        w_cmd = k_p * heading_error
        w_cmd = torch.clamp(w_cmd, -1.1, 1.1)

        # Assign the i-th control
        u[:, i] = w_cmd

    return u

def solve_safe_qp_gurobi(x, u_nom, grad_V, V_val, dynamics, relax_penalty=200.0, lambda_clf=1.0):
    """
    Solves: minimize ||u - u_nom||^2 + relax_penalty * r^2
    subject to: grad_V.T @ f(x, u) + lambda * V <= r
                r >= 0
                u_min <= u <= u_max
    """
    n_controls = 3  # for 3 vehicles (1 control each)

    model = gp.Model()
    model.setParam("OutputFlag", 0)  # suppress solver output

    u = model.addMVar(n_controls, lb=-1.1, ub=1.1, name="u")  # bounded angular rates
    r = model.addVar(lb=0.0, ub=GRB.INFINITY, name="r")

    # Cost: ||u - u_nom||^2 + p * r^2
    Q = np.eye(n_controls)
    obj = u @ Q @ u - 2 * u_nom.cpu().numpy() @ u + u_nom.cpu().numpy() @ u_nom.cpu().numpy() + relax_penalty * r * r
    model.setObjective(obj, GRB.MINIMIZE)

    # Dynamics f(x, u) is not affine in u, but we approximate it linearly here
    def f_func(u_val):
        u_tensor = torch.tensor(u_val, dtype=torch.float32).unsqueeze(0)
        return dynamics.dsdt(x.unsqueeze(0), u_tensor, []).squeeze(0).detach().cpu().numpy()

    # Use numerical finite difference to compute directional derivative: grad_V @ f(x, u)
    eps = 1e-4
    grad_term = 0.0
    for i in range(n_controls):
        u_eps = u_nom.clone()
        u_eps[i] += eps
        f_plus = dynamics.dsdt(x.unsqueeze(0), u_eps.unsqueeze(0), []).squeeze(0).cpu().numpy()

        u_eps = u_nom.clone()
        u_eps[i] -= eps
        f_minus = dynamics.dsdt(x.unsqueeze(0), u_eps.unsqueeze(0), []).squeeze(0).cpu().numpy()

        df_du_i = (f_plus - f_minus) / (2 * eps)
        grad_term += grad_V @ df_du_i * u[i]

    constraint = grad_V @ dynamics.dsdt(x.unsqueeze(0), u_nom.unsqueeze(0), []).squeeze(0).detach().cpu().numpy() + lambda_clf * V_val
    model.addConstr(grad_term + lambda_clf * V_val - r <= 0.0, name="clf")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        u_filtered = torch.tensor(u.X, dtype=torch.float32)
    else:
        u_filtered = u_nom  # fallback to nominal if infeasible


    return u_filtered


def trajectory_rollout(policy, dynamics, tMin, tMax, dt, scenario_batch_size, initial_states):
    # tMax = 0.01
    state_trajs = torch.zeros(scenario_batch_size, int((tMax - tMin) / dt) + 1, dynamics.state_dim)
    ctrl_trajs = torch.zeros(scenario_batch_size, int((tMax - tMin) / dt), dynamics.control_dim)

    state_trajs[:, 0, :] = initial_states.squeeze(1)

    for k in tqdm(range(int((tMax - tMin) / dt)), desc='Trajectory Propagation'):
        traj_time = tMax - k * dt
        traj_times = torch.full((scenario_batch_size,), traj_time)

        traj_coords = torch.cat((traj_times.unsqueeze(-1), state_trajs[:, k]), dim=-1)

        traj_policy_results = policy({'coords': dynamics.coord_to_input(traj_coords.cuda())})  # learned costate/gradient
        traj_dvs = dynamics.io_to_dv(traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach() # dvdt

        for i in range(scenario_batch_size):
            x = state_trajs[i, k].cuda()
            u_nom = u_nominal(x.unsqueeze(0)).squeeze(0)

            grad_V = traj_dvs[i, 1:].detach().cpu().numpy()  # skip time dim
            V_val = dynamics.io_to_value(
                traj_policy_results['model_in'][i],
                traj_policy_results['model_out'][i].squeeze()
            ).item()

            u_safe = solve_safe_qp_gurobi(x, u_nom, grad_V, V_val, dynamics, relax_penalty=50.0)
            ctrl_trajs[i, k] = u_safe.cpu()

        state_trajs[:, k+1] = dynamics.equivalent_wrapped_state(
            state_trajs[:, k].cuda() + dt * dynamics.dsdt(state_trajs[:, k].cuda(), ctrl_trajs[:, k].cuda(), []).cuda()
        ).cpu()

    return state_trajs, ctrl_trajs


def boundary_fn(state): # provided by original authors in dynamics, but more cleanly defined here
    xy1 = state[:, :, :2]
    xy2 = state[:, :, 2:4]
    xy3 = state[:, :, 4:6]

    distance_12 = torch.norm(xy1 - xy2, dim=-1)  
    distance_13 = torch.norm(xy1 - xy3, dim=-1)  
    distance_23 = torch.norm(xy2 - xy3, dim=-1) 

    min_distance = torch.min(torch.min(distance_12, distance_13), distance_23) - 0.25
    return torch.min(min_distance)


def neural_boundary_fn(policy, states, tMax):
        batch_size, time_steps = states.shape[:-1]

        time_coords = torch.linspace(0, tMax, steps=time_steps).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
        time_coords = time_coords.expand(batch_size, time_steps, 1)
        coords = torch.cat((time_coords, states), dim=-1)

        model_input = dynamics.coord_to_input(coords.cuda())
        model_results = policy({'coords': model_input})
        values = dynamics.io_to_value(model_results['model_in'].detach(),
                                        model_results['model_out'].squeeze(dim=-1).detach())
        return values


def rollouts(model, dynamics, z_resolution=z_res):
    gammas = torch.linspace(0, gammaMax, z_resolution)

    for j, gamma in enumerate(gammas):
        gamma_coords = torch.linspace(gamma, gamma, steps=n_initials).view(n_initials, 1, 1)
        gamma_coords = gamma_coords.expand(n_initials, 1, 1)
        initial_states_gamma = torch.cat((initial_states, gamma_coords), dim=-1)

        policy = model

        # rollout
        state_trajs, ctrl_trajs = trajectory_rollout(
            policy=policy,
            dynamics=dynamics,
            tMin=0,
            tMax=tMax,
            dt=0.01,
            scenario_batch_size=n_initials,
            initial_states=initial_states_gamma
        )

        verify(policy, initial_states_gamma, state_trajs, j, deltas[j])


def verify_safety(policy, states, delta):
    '''
    Helper function for verify method. 
    '''
    time_coords = torch.linspace(0, tMax, steps=states.shape[0]).view(states.shape[0], 1, 1)
    time_coords = time_coords.expand(states.shape[0], 1, 1)
    coords = torch.cat((time_coords, states), dim=-1)

    model_input = dynamics.coord_to_input(coords.cuda())
    model_results = policy({'coords': model_input})
    values = dynamics.io_to_value(model_results['model_in'].detach(),
                                    model_results['model_out'].squeeze(dim=-1).detach())

    count_in_range = ((values > 0) & (values < delta)).sum().item()
    print(f"Count of values where 0 < value < {delta}: {count_in_range}")

    return (values > delta)


def verify(policy, initial_states, trajs, j, delta):
    '''
    Computes the confusion matrix for the rollouts.
    '''
    initial_safety = verify_safety(policy, initial_states, delta)

    for i in range(initial_safety.shape[0]):
        if initial_safety[i] == 1: # initially safe
            if boundary_fn(trajs[i, 1:, :].unsqueeze(0)) > 0:
                confusion_matrix['tp'][j] += 1
            else: 
                confusion_matrix['fp'][j] += 1 
        
        elif initial_safety[i] == 0: # initially unsafe
            if boundary_fn(trajs[i, 1:, :].unsqueeze(0)) > 0:
                confusion_matrix['fn'][j] += 1
            else: 
                confusion_matrix['tn'][j] += 1


rollouts(model, dynamics)

print(confusion_matrix)
tp = [confusion_matrix['tp'][i] / initial_states.shape[0] for i in range(0, z_res)]
tn = [confusion_matrix['tn'][i] / initial_states.shape[0] for i in range(0, z_res)]
print("TP: ", tp)
print("TN: ", tn)
print("Success:", np.add(tp, tn))
