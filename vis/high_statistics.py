import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import torch
from tqdm import tqdm
from utils import modules
from dynamics.dynamics import Dubins3D_P, MultiVehicleCollision_P
import configargparse
import matplotlib.pyplot as plt


p = configargparse.ArgumentParser()
p.add_argument('-t', '--time', type=int, default=1, required=False, help='Time.')
p.add_argument('-g', '--gamma', type=int, default=1, required=False, help='Gamma.')

opt = p.parse_args()

z_res = 3
dynamics = MultiVehicleCollision_P()

# load initial states
# target_row = np.array([0.0080, -0.5999, 0.5293, -0.2825, -0.4753, -0.3662, 1.5841, 2.6514, 0.6565])
# initial_states = torch.tensor(np.load("./initial_states/initial_states_all_10000.npy"))[:, :-1].unsqueeze(1)
# initial_states = torch.tensor(np.load("./initial_states/initial_states.npy"))[:, :-1].unsqueeze(1)
# initial_states = torch.tensor(np.load("./initial_states/initial_states_all_500.npy"))[:, :-1].unsqueeze(1)
# initial_states = torch.tensor(np.load("../runs/mvc_t1_g1/plots/initial_conditions.npy"))[:, :-1].unsqueeze(1)
initial_states = torch.tensor(np.load("./initial_states/initial_states_5000.npy"))[:, :-1].unsqueeze(1)
# initial_states = torch.tensor(np.load("./initial_states/initial_conditions_2.npy"))[:, :-1].unsqueeze(1)

print(initial_states[0])
# quit()

n_initials = initial_states.shape[0]

# load model
epochs = 300000
tMax = opt.time
gammaMax = 1
model_name = f"mvc_t3_g{gammaMax}"
model_path = f"/home/ubuntu/deepreach_cbvf/runs/{model_name}/training/checkpoints/model_epoch_{epochs}.pth"
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

# deltas = [0.09363576114177703, 0.1574529004096985, 0.19414950370788575]
# deltas = [0.5177506792545319, 0.5874204361438752, 0.6656554412841797] # ** 
# deltas = [0.56786245, 0.6210246, 0.80056113] 
deltas = [0., 0., 0.]
# deltas = [0.08517936766147613, 0.14283565998077394, 0.17612344861030577] # optimal



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

        ctrl_trajs[:, k] = dynamics.optimal_control(traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda()) # optimal control


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
fp = [confusion_matrix['fp'][i] / (confusion_matrix['fp'][i] + confusion_matrix['tn'][i]) for i in range(0, z_res)]
fn = [confusion_matrix['fn'][i] / (confusion_matrix['fn'][i] + confusion_matrix['tp'][i]) for i in range(0, z_res)]
print("FP CM: ", fp)
print("FN CM: ", fn)

fp = [confusion_matrix['fp'][i] / initial_states.shape[0] for i in range(0, z_res)]
fn = [confusion_matrix['fn'][i] / initial_states.shape[0] for i in range(0, z_res)]
print("FP: ", fp)
print("FN: ", fn)

tp = [confusion_matrix['tp'][i] / initial_states.shape[0] for i in range(0, z_res)]
tn = [confusion_matrix['tn'][i] / initial_states.shape[0] for i in range(0, z_res)]
# print("TP: ", tp)
# print("TN: ", tn)
print("Success:", np.add(tp, tn))
