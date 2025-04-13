from abc import ABC, abstractmethod
from utils import diff_operators

import math
import torch

GAMMA_TEST = 0.
# during training, states will be sampled uniformly by each state dimension from the model-unit -1 to 1 range (for training stability),
# which may or may not correspond to proper test ranges
# note that coord refers to [time, *state], and input refers to whatever is fed directly to the model (often [time, *state, params])
# in the future, code will need to be fixed to correctly handle parameterized models
class Dynamics(ABC):
    def __init__(self, 
    loss_type:str, set_mode:str, 
    state_dim:int, input_dim:int, 
    control_dim:int, disturbance_dim:int, 
    state_mean:list, state_var:list, 
    value_mean:float, value_var:float, value_normto:float, 
    deepreach_model:str):
        self.loss_type = loss_type
        self.set_mode = set_mode
        self.state_dim = state_dim 
        self.input_dim = input_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.state_mean = torch.tensor(state_mean) 
        self.state_var = torch.tensor(state_var)
        self.value_mean = value_mean
        self.value_var = value_var
        self.value_normto = value_normto
        self.deepreach_model = deepreach_model
        assert self.loss_type in ['brt_hjivi', 'brat_hjivi'], f'loss type {self.loss_type} not recognized'
        if self.loss_type == 'brat_hjivi':
            assert callable(self.reach_fn) and callable(self.avoid_fn)
        assert self.set_mode in ['reach', 'avoid'], f'set mode {self.set_mode} not recognized'
        for state_descriptor in [self.state_mean, self.state_var]:
            assert len(state_descriptor) == self.state_dim, 'state descriptor dimension does not equal state dimension, ' + str(len(state_descriptor)) + ' != ' + str(self.state_dim)
    
    # ALL METHODS ARE BATCH COMPATIBLE

    # MODEL-UNIT CONVERSIONS (TODO: refactor into separate model-unit conversion class?)

    # convert model input to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepreach_model=="diff":
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepreach_model=="exact":
            return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        dodi = diff_operators.jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepreach_model=="diff":
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        elif self.deepreach_model=="exact":
            dvdt = (self.value_var / self.value_normto) * \
                (input[..., 0]*dodi[..., 0] + output)

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * input[..., 0].unsqueeze(-1)
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
        
        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    # ALL FOLLOWING METHODS USE REAL UNITS

    @abstractmethod
    def state_test_range(self):
        raise NotImplementedError

    @abstractmethod
    def equivalent_wrapped_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def dsdt(self, state, control, disturbance):
        raise NotImplementedError
    
    @abstractmethod
    def boundary_fn(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    @abstractmethod
    def cost_fn(self, state_traj):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def plot_config(self):
        raise NotImplementedError

class Dubins3D(Dynamics):
    def __init__(self, goalR:float, velocity:float, omega_max:float, angle_alpha_factor:float, set_mode:str, freeze_model: bool):
        self.goalR = goalR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.freeze_model = freeze_model
        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0, 
            state_mean=[0, 0, 0], 
            state_var=[1, 1, self.angle_alpha_factor*math.pi],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = ((wrapped_state[..., 2] + math.pi) % (2*math.pi)) - math.pi
        return wrapped_state
        
    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control): # , disturbance
        if self.freeze_model:
            raise NotImplementedError
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 2])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        return dsdt
    
    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.goalR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    # take the state (+1 from before)
    # index the position in the state where gamma is (put it last)
    def hamiltonian(self, state, dvds):
        if self.freeze_model:
            raise NotImplementedError
        if self.set_mode == 'reach':
            return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2]) 
        elif self.set_mode == 'avoid':
            return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 2])

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            return (-self.omega_max*torch.sign(dvds[..., 2]))[..., None]
        elif self.set_mode == 'avoid':
            return (self.omega_max*torch.sign(dvds[..., 2]))[..., None]

    def optimal_disturbance(self, state, dvds):
        return 0
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class Dubins3D_P(Dynamics):
    def __init__(self, goalR:float, velocity:float, omega_max:float, angle_alpha_factor:float, set_mode:str, freeze_model: bool):
        self.goalR = goalR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.freeze_model = freeze_model
        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=4, input_dim=5, control_dim=1, disturbance_dim=0, 
            state_mean=[0, 0, 0, 0], 
            state_var=[1, 1, self.angle_alpha_factor*math.pi, 1],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1, 1],
            [-1, 1],
            [-math.pi, math.pi],
            [0, 1]
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = ((wrapped_state[..., 2] + math.pi) % (2*math.pi)) - math.pi
        return wrapped_state
        
    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control): # , disturbance
        if self.freeze_model:
            raise NotImplementedError
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 2])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        return dsdt
    
    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.goalR

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):
        if self.freeze_model:
            raise NotImplementedError
        if self.set_mode == 'reach':
            return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2]) 
        elif self.set_mode == 'avoid':
            return self.velocity*(torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 2])

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            return (-self.omega_max*torch.sign(dvds[..., 2]))[..., None]
        elif self.set_mode == 'avoid':
            return (self.omega_max*torch.sign(dvds[..., 2]))[..., None]

    def optimal_disturbance(self, state, dvds):
        return 0
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class Dubins3D_P_aug(Dynamics):
    def __init__(self, goalR: float, velocity: float, omega_max: float, angle_alpha_factor: float, set_mode: str, freeze_model: bool):
        self.goalR = goalR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.freeze_model = freeze_model
        super().__init__(
            loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=5,  # x, y, sin(theta), cos(theta), gamma
            input_dim=6,  # x, y, sin(theta), cos(theta), gamma, time
            control_dim=1, disturbance_dim=0,
            state_mean=[0, 0, 0, 0, 0],
            state_var=[1, 1, 1, 1, 1],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1, 1],  # x
            [-1, 1],  # y
            [-1, 1],  # sin(theta)
            [-1, 1],  # cos(theta)
            [0, 1]    # gamma
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        theta = torch.atan2(wrapped_state[..., 2], wrapped_state[..., 3])  # sin(theta), cos(theta)
        wrapped_theta = ((theta + math.pi) % (2 * math.pi)) - math.pi
        wrapped_state[..., 2] = torch.sin(wrapped_theta)  # Update sin(theta)
        wrapped_state[..., 3] = torch.cos(wrapped_theta)  # Update cos(theta)
        return wrapped_state

    # Dubins3D dynamics with sin/cos(theta) representation
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot sin(theta) = -u * cos(theta)
    # \dot cos(theta) = u * sin(theta)
    def dsdt(self, state, control, disturbance):
        if self.freeze_model:
            raise NotImplementedError
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity * state[..., 3]  # x_dot = v * cos(theta)
        dsdt[..., 1] = self.velocity * state[..., 2]  # y_dot = v * sin(theta)
        dsdt[..., 2] = -control[..., 0] * state[..., 3]  # sin(theta)_dot
        dsdt[..., 3] = control[..., 0] * state[..., 2]   # cos(theta)_dot
        return dsdt

    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - self.goalR

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    # Compute the Hamiltonian for reach/avoid objectives
    def hamiltonian(self, state, dvds):
        if self.freeze_model:
            raise NotImplementedError
        if self.set_mode == 'reach':
            return self.velocity * (state[..., 3] * dvds[..., 0] + state[..., 2] * dvds[..., 1]) - self.omega_max * torch.abs(state[..., 3] * dvds[..., 2] - state[..., 2] * dvds[..., 3])
        elif self.set_mode == 'avoid':
            return self.velocity * (state[..., 3] * dvds[..., 0] + state[..., 2] * dvds[..., 1]) + self.omega_max * torch.abs(state[..., 3] * dvds[..., 2] - state[..., 2] * dvds[..., 3])

    # Compute the optimal control based on dvds
    def optimal_control(self, state, dvds):
        # dvds[..., 2] and dvds[..., 3] correspond to sin(theta) and cos(theta) components
        angle_term = state[..., 3] * dvds[..., 2] - state[..., 2] * dvds[..., 3]
        if self.set_mode == 'reach':
            return (-self.omega_max * torch.sign(angle_term))[..., None]
        elif self.set_mode == 'avoid':
            return (self.omega_max * torch.sign(angle_term))[..., None]

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 0, 0],
            'state_labels': ['x', 'y', r'$\sin(\theta)$', r'$\cos(\theta)$', r'$\gamma$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': (2, 3),
            'gamma_idx': 4
        }

class MultiVehicleCollision(Dynamics):
    def __init__(self):
        self.angle_alpha_factor = 1.2
        self.velocity = 0.6
        self.omega_max = 1.1
        self.collisionR = 0.25
        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=9, input_dim=10, control_dim=3, disturbance_dim=0,
            state_mean=[
                0, 0,
                0, 0, 
                0, 0,
                0, 0, 0,
            ],
            state_var=[
                1, 1,
                1, 1,
                1, 1,
                self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi,
            ],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi],           
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 6] = (wrapped_state[..., 6] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 7] = (wrapped_state[..., 7] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 8] = (wrapped_state[..., 8] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state
        
    # dynamics (per car)
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 6])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 6])
        dsdt[..., 2] = self.velocity*torch.cos(state[..., 7])
        dsdt[..., 3] = self.velocity*torch.sin(state[..., 7])
        dsdt[..., 4] = self.velocity*torch.cos(state[..., 8])
        dsdt[..., 5] = self.velocity*torch.sin(state[..., 8])
        dsdt[..., 6] = control[..., 0]
        dsdt[..., 7] = control[..., 1]
        dsdt[..., 8] = control[..., 2]
        return dsdt
    
    def boundary_fn(self, state):
        boundary_values = torch.norm(state[..., 0:2] - state[..., 2:4], dim=-1) - self.collisionR
        for i in range(1, 2):
            boundary_values_current = torch.norm(state[..., 0:2] - state[..., 2*(i+1):2*(i+1)+2], dim=-1) - self.collisionR
            boundary_values = torch.min(boundary_values, boundary_values_current)
        # Collision cost between the evaders themselves
        for i in range(2):
            for j in range(i+1, 2):
                evader1_coords_index = (i+1)*2
                evader2_coords_index = (j+1)*2
                boundary_values_current = torch.norm(state[..., evader1_coords_index:evader1_coords_index+2] - state[..., evader2_coords_index:evader2_coords_index+2], dim=-1) - self.collisionR
                boundary_values = torch.min(boundary_values, boundary_values_current)
        return boundary_values

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds): # FIXME: mkim added these args (value, maxGamma) which caused breaking changes, but unneeded?
        # Compute the hamiltonian for the ego vehicle
        ham = self.velocity*(torch.cos(state[..., 6]) * dvds[..., 0] + torch.sin(state[..., 6]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 6])
        # Hamiltonian effect due to other vehicles
        ham += self.velocity*(torch.cos(state[..., 7]) * dvds[..., 2] + torch.sin(state[..., 7]) * dvds[..., 3]) + self.omega_max * torch.abs(dvds[..., 7])
        ham += self.velocity*(torch.cos(state[..., 8]) * dvds[..., 4] + torch.sin(state[..., 8]) * dvds[..., 5]) + self.omega_max * torch.abs(dvds[..., 8])
        return ham

    def optimal_control(self, state, dvds):
        return self.omega_max*torch.sign(dvds[..., [6, 7, 8]])

    def optimal_disturbance(self, state, dvds):
        return 0
    
    def plot_config(self):
        return {
            'state_slices': [
                0, 0, 
                -0.4, 0, 
                0.4, 0,
                math.pi/2, math.pi/4, 3*math.pi/4,
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$x_2$', r'$y_2$',
                r'$x_3$', r'$y_3$',
                r'$\theta_1$', r'$\theta_2$', r'$\theta_3$',
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 6,
        }

class MultiVehicleCollision_P(Dynamics):
    def __init__(self):
        self.angle_alpha_factor = 1.2
        self.velocity = 0.6
        self.omega_max = 1.1
        self.collisionR = 0.25 # 0.25

        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=10, input_dim=11, control_dim=3, disturbance_dim=0,
            state_mean=[
                0, 0,
                0, 0, 
                0, 0,
                0, 0, 0, 
                0 # TODO: figure out what this does
            ],
            state_var=[
                1, 1,
                1, 1,
                1, 1,
                self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi,
                1 # TODO
            ],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi],  
            [0, 1] # Original
            # [0, 0] # gamma = 0 only
        ]


    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 6] = (wrapped_state[..., 6] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 7] = (wrapped_state[..., 7] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 8] = (wrapped_state[..., 8] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state
        
    # dynamics (per car)
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 6])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 6])
        dsdt[..., 2] = self.velocity*torch.cos(state[..., 7])
        dsdt[..., 3] = self.velocity*torch.sin(state[..., 7])
        dsdt[..., 4] = self.velocity*torch.cos(state[..., 8])
        dsdt[..., 5] = self.velocity*torch.sin(state[..., 8])
        dsdt[..., 6] = control[..., 0]
        dsdt[..., 7] = control[..., 1]
        dsdt[..., 8] = control[..., 2]
        return dsdt
    
    def boundary_fn(self, state):
        boundary_values = torch.norm(state[..., 0:2] - state[..., 2:4], dim=-1) - self.collisionR
        for i in range(1, 2):
            boundary_values_current = torch.norm(state[..., 0:2] - state[..., 2*(i+1):2*(i+1)+2], dim=-1) - self.collisionR
            boundary_values = torch.min(boundary_values, boundary_values_current)
        # Collision cost between the evaders themselves
        for i in range(2):
            for j in range(i+1, 2):
                evader1_coords_index = (i+1)*2
                evader2_coords_index = (j+1)*2
                boundary_values_current = torch.norm(state[..., evader1_coords_index:evader1_coords_index+2] - state[..., evader2_coords_index:evader2_coords_index+2], dim=-1) - self.collisionR
                boundary_values = torch.min(boundary_values, boundary_values_current)
        return boundary_values

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):
        # Compute the hamiltonian for the ego vehicle
        ham = self.velocity*(torch.cos(state[..., 6]) * dvds[..., 0] + torch.sin(state[..., 6]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 6])
        # Hamiltonian effect due to other vehicles
        ham += self.velocity*(torch.cos(state[..., 7]) * dvds[..., 2] + torch.sin(state[..., 7]) * dvds[..., 3]) + self.omega_max * torch.abs(dvds[..., 7])
        ham += self.velocity*(torch.cos(state[..., 8]) * dvds[..., 4] + torch.sin(state[..., 8]) * dvds[..., 5]) + self.omega_max * torch.abs(dvds[..., 8])
        return ham

    def optimal_control(self, state, dvds):
        return self.omega_max*torch.sign(dvds[..., [6, 7, 8]])

    def optimal_disturbance(self, state, dvds):
        return 0
    
    def plot_config(self):
        return {
            'state_slices': [
                0, 0, 
                -0.4, 0, 
                0.4, 0,
                math.pi/2, math.pi/4, 3*math.pi/4,
                0.0
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$x_2$', r'$y_2$',
                r'$x_3$', r'$y_3$',
                r'$\theta_1$', r'$\theta_2$', r'$\theta_3$',
                r'$\gamma$'
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 6,
        }

class MultiVehicleCollision_P_Real(Dynamics):
    def __init__(self):
        self.angle_alpha_factor = 1.2
        self.velocity = 0.50
        self.omega_max = 1.1
        self.collisionR = 0.4

        super().__init__(
            loss_type='brt_hjivi', set_mode='avoid',
            state_dim=10, input_dim=11, control_dim=3, disturbance_dim=0,
            state_mean=[
                0, 0,
                0, 0, 
                0, 0,
                0, 0, 0, 
                0
            ],
            state_var=[
                1, 1,
                1, 1,
                1, 1,
                self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi, self.angle_alpha_factor*math.pi,
                1
            ],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-1, 1], [-1, 1],
            [-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi],  
            [0, 1] # Original
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 6] = (wrapped_state[..., 6] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 7] = (wrapped_state[..., 7] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 8] = (wrapped_state[..., 8] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state
        
    # dynamics (per car)
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity*torch.cos(state[..., 6])
        dsdt[..., 1] = self.velocity*torch.sin(state[..., 6])
        dsdt[..., 2] = self.velocity*torch.cos(state[..., 7])
        dsdt[..., 3] = self.velocity*torch.sin(state[..., 7])
        dsdt[..., 4] = self.velocity*torch.cos(state[..., 8])
        dsdt[..., 5] = self.velocity*torch.sin(state[..., 8])
        dsdt[..., 6] = control[..., 0]
        dsdt[..., 7] = control[..., 1]
        dsdt[..., 8] = control[..., 2]
        return dsdt
    
    def boundary_fn(self, state):
        boundary_values = torch.norm(state[..., 0:2] - state[..., 2:4], dim=-1) - self.collisionR
        for i in range(1, 2):
            boundary_values_current = torch.norm(state[..., 0:2] - state[..., 2*(i+1):2*(i+1)+2], dim=-1) - self.collisionR
            boundary_values = torch.min(boundary_values, boundary_values_current)
        # Collision cost between the evaders themselves
        for i in range(2):
            for j in range(i+1, 2):
                evader1_coords_index = (i+1)*2
                evader2_coords_index = (j+1)*2
                boundary_values_current = torch.norm(state[..., evader1_coords_index:evader1_coords_index+2] - state[..., evader2_coords_index:evader2_coords_index+2], dim=-1) - self.collisionR
                boundary_values = torch.min(boundary_values, boundary_values_current)
        return boundary_values

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):
        # Compute the hamiltonian for the ego vehicle
        ham = -self.velocity*(torch.cos(state[..., 6]) * dvds[..., 0] + torch.sin(state[..., 6]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 6])
        # Hamiltonian effect due to other vehicles
        ham += self.velocity*(torch.cos(state[..., 7]) * dvds[..., 2] + torch.sin(state[..., 7]) * dvds[..., 3]) + self.omega_max * torch.abs(dvds[..., 7])
        ham += self.velocity*(torch.cos(state[..., 8]) * dvds[..., 4] + torch.sin(state[..., 8]) * dvds[..., 5]) + self.omega_max * torch.abs(dvds[..., 8])
        return ham

    def optimal_control(self, state, dvds):
        return self.omega_max*torch.sign(dvds[..., [6, 7, 8]])

    def optimal_disturbance(self, state, dvds):
        return 0
    
    def plot_config(self):
        return {
            'state_slices': [
                0, 0, 
                -0.4, 0, 
                0.4, 0,
                math.pi/2, math.pi/4, 3*math.pi/4,
                0.0
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$x_2$', r'$y_2$',
                r'$x_3$', r'$y_3$',
                r'$\theta_1$', r'$\theta_2$', r'$\theta_3$',
                r'$\gamma$'
            ],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 6,
        }


class MultiVehicleCollisionRelative2(Dynamics):
    def __init__(self):
        self.angle_alpha_factor = 1.2
        self.velocity = 0.55
        self.omega_max = 1.1
        self.collisionR = 0.4

        # State layout (10D):
        #   [x1, y1, rx2, ry2, rx3, ry3, theta1, rtheta2, rtheta3, gamma]
        super().__init__(
            loss_type='brt_hjivi',
            set_mode='avoid',
            state_dim=10,
            input_dim=11,   # same as your old code if you like
            control_dim=3,  # one heading control per vehicle
            disturbance_dim=0,
            state_mean=[
                0, 0,  # x1, y1
                0, 0,  # rx2, ry2
                0, 0,  # rx3, ry3
                0, 0, 0,  # theta1, rtheta2, rtheta3
                0        # gamma
            ],
            state_var=[
                1, 1,  # x1, y1
                1, 1,  # rx2, ry2
                1, 1,  # rx3, ry3
                self.angle_alpha_factor*math.pi,  # theta1
                self.angle_alpha_factor*math.pi,  # rtheta2
                self.angle_alpha_factor*math.pi,  # rtheta3
                1
            ],
            value_mean=0.25,
            value_var=0.5,
            value_normto=0.02,
            deepreach_model="exact"
        )

    def state_test_range(self):
        return [
            [-1, 1],  # x1
            [-1, 1],  # y1
            [-2, 2],  # rx2
            [-2, 2],  # ry2
            [-2, 2],  # rx3
            [-2, 2],  # ry3
            [-math.pi, math.pi],  # theta1
            [-math.pi, math.pi],  # rtheta2
            [-math.pi, math.pi],  # rtheta3
            [0, 1]  # gamma fixed
        ]

    def equivalent_wrapped_state(self, state):
        # Wrap angles to [-pi, pi]
        wrapped = torch.clone(state)
        wrapped[..., 6] = (wrapped[..., 6] + math.pi) % (2*math.pi) - math.pi  # theta1
        wrapped[..., 7] = (wrapped[..., 7] + math.pi) % (2*math.pi) - math.pi  # rtheta2
        wrapped[..., 8] = (wrapped[..., 8] + math.pi) % (2*math.pi) - math.pi  # rtheta3
        return wrapped

    def dsdt(self, state, control, disturbance):
        """
        state = [x1, y1, rx2, ry2, rx3, ry3, theta1, rtheta2, rtheta3, gamma]
        control = [u1, u2, u3]
        """
        ds = torch.zeros_like(state)

        x1       = state[..., 0]
        y1       = state[..., 1]
        rx2      = state[..., 2]
        ry2      = state[..., 3]
        rx3      = state[..., 4]
        ry3      = state[..., 5]
        theta1   = state[..., 6]
        rtheta2  = state[..., 7]
        rtheta3  = state[..., 8]
        # gamma  = state[..., 9]  # often static

        u1 = control[..., 0]
        u2 = control[..., 1]
        u3 = control[..., 2]

        v = self.velocity

        # Vehicle 1 (Ego) absolute
        ds[..., 0] = v * torch.cos(theta1)  # dot{x1}
        ds[..., 1] = v * torch.sin(theta1)  # dot{y1}
        ds[..., 6] = u1                    # dot{theta1}

        # Relative coords for vehicle 2
        #   dot{rx2} = v[cos(theta1 + rtheta2) - cos(theta1)]
        #   dot{ry2} = v[sin(theta1 + rtheta2) - sin(theta1)]
        ds[..., 2] = v*(torch.cos(theta1 + rtheta2) - torch.cos(theta1))
        ds[..., 3] = v*(torch.sin(theta1 + rtheta2) - torch.sin(theta1))
        #   dot{rtheta2} = u2 - u1
        ds[..., 7] = u2 - u1

        # Relative coords for vehicle 3
        ds[..., 4] = v*(torch.cos(theta1 + rtheta3) - torch.cos(theta1))
        ds[..., 5] = v*(torch.sin(theta1 + rtheta3) - torch.sin(theta1))
        ds[..., 8] = u3 - u1

        return ds

    def boundary_fn(self, state):
        """
        In relative coords:
          - Ego is effectively at (0,0) in the 'relative' subspace 
            (but we stored x1,y1 if we want absolute).
          - Vehicle 2 at (rx2, ry2).
          - Vehicle 3 at (rx3, ry3).
        Distances:
          (1) d(Ego, V2) = ||(rx2, ry2)||
          (2) d(Ego, V3) = ||(rx3, ry3)||
          (3) d(V2, V3)  = ||((rx3 - rx2),(ry3 - ry2))||
        """
        rx2 = state[..., 2]
        ry2 = state[..., 3]
        rx3 = state[..., 4]
        ry3 = state[..., 5]

        # Distance between Ego and Vehicle 2
        dist_12 = torch.sqrt(rx2**2 + ry2**2)
        # Distance between Ego and Vehicle 3
        dist_13 = torch.sqrt(rx3**2 + ry3**2)
        # Distance between Vehicle 2 and Vehicle 3
        dist_23 = torch.sqrt((rx3 - rx2)**2 + (ry3 - ry2)**2)

        # The boundary function is the minimal distance minus collision radius
        return torch.min(torch.min(dist_12, dist_13), dist_23) - self.collisionR

    def cost_fn(self, state_traj):
        # For time series data, boundary_fn is per state. 
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        """
        Summation of velocity dot gradient for each vehicle. 
        In the reference-based approach:
          - Vehicle 1 uses (theta1),
          - Vehicle 2 uses (theta1 + rtheta2),
          - Vehicle 3 uses (theta1 + rtheta3).
        plus the heading rates.
        """
        v = self.velocity

        # partial derivatives:
        dV_dx1  = dvds[..., 0]
        dV_dy1  = dvds[..., 1]
        dV_drx2 = dvds[..., 2]
        dV_dry2 = dvds[..., 3]
        dV_drx3 = dvds[..., 4]
        dV_dry3 = dvds[..., 5]
        dV_dth1 = dvds[..., 6]
        dV_drth2 = dvds[..., 7]
        dV_drth3 = dvds[..., 8]
        # dV_dgamma = dvds[..., 9]  # often 0 if gamma is static

        theta1  = state[..., 6]
        rtheta2 = state[..., 7]
        rtheta3 = state[..., 8]

        # Ego (Vehicle 1)
        ham = v*(torch.cos(theta1)*dV_dx1 + torch.sin(theta1)*dV_dy1) \
              + self.omega_max * torch.abs(dV_dth1)

        # Vehicle 2
        # Velocity terms appear in rx2, ry2 => direction is (theta1 + rtheta2)
        ham += v*(torch.cos(theta1 + rtheta2)*dV_drx2 + torch.sin(theta1 + rtheta2)*dV_dry2) \
               + self.omega_max * torch.abs(dV_drth2)

        # Vehicle 3
        ham += v*(torch.cos(theta1 + rtheta3)*dV_drx3 + torch.sin(theta1 + rtheta3)*dV_dry3) \
               + self.omega_max * torch.abs(dV_drth3)

        return ham

    def optimal_control(self, state, dvds):
        """
        Each vehicle's heading control is the sign that maximizes/minimizes the Hamiltonian 
        (depending on if you are computing a BRT for avoid). 
        Typically it's just sign(dV_dtheta_i).
        But in reference coords:
          - Vehicle 1 = dV_dth1
          - Vehicle 2 = dV_drth2
          - Vehicle 3 = dV_drth3
        """
        return self.omega_max * torch.sign(dvds[..., [6, 7, 8]])

    def optimal_disturbance(self, state, dvds):
        # No disturbances
        return torch.zeros_like(state[..., :0])

    def plot_config(self):
        # Example slicing: pick some "reference" slices for rx2, ry2, rx3, ry3, etc.
        return {
            'state_slices': [
                0, 0,  # x1=0, y1=0
                0, 0,  # rx2=0, ry2=0
                0, 0,  # rx3=0, ry3=0
                0, 0,  # theta1=0, rtheta2=0
                0      # rtheta3=0
            ],
            'state_labels': [
                r'$x_1$', r'$y_1$',
                r'$r x_2$', r'$r y_2$',
                r'$r x_3$', r'$r y_3$',
                r'$\theta_1$', r'$r\theta_2$', r'$r\theta_3$', 
                r'$\gamma$'
            ],
            'x_axis_idx': 2,  # e.g., show rx2 on x-axis
            'y_axis_idx': 3,  # ry2 on y-axis
            'z_axis_idx': 6   # or maybe show theta1 in 3D plots
        }

    def sample_target_state(self, num_samples):
        raise NotImplementedError
