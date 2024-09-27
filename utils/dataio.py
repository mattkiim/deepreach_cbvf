import torch
from torch.utils.data import Dataset
import math

# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples):
        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin 
        self.tMax = tMax 
        self.counter = counter_start 
        self.counter_end = counter_end 
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples

        # TODO: below is wrong 9/27/24
        self.n_grid_pts = 300  # Example value for the number of grid points
        self.X1g = torch.linspace(-1, 1, int(self.n_grid_pts ** 0.5))  # Example grid along X1 axis
        self.X2g = torch.linspace(-1, 1, int(self.n_grid_pts ** 0.5))  # Example grid along X2 axis
        self.values_DP_grid = torch.zeros(self.n_grid_pts)  # Dummy initialization for DP grid values
        self.values_DP_linear_grid = torch.zeros(self.n_grid_pts)  # Dummy initialization for DP linear grid values


        # optimal trajectory data
        self.optimal_trajectory = None

    def __len__(self):
        return 1

    # def __getitem__(self, idx):
    #     # uniformly sample domain and include coordinates where source is non-zero 
    #     model_states = torch.zeros(self.numpoints, self.dynamics.state_dim).uniform_(-1, 1)
    #     if self.num_target_samples > 0:
    #         target_state_samples = self.dynamics.sample_target_state(self.num_target_samples)
    #         model_states[-self.num_target_samples:] = self.dynamics.coord_to_input(torch.cat((torch.zeros(self.num_target_samples, 1), target_state_samples), dim=-1))[:, 1:self.dynamics.state_dim+1]

    #     if self.pretrain:
    #         # only sample in time around the initial condition
    #         times = torch.full((self.numpoints, 1), self.tMin)
    #     else:
    #         # slowly grow time values from start time
    #         times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax-self.tMin) * (self.counter / self.counter_end))
    #         # make sure we always have training samples at the initial time
    #         times[-self.num_src_samples:, 0] = self.tMin
    #     model_coords = torch.cat((times, model_states), dim=1)        
    #     if self.dynamics.input_dim > self.dynamics.state_dim + 1: # temporary workaround for having to deal with dynamics classes for parametrized models with extra inputs
    #         model_coords = torch.cat((model_coords, torch.zeros(self.numpoints, self.dynamics.input_dim - self.dynamics.state_dim - 1)), dim=1)      

    #     boundary_values = self.dynamics.boundary_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
    #     if self.dynamics.loss_type == 'brat_hjivi':
    #         reach_values = self.dynamics.reach_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
    #         avoid_values = self.dynamics.avoid_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
        
    #     if self.pretrain:
    #         dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
    #     else:
    #         # only enforce initial conditions around self.tMin
    #         dirichlet_masks = (model_coords[:, 0] == self.tMin)

    #     if self.pretrain:
    #         self.pretrain_counter += 1
    #     elif self.counter < self.counter_end:
    #         self.counter += 1

    #     if self.pretrain and self.pretrain_counter == self.pretrain_iters:
    #         self.pretrain = False

    #     if self.dynamics.loss_type == 'brt_hjivi':
    #         return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks}
    #     elif self.dynamics.loss_type == 'brat_hjivi':
    #         return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'reach_values': reach_values, 'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks}
    #     else:
    #         raise NotImplementedError


    def __getitem__(self, idx):
        # Uniformly sample domain and include coordinates where source is non-zero
        model_states = torch.zeros(self.numpoints, self.dynamics.state_dim).uniform_(-1, 1)

        # Emphasize sampling for theta around -3.14 and 3.14
        # Assuming theta is the last dimension in model_states
        theta_samples = torch.cat([
            torch.full((self.numpoints // 4, 1), -3.14),  # 25% samples at theta = -3.14
            torch.zeros(self.numpoints // 2, 1).uniform_(-3.14, 3.14),  # 50% uniform samples
            torch.full((self.numpoints // 4, 1), 3.14)  # 25% samples at theta = 3.14
        ], dim=0)

        # Replace the theta component in model_states with theta_samples
        model_states[:, -1] = theta_samples.squeeze()

        if self.num_target_samples > 0:
            target_state_samples = self.dynamics.sample_target_state(self.num_target_samples)
            model_states[-self.num_target_samples:] = self.dynamics.coord_to_input(
                torch.cat((torch.zeros(self.num_target_samples, 1), target_state_samples), dim=-1)
            )[:, 1:self.dynamics.state_dim+1]

        if self.pretrain:
            # Only sample in time around the initial condition
            times = torch.full((self.numpoints, 1), self.tMin)
        else:
            # Slowly grow time values from start time
            times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, (self.tMax - self.tMin) * (self.counter / self.counter_end))
            # Ensure training samples at the initial time
            times[-self.num_src_samples:, 0] = self.tMin

        model_coords = torch.cat((times, model_states), dim=1)

        if self.dynamics.input_dim > self.dynamics.state_dim + 1:
            model_coords = torch.cat((model_coords, torch.zeros(self.numpoints, self.dynamics.input_dim - self.dynamics.state_dim - 1)), dim=1)

        boundary_values = self.dynamics.boundary_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])

        if self.dynamics.loss_type == 'brat_hjivi':
            reach_values = self.dynamics.reach_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])
            avoid_values = self.dynamics.avoid_fn(self.dynamics.input_to_coord(model_coords)[..., 1:])

        if self.pretrain:
            dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
        else:
            dirichlet_masks = (model_coords[:, 0] == self.tMin)

        if self.pretrain:
            self.pretrain_counter += 1
        elif self.counter < self.counter_end:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.dynamics.loss_type == 'brt_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks}
        elif self.dynamics.loss_type == 'brat_hjivi':
            return {'model_coords': model_coords}, {'boundary_values': boundary_values, 'reach_values': reach_values, 'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks}
        else:
            raise NotImplementedError
