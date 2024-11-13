import torch

# uses real units
def init_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor, maxGamma):
    def brt_hjivi_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)
            
            diff_constraint_hom = dvdt - ham - state[..., -1] * maxGamma * value
            if minWith == 'target':
                diff_constraint_hom = torch.max(
                    diff_constraint_hom, value - boundary_value)
        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        if dynamics.deepreach_model == 'exact':
            if torch.all(dirichlet_mask):
                # pretraining
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
        
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return brt_hjivi_loss


# def init_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor, maxGamma):
#     def cbf_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output):
#         # print(dvds)
#         # exit()
#         if torch.all(dirichlet_mask):
#             cbf_loss_value = torch.zeros(1, requires_grad=True)
#         else:
#             ham = dynamics.hamiltonian(state, dvds)
#             # print(state, dvds, ham)
#             # exit()
#             cbf_condition = ham + maxGamma * value
#             cbf_loss_value = torch.min(cbf_condition, 0).values
#             # print(cbf_condition.shape, cbf_loss_value.shape)
#             # print("loss:", torch.abs(cbf_loss_value).sum())
#             # exit()

#         return {'diff_constraint_hom': torch.abs(cbf_loss_value).sum()}

#     return cbf_loss




def init_brat_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor, maxGamma):
    def brat_hjivi_loss(state, value, dvdt, dvds, boundary_value, reach_value, avoid_value, dirichlet_mask, output):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham 
            if minWith == 'target':
                diff_constraint_hom = torch.min(
                    torch.max(diff_constraint_hom, value - reach_value), value + avoid_value)

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        if dynamics.deepreach_model == 'exact':
            if torch.all(dirichlet_mask):
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
    return brat_hjivi_loss