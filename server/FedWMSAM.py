import scipy.stats
import torch

from client import *
from .server import Server
import numpy as np


class FedWMSAM(Server):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        super(FedWMSAM, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        print(" Var Reduction Param List  --->  {:d} * {:d}".format(
            self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))
        # Initialize communication vectors
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
            'Client_momentum': torch.zeros((init_par_list.shape[0])),
        }
        self.Client = fedwmsam
        self.local_iteration = self.args.local_epochs * (self.datasets.client_x[0].shape[0] / self.args.batchsize)
        self.value = []
        self.c_i_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        self.c_params_list = torch.zeros((init_par_list.shape[0]))
        if args.mode == "dg":
            self.local_iteration = self.args.local_epochs * (
                    len(self.datasets.client_train_set[0]) / self.args.batchsize)
        else:
            self.local_iteration = self.args.local_epochs * (self.datasets.client_x[0].shape[0] / self.args.batchsize)
        self.delta_c = torch.zeros((init_par_list.shape[0]))
        self.momentum = torch.zeros((init_par_list.shape[0]))

    def process_for_communication(self, client, Averaged_update):
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta \
                                                * (self.server_model_params_list - self.clients_params_list[client]))

        local_vr_correction = self.c_params_list - self.c_i_params_list[client]
        # Compute client personalized momentum
        self.comm_vecs['Client_momentum'].copy_(
            self.momentum + local_vr_correction * self.args.alpha / (
                        1 - self.args.alpha)
        )

    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        value = np.mean(self.value)
        self.args.alpha = 0.99 * self.args.alpha + 0.01 * max(0.1, min([value, 0.9]))
        self.value = []
        # Update global model parameters
        # SCAFFOLD (ServerOpt)
        # updated global c
        self.c_params_list += self.delta_c / self.args.total_client
        # zero delta_c for the training on the next communication round
        self.delta_c *= 0.
        self.momentum = Averaged_update / self.local_iteration / self.lr * self.args.lr_decay * -1.
        try:
            mom_norm = torch.norm(self.momentum).item()
        except Exception:
            mom_norm = float(np.linalg.norm(self.momentum))
        print("   [FedWMSAM] alpha={:.4f}  value(mean)={:.4f}  |momentum|={:.4f}  local_iter={:.2f}  lr={:.6f}".format(
            self.args.alpha, value, mom_norm, float(self.local_iteration), float(self.lr)
        ), flush=True)
        return self.server_model_params_list + self.args.global_learning_rate * Averaged_update

    def postprocess(self, client, received_vecs):
        local_value = cosine_similarity(self.clients_updated_params_list[client],
                                        self.momentum)
        self.value.append(local_value)
        updated_c_i = self.c_i_params_list[client] - self.c_params_list - \
                      self.clients_updated_params_list[client] / self.local_iteration / self.lr
        self.delta_c += updated_c_i - self.c_i_params_list[client]
        self.c_i_params_list[client] = updated_c_i


def cosine_similarity(vector_a, vector_b):
    import torch
    a = vector_a if isinstance(vector_a, torch.Tensor) else torch.as_tensor(vector_a)
    b = vector_b if isinstance(vector_b, torch.Tensor) else torch.as_tensor(vector_b)
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return 0.0
    return (torch.dot(a, b) / denom).item() + 1.0