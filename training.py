from Model import PASNet
from SubNetwork_SparseCoding import dropout_mask, s_mask
from Imbalance_CostFunc import binary_cross_entropy_for_imbalance
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import copy
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score, f1_score

from EvalFunc import auc, f1

dtype = torch.FloatTensor


# usePathwaydata, useDropout,

def trainPASNet(usePathwaydata, useDropout,train_x, train_y, eval_x, eval_y, pathway_mask, In_Nodes, Pathway_Nodes,
                Hidden_Nodes, Out_Nodes, Learning_Rate, L2_Lambda, nEpochs, Dropout_Rates, optimizer="Adam"):
    net = PASNet(usePathwaydata,In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, pathway_mask)
    # usePathwaydata,
    ###if gpu is being used
    if torch.cuda.is_available():
        net.cuda()

    ###the default optimizer is Adam
    opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay=L2_Lambda)

    for epoch in range(nEpochs):
        net.train()
        opt.zero_grad()  ###reset gradients to zeros
        pred = net(train_x)  ###Forward
        loss = F.cross_entropy(pred, train_y)
        loss.backward()  ###calculate gradients
        opt.step()  ###update weights and biases
        ###force the connections between gene layer and pathway layer
        if usePathwaydata:
            net.sc1.weight.data = net.sc1.weight.data.mul(net.pathway_mask)

        if useDropout:
            ###Randomize dropout masks
            net.do_m1 = dropout_mask(Pathway_Nodes, Dropout_Rates[0])
            net.do_m2 = dropout_mask(Hidden_Nodes, Dropout_Rates[1])
            ###obtain the small sub-network's connections
            do_m1_grad = copy.deepcopy(net.sc2.weight._grad.data)
            do_m2_grad = copy.deepcopy(net.sc3.weight._grad.data)
            do_m1_grad_mask = torch.where(do_m1_grad == 0, do_m1_grad, torch.ones_like(do_m1_grad))
            do_m2_grad_mask = torch.where(do_m2_grad == 0, do_m2_grad, torch.ones_like(do_m2_grad))
            ###copy the weights
            net_sc2_weight = copy.deepcopy(net.sc2.weight.data)
            net_sc3_weight = copy.deepcopy(net.sc3.weight.data)

            ###serializing net
            net_state_dict = net.state_dict()

            ###Sparse Coding
            ###make a copy for net, and then optimize sparsity level via copied net
            copy_net = copy.deepcopy(net)
            copy_state_dict = copy_net.state_dict()
            for name, param in copy_state_dict.items():
                ###omit the param if it is not a weight matrix
                if not "weight" in name:
                    continue
                ###omit gene layer
                if "sc1" in name:
                    continue
                ###sparse coding between the current two consecutive layers is in the trained small sub-network
                if "sc2" in name:
                    active_param = net_sc2_weight.mul(do_m1_grad_mask)
                if "sc3" in name:
                    active_param = net_sc3_weight.mul(do_m2_grad_mask)
                nonzero_param_1d = active_param[active_param != 0]
                if nonzero_param_1d.size(
                        0) == 0:  ###stop sparse coding between the current two consecutive layers if there are no valid weights
                    break
                copy_param_1d = copy.deepcopy(nonzero_param_1d)
                ###set up potential sparsity level in [0, 100)
                S_set = torch.arange(100, -1, -10)[1:]
                copy_param = copy.deepcopy(active_param)
                S_loss = []
                for S in S_set:
                    param_mask = s_mask(sparse_level=S.item(), param_matrix=copy_param, nonzero_param_1D=copy_param_1d,
                                        dtype=dtype)
                    transformed_param = copy_param.mul(param_mask)
                    copy_state_dict[name].copy_(transformed_param)
                    copy_net.train()
                    y_tmp = copy_net(train_x)
                    loss_tmp = F.cross_entropy(y_tmp, train_y)
                    S_loss.append(loss_tmp)
                ###apply cubic interpolation
                interp_S_loss = interp1d(S_set, S_loss, kind='cubic')
                interp_S_set = torch.linspace(min(S_set), max(S_set), steps=100)
                interp_loss = interp_S_loss(interp_S_set)
                optimal_S = interp_S_set[np.argmin(interp_loss)]
                optimal_param_mask = s_mask(sparse_level=optimal_S.item(), param_matrix=copy_param,
                                            nonzero_param_1D=copy_param_1d, dtype=dtype)
                if "sc2" in name:
                    final_optimal_param_mask = torch.where(do_m1_grad_mask == 0, torch.ones_like(do_m1_grad_mask),
                                                           optimal_param_mask)
                    optimal_transformed_param = net_sc2_weight.mul(final_optimal_param_mask)
                if "sc3" in name:
                    final_optimal_param_mask = torch.where(do_m2_grad_mask == 0, torch.ones_like(do_m2_grad_mask),
                                                           optimal_param_mask)
                    optimal_transformed_param = net_sc3_weight.mul(final_optimal_param_mask)
                ###update weights in copied net
                copy_state_dict[name].copy_(optimal_transformed_param)
                ###update weights in net
                net_state_dict[name].copy_(optimal_transformed_param)

        if (epoch % 1000) == 0:
            net.eval()
            eval_pred = net(eval_x)
            # eval_loss = F.cross_entropy(eval_pred, eval_y).view(1, )
            f1_eval = f1(eval_y, eval_pred)

            net.train()
            train_pred = net(train_x)
            f1_train = f1(train_y, train_pred)

            print("epoch: ", epoch)
            print("f1 train ", f1_train, " and loss train ", loss)
            print("f1 test", f1_eval)

        if epoch == nEpochs - 1:
            net.train()
            train_pred = net(train_x)
            train_loss = F.cross_entropy(train_pred, train_y).view(1, )

            net.eval()
            eval_pred = net(eval_x)
            eval_loss = F.cross_entropy(eval_pred, eval_y).view(1, )

    return (train_pred, eval_pred, train_loss, eval_loss, f1_train, f1_eval)
