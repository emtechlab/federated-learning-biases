import pandas as pd
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math
import copy

from FL.utils.utils import weighted_average_weights, euclidean_proj_simplex

from .models import *

class GlobalBase():

   def __init__(self, device, args):

    self.args = args
    self.device = device

    if (self.args.dataset == 'cifar10'):

        self.model = ResNet18_cifar10().to(device)

   def distribute_weight(self):

    return self.model

class Fedavg_Global(GlobalBase):

    def __init__(self, device, args):

        super().__init__(device, args)

        self.args = args
        self.device = device

    def aggregate(self, local_params, round):

        print("aggregating weights with FedAvg...")

        global_weight = self.model
        local_weights = []

        for client_id ,dataclass in local_params.items():

            local_weights.append(dataclass.weight)

        w_avg = weighted_average_weights(local_weights, global_weight.state_dict())

        self.model.load_state_dict(w_avg)

def define_globalnode(device, args):

    if (args.federated_type == 'batchnorm'):

        return Fedavg_Global(device, args) 