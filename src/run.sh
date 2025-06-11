#!/bin/bash

cd $HOME/Federated-Learning-Biases/baselines/BatchNorm/src

#python main.py --federated_type batchnorm --dataset cifar10 --lr 0.1 --global_epochs 65 --local_epochs 1 --batch_size 128 --n_clients 2 --dirichlet_alpha 100.0 --partition dirichlet --sampled_clients 1.0

cd $HOME/Federated-Learning-Biases/baselines/GroupNorm/src

#python main.py --federated_type groupnorm --dataset cifar10 --lr 0.1 --global_epochs 65 --local_epochs 1 --batch_size 128 --n_clients 2 --dirichlet_alpha 100.0 --partition dirichlet --sampled_clients 1.0

cd $HOME/Federated-Learning-Biases/baselines/InstanceNorm/src

#python main.py --federated_type instancenorm --dataset cifar10 --lr 0.1 --global_epochs 65 --local_epochs 1 --batch_size 128 --n_clients 2 --dirichlet_alpha 100.0 --partition dirichlet --sampled_clients 1.0

cd $HOME/Federated-Learning-Biases/baselines/LayerNorm/src

python main.py --federated_type layernorm --dataset cifar10 --lr 0.1 --global_epochs 65 --local_epochs 1 --batch_size 128 --n_clients 2 --dirichlet_alpha 100.0 --partition dirichlet --sampled_clients 1.0