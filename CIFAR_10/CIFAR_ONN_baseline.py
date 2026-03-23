# Training from scratch (without distillation)
# Created: Sep 4, 2025
# Last Update: Sep 5, 2025
# Author: Xuening D

import numpy as np
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch
import sys
sys.path.append("../TA_for_ONNs")

import neuroptica_new as neu
import torch.nn as nn

from neuroptica_new.losses import DistillationLoss
from neuroptica_new.utils import pbar, to_one_hot
from neuroptica_new.optimizers import InSituAdam_QAT_KD, InSituAdam_QAT
from neuroptica_new.quantization_util import get_phase_levels, set_phases_quantized, quantize_fixed_point_unsigned
from neuroptica_new.lr_scheduler import *
from neuroptica_new.preprocess import normalize_to_power
from copy import deepcopy

from helpers.save_result import save_dict_to_excel
import matplotlib.pyplot as plt

def get_accuracy(model, test_data, test_label):
    Y_hat = model.forward_pass(test_data)
    pred = np.argmax(Y_hat, axis=0)
    gt = np.argmax(test_label, axis=0)
    return (np.mean(pred == gt) * 100)

def get_accuracy_with_quantization(model, test_data, test_label, num_bits = 8, decimal=7):
    Y_hat = model.forward_pass(test_data)
    Y_hat = quantize_fixed_point_unsigned(Y_hat.copy(), bits=num_bits, decimal=decimal)

    pred = np.argmax(Y_hat, axis=0)
    gt = np.argmax(test_label, axis=0)
    return (np.mean(pred == gt) * 100)

def dataset_to_numpy(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset))
    images, labels = next(iter(loader))
    images = images.view(images.size(0), -1)  # Flatten to [N, 3072]
    return images.numpy(), labels.numpy()

def softmax(x):
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

if __name__ == "__main__":
    # Download CIFAR-10 dataset and convert to numpy arrays
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # remember to create the corresponding directory
    os.makedirs('./results/data', exist_ok=True)
    cifar_train = datasets.CIFAR10(root='./results/data', train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root='./results/data', train=False, download=True, transform=transform)

    X_train, y_train = dataset_to_numpy(cifar_train)
    X_test, y_test = dataset_to_numpy(cifar_test)

    X_train_inter = np.load("CIFAR_10/teachers/cifar_train_logits_resnet9_conv_train.npy")
    X_test_inter = np.load("CIFAR_10/teachers/cifar_train_logits_resnet9_conv_test.npy")

    # generate the pre-processed data, using the same input data even for baseline
    processing = nn.Sequential(nn.MaxPool2d(4),
                                nn.Flatten())
    X_train_inter = processing(torch.from_numpy(X_train_inter)).detach().numpy()
    X_test_inter = processing(torch.from_numpy(X_test_inter)).detach().numpy()

    N = 10 # or 16
    pca = PCA(n_components=N)
    X_train_pca = pca.fit_transform(X_train_inter)
    X_test_pca = pca.transform(X_test_inter)

    X_train_pca = normalize_to_power(X_train_pca, power_sum=N*2)
    X_test_pca_p = normalize_to_power(X_test_pca.copy(), power_sum=N)

    X_train_pca = quantize_fixed_point_unsigned(X_train_pca.copy(), bits=8, decimal=6)
    X_test_pca_p = quantize_fixed_point_unsigned(X_test_pca_p.copy(), bits=8, decimal=7)

    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)


    onn_settings = {
        'N': N,
        'eo_settings': {'alpha': 0.1, 'g':0.5 * np.pi, 'phi_b': -1 * np.pi},
        'topology': 'clements', # or "bokun"
        'output_ports': range((N-10)//2, (N-10)//2+10)
    }

    bit_length = 8

    acc_student, acc_scratch = [], []
    acc_student_Q, acc_scratch_Q = [], []
    acc_student_lossy, acc_scratch_lossy = [], []
    best_acc = 0

    # define file save path
    save_path = 'CIFAR_10/saved_models/from_scratch'
    os.makedirs(save_path, exist_ok=True)

    for seed in range(1, 20, 2):
        
        np.random.seed(seed)

        # model for Clements layer
        model_student = neu.Sequential([
            neu.ClementsLayer(onn_settings['N']),
            neu.Activation(neu.AbsSquared(onn_settings['N'])), # photodetector measurement
            neu.DropMask(onn_settings['N'], keep_ports=onn_settings['output_ports'])
        ]) 

        # model for Bokun layer
        # model_student = neu.Sequential([
        #     neu.AddMaskDiamond(onn_settings['N']),
        #     neu.DiamondLayer(onn_settings['N']),
        #     neu.DropMask(2*onn_settings['N'] - 2, keep_ports=range(onn_settings['N']//2-1, floor(onn_settings['N']*1.5)-1)), # Middle Diamond Topology
        #     neu.Activation(neu.AbsSquared(onn_settings['N'])),
        #     neu.DropMask(onn_settings['N'], keep_ports=onn_settings['output_ports'])
        # ])

        student_phase = deepcopy(model_student.get_all_phases())

        train_cfg = {
            'loss': neu.CategoricalCrossEntropy,
            'bit_length': bit_length,
            'learning_rate': 0.15,
            'epochs': 40,
            'batch_size': 500,
            'show_progress': True,
            'cache_fields': False,
            'use_partial_vectors': False,
            'loss_dB': 0,  # loss per MZI, change to 0.6 for imperfect training
            'phase_uncert_phi': 0,  # phase uncertainty for phi, change to 0.1 for imperfect training
            'phase_uncert_theta': 0,  # phase uncertainty for theta, change to 0.1 for imperfect training
            'lr_scheduler': None#lambda epoch, lr: step_decay(epoch, lr, drop=0.9, epochs_drop=5)
        }

        model_student.set_all_phases_uncerts_losses(student_phase, 
                                                    loss_dB=train_cfg.get('loss_dB', 0), 
                                                    phase_uncert_phi=train_cfg.get('phase_uncert_phi', 0), 
                                                    phase_uncert_theta=train_cfg.get('phase_uncert_theta', 0))
        optimizer = InSituAdam_QAT(
            model_student,
            neu.CategoricalCrossEntropy,
            step_size=train_cfg["learning_rate"],
            bit_length=train_cfg["bit_length"],
            is_diamond=False,
            train_cfg=train_cfg
        )

        losses_scratch, train_accuracy_scratch, val_accuracy_scratch, \
            best_phases_perfect_scratch, best_trf_matrix_scratch = optimizer.fit(X_train_pca.T, y_train.T, 
                                                                                X_test_pca_p.T, y_test.T, 
                                                                                epochs=train_cfg.get("epochs"),
                                                                                batch_size=train_cfg.get("batch_size"),
                                                                                show_progress=train_cfg.get("show_progress"),
                                                                                cache_fields=train_cfg.get("cache_fields"),
                                                                                use_partial_vectors=train_cfg.get("use_partial_vectors"))
        
        acc_scratch.append(np.max(val_accuracy_scratch))
        
        model_student.set_all_phases_uncerts_losses(best_phases_perfect_scratch, loss_dB=0, phase_uncert_phi=0, phase_uncert_theta=0)
        phase_levels = get_phase_levels(b = train_cfg.get("bit_length", 8))
        set_phases_quantized(model_student, phase_levels, is_diamond=False)
        
        phases_quantized = deepcopy(model_student.get_all_phases())
        model_student.set_all_phases_uncerts_losses(phases_quantized, loss_dB=0, phase_uncert_phi=0, phase_uncert_theta=0)

        acc_scratch_Q.append(get_accuracy_with_quantization(model=model_student, 
                                                            test_data=X_test_pca_p.T, test_label=y_test.T, 
                                                            num_bits=8, decimal=7))
        print(f"accuracy (quantized): {acc_scratch_Q[-1]:.2f} [%]")
        
        model_student.set_all_phases_uncerts_losses(phases_quantized, loss_dB=0.6, phase_uncert_phi=0.1, phase_uncert_theta=0.1)
        accs = []
        np.random.seed(21)
        for _ in range(50):
            accs.append(get_accuracy_with_quantization(model=model_student, 
                                                            test_data=X_test_pca_p.T, test_label=y_test.T, 
                                                            num_bits=8, decimal=7))
        acc_scratch_lossy.append(np.mean(accs))
        print(f"accuracy (lossy): {np.mean(accs):.2f} [%]")
        

    print("Scratch accuracy:", acc_scratch)

    save_dict = {
        'trained_scratch_acc': acc_scratch,
        'scratch_acc_Q': acc_scratch_Q,
        'scratch_acc_lossy': acc_scratch_lossy
    }

    save_dict_to_excel(save_dict, f"{save_path}/ONN_CIFAR_{N}_{onn_settings['topology'][0].upper()}_baseline.xlsx")