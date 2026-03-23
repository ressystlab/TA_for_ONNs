# Training ONN TAs with DNN teacher
# Created: Aug 29, 2025
# Last Update: Aug 30, 2025
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
from neuroptica_new.optimizers import InSituAdamKnowledgeDistill
from neuroptica_new.quantization_util import get_phase_levels, set_phases_quantized, quantize_fixed_point_unsigned
from neuroptica_new.lr_scheduler import *
from neuroptica_new.preprocess import normalize_to_power
from copy import deepcopy

from helpers.save_result import save_dict_to_excel
import matplotlib.pyplot as plt
from math import floor

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
    cifar_train = datasets.CIFAR10(root='./results/data', train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root='./results/data', train=False, download=True, transform=transform)

    X_train, y_train = dataset_to_numpy(cifar_train)
    X_test, y_test = dataset_to_numpy(cifar_test)

    X_train_inter = np.load("CIFAR_10/teachers/cifar_train_logits_resnet9_conv_train.npy")
    X_test_inter = np.load("CIFAR_10/teachers/cifar_train_logits_resnet9_conv_test.npy")

    processing = nn.Sequential(nn.MaxPool2d(4),
                                nn.Flatten())
    X_train_inter = processing(torch.from_numpy(X_train_inter)).detach().numpy()
    X_test_inter = processing(torch.from_numpy(X_test_inter)).detach().numpy()

    N = 16 # or 32
    pca = PCA(n_components=N)
    X_train_pca = pca.fit_transform(X_train_inter)
    X_test_pca = pca.transform(X_test_inter)

    X_train_pca = normalize_to_power(X_train_pca, power_sum=N*2)
    X_test_pca_p = normalize_to_power(X_test_pca.copy(), power_sum=N*2)

    X_train_pca = quantize_fixed_point_unsigned(X_train_pca.copy(), bits=8, decimal=6)
    X_test_pca_p = quantize_fixed_point_unsigned(X_test_pca_p.copy(), bits=8, decimal=6)

    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)

    # direct distillation from the DNN teacher
    model_output = np.load("CIFAR_10/teachers/cifar_train_logits_resnet9.npy")
    model_output = softmax(model_output)

    onn_settings = {
        'N': N,
        'eo_settings': {'alpha': 0.1, 'g':0.5 * np.pi, 'phi_b': -1 * np.pi},
        'topology': 'Clements',
        'output_ports': range((N-10)//2, (N-10)//2+10) # select the middle 10 ports
    }


    scratch_train_acc, scratch_test_acc, scratch_lossy_acc = [], [], []
    best_acc = 0

    # define file save path
    save_path = 'CIFAR_10/TA'
    os.makedirs(save_path, exist_ok=True)

    for seed in range(1, 20, 2):
        
        np.random.seed(seed)

        # test perfect model with perfect conditions
        # Clements model
        model_TA = neu.Sequential([
            neu.ClementsLayer(onn_settings['N']),
            # neu.Activation(neu.ElectroOpticActivation(onn_settings['N'], **onn_settings['eo_settings'])),
            neu.Activation(neu.cReLU(onn_settings['N'])),
            neu.ClementsLayer(onn_settings['N']),
            neu.Activation(neu.AbsSquared(onn_settings['N'])), # photodetector measurement
            neu.DropMask(onn_settings['N'], keep_ports=onn_settings['output_ports'])
        ])

        student_phase = deepcopy(model_TA.get_all_phases())

        # Model 0: with imperfect conditions
        train_cfg = {
            "learning_rate": 0.18,
            "epochs": 60,
            "data_T": model_output,
            "batch_size": 500,
            "alpha": 0.381,
            "temperature": 0.192,
            "show_progress": True,
            "cache_fields": False,
            'loss_dB': 0,  # loss per MZI, change to 0.6 for imperfect training
            'phase_uncert_phi': 0,  # phase uncertainty for phi, change to 0.1 for imperfect training
            'phase_uncert_theta': 0,  # phase uncertainty for theta, change to 0.1 for imperfect training
            # 'lr_scheduler': lambda epoch, lr: step_decay(epoch, lr, drop=0.9, epochs_drop=5)
        }

        model_TA.set_all_phases_uncerts_losses(student_phase, 
                                                loss_dB=train_cfg.get('loss_dB', 0), 
                                                phase_uncert_phi=train_cfg.get('phase_uncert_phi', 0), 
                                                phase_uncert_theta=train_cfg.get('phase_uncert_theta', 0))
        
        loss = DistillationLoss(
            alpha=train_cfg.get("alpha", 0.7), 
            temperature=train_cfg.get("temperature", 3)
        )

        optimizer = InSituAdamKnowledgeDistill(model_TA, loss, step_size=train_cfg.get("learning_rate"))

        losses_perfect, train_accuracy_perfect, val_accuracy_perfect, best_phases_perfect, best_trf_matrix_perfect = optimizer.fit(
            X_train_pca.T, y_train.T, X_test_pca_p.T, y_test.T,
            train_cfg=train_cfg
        )
        scratch_train_acc.append(np.max(train_accuracy_perfect))
        scratch_test_acc.append(np.max(val_accuracy_perfect))

        # save the best logits
        if np.max(val_accuracy_perfect) >= best_acc:
            best_acc = np.max(val_accuracy_perfect)
            teacher_logits = deepcopy(model_TA.forward_pass(X_train_pca.T).T)
            np.save(f"{save_path}/{onn_settings['topology']}_{N}_TA_phases.npy", best_phases_perfect)
            np.save(f"{save_path}/{onn_settings['topology']}_{N}_TA_logits.npy", teacher_logits)

        model_TA.set_all_phases_uncerts_losses(best_phases_perfect, loss_dB=0.6, phase_uncert_phi=0.1, phase_uncert_theta=0.1)
        phase_levels = get_phase_levels(b = train_cfg.get("bit_length", 8))
        set_phases_quantized(model_TA, phase_levels, is_diamond=False)

        phases_quantized = model_TA.get_all_phases()
        model_TA.set_all_phases_uncerts_losses(phases_quantized, loss_dB=0.6, phase_uncert_phi=0.1, phase_uncert_theta=0.1)   

        accs = []
        np.random.seed(21)
        for _ in range(50):
            accs.append(get_accuracy_with_quantization(model_TA, X_test_pca.T, y_test.T))
        scratch_lossy_acc.append(np.mean(accs))
        print(f"accuracy (lossy): {np.mean(accs):.2f} [%]")

    save_dict = {
        "Training_accuracy": scratch_train_acc,
        "Test_accuracy": scratch_test_acc,
        "Final_test_accuracy": scratch_lossy_acc
    }
    save_dict_to_excel(save_dict, f"{save_path}/KD_CIFAR_{onn_settings['topology'][0]}_{N}.xlsx")
