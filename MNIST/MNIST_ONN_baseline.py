# Training from scratch (without distillation)
# Created: Aug 15, 2025
# Last Update: Aug 15, 2025
# Author: Xuening D

import sys
sys.path.append("../TA_for_ONNs")

import numpy as np
import neuroptica_new as neu

from torchvision import datasets, transforms

from neuroptica_new.losses import DistillationLoss
from neuroptica_new.utils import pbar, to_one_hot
from neuroptica_new.optimizers import InSituAdam_QAT_KD, InSituAdam_QAT
from neuroptica_new.quantization_util import get_phase_levels, set_phases_quantized, quantize_fixed_point_unsigned
from neuroptica_new.lr_scheduler import *
from neuroptica_new.preprocess import normalize_to_power
from helpers.save_result import save_dict_to_excel

from copy import deepcopy

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

from torch.utils.data import DataLoader
from itertools import product
from math import floor

def dataset_to_numpy(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset))
    images, labels = next(iter(loader))
    images = images.view(images.size(0), -1)  # Flatten to [N, 784]
    return images.numpy(), labels.numpy()

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


if __name__ == "__main__":
    transform = transforms.ToTensor()

    # remember to create the corresponding directory
    os.makedirs('./results/data', exist_ok=True)
    train_dataset = datasets.MNIST(root='./results/data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./results/data', train=False, transform=transform, download=True)

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)

    # dimensionality reduction of the dataset
    N = 10
    pca = PCA(n_components=N)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    X_train = normalize_to_power(pca.fit_transform(X_train), power_sum=N*2)
    X_test = normalize_to_power(pca.transform(X_test), power_sum=N)

    X_train = quantize_fixed_point_unsigned(X_train.copy(), bits=8, decimal=6)
    X_test = quantize_fixed_point_unsigned(X_test.copy(), bits=8, decimal=7)

    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)

    # set up the ONN
    onn_settings = {
        'N': N,
        'eo_settings': {'alpha': 0.1, 'g':0.5 * np.pi, 'phi_b': -1 * np.pi},
        'topology': 'Clements',
        'output_ports': range((N-10)//2, (N-10)//2+10)
    }

    bit_length = 8
    learning_rate = 0.0998

    scratch_train_acc, scratch_test_acc, scratch_lossy_acc = [], [], []
    best_acc, best_phases, best_profile = 0, None, None

    # define file save path
    save_path = 'MNIST/saved_models/from_scratch'
    os.makedirs(save_path, exist_ok=True)

    for seed in range(1, 20, 2):
        
        np.random.seed(seed)

        # test perfect model with perfect conditions
        # Clements model
        # model_student = neu.Sequential([
        #     neu.ClementsLayer(onn_settings['N']),
        #     neu.Activation(neu.AbsSquared(onn_settings['N'])), # photodetector measurement
        #     neu.DropMask(onn_settings['N'], keep_ports=onn_settings['output_ports'])
        # ]) 

        model_student = neu.Sequential([
            neu.ReckLayer(onn_settings['N']),
            neu.Activation(neu.AbsSquared(onn_settings['N'])), # photodetector measurement
            neu.DropMask(onn_settings['N'], keep_ports=onn_settings['output_ports'])
        ]) 

        # model_student = neu.Sequential([
            # uncomment the following for Diamond Layer
            # neu.AddMaskDiamond(onn_settings['N']),
            # neu.DiamondLayer(onn_settings['N']),
            # neu.DropMask(2*onn_settings['N'] - 2, keep_ports=range(onn_settings['N'] - 2, 2*onn_settings['N'] - 2)),

            # uncomment the following for Bokun layer
        #     neu.AddMaskDiamond(onn_settings['N']),
        #     neu.DiamondLayer(onn_settings['N']),
        #     neu.DropMask(2*onn_settings['N'] - 2, keep_ports=range(onn_settings['N']//2-1, floor(onn_settings['N']*1.5)-1)), # Middle Diamond Topology
        #     neu.Activation(neu.AbsSquared(onn_settings['N'])),
        #     neu.DropMask(onn_settings['N'], keep_ports=onn_settings['output_ports'])
        # ])
        student_phase = deepcopy(model_student.get_all_phases())

        # Model 0: with perfect conditions
        train_cfg = {
            'bit_length': bit_length,
            "learning_rate": learning_rate,
            "epochs": 80,
            "batch_size": 500,
            "alpha": 0.7,
            "temperature": 1,
            "show_progress": True,
            "cache_fields": False,
            'loss_dB': 0,  # loss per MZI, change to 0.6 for imperfect training
            'phase_uncert_phi': 0,  # phase uncertainty for phi, change to 0.1 for imperfect training
            'phase_uncert_theta': 0,  # phase uncertainty for theta, change to 0.1 for imperfect training
            # 'lr_scheduler': lambda epoch, lr: step_decay(epoch, lr, drop=0.9, epochs_drop=5)
        }

        model_student.set_all_phases_uncerts_losses(student_phase, 
                                                loss_dB=train_cfg.get('loss_dB', 0), 
                                                phase_uncert_phi=train_cfg.get('phase_uncert_phi', 0), 
                                                phase_uncert_theta=train_cfg.get('phase_uncert_theta', 0))
        
        optimizer = InSituAdam_QAT(model_student, neu.CategoricalCrossEntropy, step_size=train_cfg.get("learning_rate"), bit_length=train_cfg.get("bit_length", 8), is_diamond=False, train_cfg=train_cfg)
        losses_perfect, train_accuracy_perfect, val_accuracy_perfect, best_phases_perfect, best_trf_matrix_perfect = optimizer.fit(X_train.T, y_train.T, X_test.T, y_test.T, epochs=train_cfg.get("epochs", 50), batch_size=train_cfg.get("batch_size", 500))
        scratch_train_acc.append(np.max(train_accuracy_perfect))
        scratch_test_acc.append(np.max(val_accuracy_perfect))

        model_student.set_all_phases_uncerts_losses(best_phases_perfect, loss_dB=0.6, phase_uncert_phi=0.1, phase_uncert_theta=0.1)
        phase_levels = get_phase_levels(b = train_cfg.get("bit_length", 8))
        set_phases_quantized(model_student, phase_levels, is_diamond=False)

        phases_quantized = model_student.get_all_phases()
        model_student.set_all_phases_uncerts_losses(phases_quantized, loss_dB=0.6, phase_uncert_phi=0.1, phase_uncert_theta=0.1)   

        accs = []
        np.random.seed(21)
        for _ in range(50):
            accs.append(get_accuracy_with_quantization(model_student, X_test.T, y_test.T))
        scratch_lossy_acc.append(np.mean(accs))
        print(f"accuracy (lossy): {np.mean(accs):.2f} [%]")

        np.save(f"{save_path}/phases_Reck_baseline.npy", best_phases_perfect)

    save_dict = {
        "Training_accuracy": scratch_train_acc,
        "Test_accuracy": scratch_test_acc,
        "Final_test_accuracy": scratch_lossy_acc
    }
    save_dict_to_excel(save_dict, f"{save_path}/MNIST_baseline_reck.xlsx")