# knowledge distillation with DNN TA and student ONN
# Last updated: Nov 19, 2025
# Author: Xuening D
from math import floor
import sys, os
sys.path.append("../TA_for_ONNS")

import numpy as np
import neuroptica_new as neu

from neuroptica_new.losses import DistillationLoss
from neuroptica_new.utils import pbar, to_one_hot
from neuroptica_new.optimizers import InSituAdamKnowledgeDistill, InSituAdam_QAT_KD, InSituAdam_QAT
from neuroptica_new.quantization_util import get_phase_levels, set_phases_quantized, quantize_fixed_point_unsigned
from neuroptica_new.lr_scheduler import *
from neuroptica_new.preprocess import normalize_to_power
from helpers.save_result import save_dict_to_excel

from sklearn.decomposition import PCA

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from copy import deepcopy

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

def round_sig(x, sig=3):
    """Round to significant figures."""
    if x == 0:
        return 0
    from math import log10, floor
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def softmax(x):
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


if __name__ == "__main__":
    transform = transforms.ToTensor()

    # remember to create the corresponding directory
    os.makedirs('./results/data', exist_ok=True)
    train_dataset = datasets.MNIST(root='./results/data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./results/data', train=False, transform=transform, download=True)

    X_train, y_train = dataset_to_numpy(train_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)

    # dimensionality reduction of the dataset
    N = 16
    pca = PCA(n_components=N)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    X_train = normalize_to_power(pca.fit_transform(X_train), power_sum=N*2)
    X_test = normalize_to_power(pca.transform(X_test), power_sum=N)

    X_train = quantize_fixed_point_unsigned(X_train.copy(), bits=8, decimal=6)
    X_test = quantize_fixed_point_unsigned(X_test.copy(), bits=8, decimal=7)

    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)

    model_output = np.load("MNIST/TA/digital_TA_16.npy")
    model_output = softmax(model_output)

    # set up the ONN
    onn_settings = {
        'N': N,
        'eo_settings': {'alpha': 0.1, 'g':0.5 * np.pi, 'phi_b': -1 * np.pi},
        'topology': 'Clements',
        'output_ports': range((N-10)//2, (N-10)//2+10)
    }

    bit_length = 8
    learning_rate = 0.0914
    best_acc = 0

    scratch_train_acc, scratch_test_acc, scratch_lossy_acc = [], [], []

    # define file save path
    save_path = 'MNIST/saved_models'
    os.makedirs(save_path, exist_ok=True)

    for seed in range(1, 20, 2):
        
        np.random.seed(seed)

        # test perfect model with perfect conditions
        model_student = neu.Sequential([
            neu.ClementsLayer(onn_settings['N']),   # Uncomment for Clements layer
            # neu.ReckLayer(onn_settings['N']),     # Uncomment for reck layer

            # uncomment the following for Diamond Layer
            # neu.AddMaskDiamond(onn_settings['N']),
            # neu.DiamondLayer(onn_settings['N']),
            # neu.DropMask(2*onn_settings['N'] - 2, keep_ports=range(onn_settings['N'] - 2, 2*onn_settings['N'] - 2)),

            # uncomment the following for Bokun layer
            # neu.AddMaskDiamond(onn_settings['N']),
            # neu.DiamondLayer(onn_settings['N']),
            # neu.DropMask(2*onn_settings['N'] - 2, keep_ports=range(onn_settings['N']//2-1, floor(onn_settings['N']*1.5)-1)), # Middle Diamond Topology
            neu.Activation(neu.AbsSquared(onn_settings['N'])),
            neu.DropMask(onn_settings['N'], keep_ports=onn_settings['output_ports'])
        ])
        student_phase = deepcopy(model_student.get_all_phases())

        # Model 0: with perfect conditions
        train_cfg = {
            'bit_length': bit_length,
            "learning_rate": learning_rate,
            "epochs": 80,
            "data_T": model_output,
            "batch_size": 200,
            "alpha": 0.326,
            "temperature": 0.476,
            "show_progress": True,
            "cache_fields": False,
            'loss_dB': 0.6,  # loss per MZI
            'phase_uncert_phi': 0.1,  # phase uncertainty for phi
            'phase_uncert_theta': 0.1,  # phase uncertainty for theta
            # 'lr_scheduler': lambda epoch, lr: step_decay(epoch, lr, drop=0.9, epochs_drop=5)
        }

        model_student.set_all_phases_uncerts_losses(student_phase, 
                                                loss_dB=train_cfg.get('loss_dB', 0), 
                                                phase_uncert_phi=train_cfg.get('phase_uncert_phi', 0), 
                                                phase_uncert_theta=train_cfg.get('phase_uncert_theta', 0))
        
        loss = DistillationLoss(
            alpha=train_cfg.get("alpha", 0.7), 
            temperature=train_cfg.get("temperature", 3)
        )

        optimizer = InSituAdam_QAT_KD(
            model_student, 
            loss, 
            step_size=train_cfg.get("learning_rate"), 
            bit_length=train_cfg.get("bit_length", 8), 
            is_diamond=False,   # set to True for Diamond and Bokun Layers
            train_cfg=train_cfg
        )

        losses_perfect, train_accuracy_perfect, val_accuracy_perfect, best_phases_perfect, best_trf_matrix_perfect = optimizer.fit(
            X_train.T, y_train.T, X_test.T, y_test.T,
            train_cfg=train_cfg
        )
        scratch_train_acc.append(np.max(train_accuracy_perfect))
        scratch_test_acc.append(np.max(val_accuracy_perfect))

        if np.max(val_accuracy_perfect) >= best_acc:
            best_acc = np.max(val_accuracy_perfect)
            np.save(f"{save_path}/phases_Clements_KD_DNN_TA_16.npy", best_phases_perfect)

        model_student.set_all_phases_uncerts_losses(best_phases_perfect, loss_dB=0.6, phase_uncert_phi=0.1, phase_uncert_theta=0.1)
        phase_levels = get_phase_levels(b = train_cfg.get("bit_length", 8))
        set_phases_quantized(model_student, phase_levels, is_diamond=False) # set to True for Diamond and Bokun Layers

        phases_quantized = model_student.get_all_phases()
        model_student.set_all_phases_uncerts_losses(phases_quantized, loss_dB=0.6, phase_uncert_phi=0.1, phase_uncert_theta=0.1)   

        accs = []
        np.random.seed(21)
        for _ in range(50):
            accs.append(get_accuracy_with_quantization(model_student, X_test.T, y_test.T))
        scratch_lossy_acc.append(np.mean(accs))
        print(f"accuracy (lossy): {np.mean(accs):.2f} [%]")

    save_dict = {
        "Training_accuracy": scratch_train_acc,
        "Test_accuracy": scratch_test_acc,
        "Final_test_accuracy": scratch_lossy_acc
    }
    save_dict_to_excel(save_dict, f"{save_path}/MNIST_KD_C_LeNet_lossy_DNN_TA_16.xlsx")
