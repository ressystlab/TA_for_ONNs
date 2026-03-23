'''This module contains a collection of optimizers for training neuroptica models to fit labeled data. All optimizers
starting with "InSitu" use the on-chip interferometric gradient calculation routine described in Hughes, et al. (2018),
"Training of photonic neural networks through in situ backpropagation and gradient measurement".'''
import sys
sys.path.append("../TA_for_ONNs")

import neuroptica_new as neu
from typing import Tuple, Type
import numpy as np
from numpy import pi
from copy import deepcopy

from neuroptica_new.components import MZI, PhaseShifter
from neuroptica_new.layers import OpticalMeshNetworkLayer
from neuroptica_new.losses import Loss
from neuroptica_new.models import Sequential
from neuroptica_new.utils import pbar

from neuroptica_new.quantization_util import get_phase_levels, find_nearest, set_phases_quantized


class Optimizer:
    '''
    Base class for an optimizer
    '''

    def __init__(self, model: Sequential, loss: Type[Loss]):
        self.model = model
        self.loss = loss

    @staticmethod
    def make_batches(data: np.ndarray, labels: np.ndarray, batch_size: int,
                     shuffle=True, seed = 21):
        '''
        Prepare batches of a given size from data and labels
        :param data: features vector, shape: (n_features, n_samples)
        :param labels: labels vector, shape: (n_label_dim, n_samples)
        :param batch_size: size of the batch
        :param shuffle: if true, batches will be randomized
        :return: yields a tuple (data_batch, label_batch)
        '''
        np.random.seed(seed)
        n_features, n_samples = data.shape

        batch_indices = np.arange(0, n_samples, batch_size)

        if shuffle:
            permutation = np.random.permutation(n_samples)
            data = data[:, permutation]  # this doesn't overwrite data from outside function call
            labels = labels[:, permutation]

        for i in batch_indices:
            X = data[:, i:i + batch_size]
            Y = labels[:, i:i + batch_size]
            yield X, Y

    def fit(self, data: np.ndarray, labels: np.ndarray, epochs=None, batch_size=None):
        raise NotImplementedError("must extend Optimizer.fit() method in child classes!")

class InSituGradientDescent(Optimizer):
    '''
    On-chip training with in-situ backpropagation using adjoint field method and standard gradient descent
    '''

    def __init__(self, model: Sequential, loss: Type[Loss], learning_rate=0.01):
        super().__init__(model, loss)
        self.learning_rate = learning_rate

    def fit(self, data: np.ndarray, labels: np.ndarray, epochs=1000, batch_size=32, show_progress=True):
        '''
        Fit the model to the labeled data
        :param data: features vector, shape: (n_features, n_samples)
        :param labels: labels vector, shape: (n_label_dim, n_samples)
        :param epochs:
        :param learning_rate:
        :param batch_size:
        :param show_progress:
        :return:
        '''

        losses = []
        accuracy = []

        n_features, n_samples = data.shape

        iterator = range(epochs)
        if show_progress: iterator = pbar(iterator)

        for epoch in iterator:

            total_epoch_loss = 0.0

            for X, Y in self.make_batches(data, labels, batch_size):

                # Propagate the data forward
                Y_hat = self.model.forward_pass(X)
                d_loss = self.loss.dL(Y_hat, Y)
                total_epoch_loss += np.sum(self.loss.L(Y_hat, Y))

                # Compute the backpropagated signals for the model
                gradients = self.model.backward_pass(d_loss)
                delta_prev = d_loss  # backprop signal to send in the final layer

                # Compute the foward and adjoint fields at each phase shifter in all tunable layers
                for layer in reversed(self.model.layers):
                    if isinstance(layer, OpticalMeshNetworkLayer):
                        layer.mesh.adjoint_optimize(layer.input_prev, delta_prev,
                                                    lambda dx: -1 * self.learning_rate * dx)

                    # Set the backprop signal for the subsequent (spatially previous) layer
                    delta_prev = gradients[layer.__name__]

            total_epoch_loss /= n_samples
            losses.append(total_epoch_loss)
            # Append accuracy per epoch
            Y_hat = self.model.forward_pass(data)
            pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
            gt = np.array([np.argmax(tru) for tru in labels.T])
            accuracy.append(np.sum(pred == gt)/data.shape[1]*100)

            if show_progress:
                iterator.set_description("ℒ = {:.2f}".format(total_epoch_loss), refresh=False)

        return losses, accuracy

class InSituAdam(Optimizer):
    '''
    On-chip training with in-situ backpropagation using adjoint field method and adam optimizer
    '''
    def __init__(self, model: Sequential, loss: Type[Loss], step_size=0.01,
                 beta1=0.9, beta2=0.99, epsilon=1e-8, pKeep=0.8):
        super().__init__(model, loss)
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.pKeep = pKeep

        self.t = 0
        self.m = {}
        self.v = {}
        self.g = {}
        for layer in model.layers:
            if isinstance(layer, OpticalMeshNetworkLayer):
                for component in layer.mesh.all_tunable_components():
                    self.m[component] = np.zeros(component.dof)
                    self.v[component] = np.zeros(component.dof)
                    self.g[component] = np.zeros(component.dof)
        print('\n')

    def fit(self, data: np.ndarray, labels: np.ndarray, val_data: np.ndarray,
            val_labels: np.ndarray, epochs=1000, batch_size=32,
            show_progress=True, cache_fields=False, use_partial_vectors=False):
        '''
        Fit the model to the labeled data
        :param data: features vector, shape: (n_features, n_samples)
        :param labels: labels vector, shape: (n_label_dim, n_samples)
        :param epochs:
        :param batch_size:
        :param show_progress:
        :param cache_fields: if set to True, will cache fields at the phase shifters on the forward and backward pass
        :param use_partial_vectors: if set to True, the MZI partial matrices will be stored as Nx2 vectors
        :return: losses, accuracy
        '''
        losses = []
        trn_accuracy = [0]
        val_accuracy = [0]

        best_phases = self.model.get_all_phases()
        best_trf_matrix = self.model.get_transformation_matrix()
        n_features, n_samples = data.shape
        iterator = range(epochs)
        if show_progress: iterator = pbar(iterator)

        for epoch in iterator:

            total_epoch_loss = 0.0
            batch = 0
            for X, Y in self.make_batches(data, labels, batch_size):
                batch += 1
                self.t += 1
                # Propagate the data forward
                Y_hat = self.model.forward_pass(X, cache_fields=cache_fields, use_partial_vectors=use_partial_vectors)
                d_loss = self.loss.dL(Y_hat, Y)
                total_epoch_loss += np.sum(self.loss.L(Y_hat, Y))

                # Compute the backpropagated signals for the model
                deltas = self.model.backward_pass(d_loss, cache_fields=cache_fields,
                                                  use_partial_vectors=use_partial_vectors)
                delta_prev = d_loss  # backprop signal to send in the final layer

                # Compute the foward and adjoint fields at each phase shifter in all tunable layers
                for layer in reversed(self.model.layers):
                    if isinstance(layer, OpticalMeshNetworkLayer):
                        gradients = layer.mesh.compute_gradients(layer.input_prev, delta_prev,
                                                                 cache_fields=cache_fields,
                                                                 use_partial_vectors=use_partial_vectors)
                        for cmpt in gradients:
                            self.g[cmpt] = np.mean(gradients[cmpt], axis=-1)
                            self.m[cmpt] = self.beta1 * self.m[cmpt] + (1 - self.beta1) * self.g[cmpt]
                            self.v[cmpt] = self.beta2 * self.v[cmpt] + (1 - self.beta2) * self.g[cmpt] ** 2
                            mhat = self.m[cmpt] / (1 - self.beta1 ** self.t)
                            vhat = self.v[cmpt] / (1 - self.beta2 ** self.t)

                            grad = -1 * self.step_size * mhat / (np.sqrt(vhat) + self.epsilon)

                            # Adjust settings by gradient amount
                            if isinstance(cmpt, PhaseShifter):
                                cmpt.phi += grad[0]
                                if cmpt.phi < 0:
                                    cmpt.phi += 2*pi
                                if cmpt.phi > 2*pi:
                                    cmpt.phi -= 2*pi

                            elif isinstance(cmpt, MZI):
                                dtheta, dphi = grad

                                dtheta, dphi = grad
                                if cmpt.phi + dphi < 0:
                                    cmpt.phi += 2*pi
                                if cmpt.phi + dphi > 2*pi:
                                    cmpt.phi -= 2*pi
                                if cmpt.theta + dtheta < 0:
                                    cmpt.theta += 2*pi
                                if cmpt.theta + dtheta > 2*pi:
                                    cmpt.theta -= 2*pi

                                cmpt.phi += dphi
                                cmpt.theta += dtheta

                    # Set the backprop signal for the subsequent (spatially previous) layer
                    delta_prev = deltas[layer.__name__]

            # Append loss per epoch
            total_epoch_loss /= n_samples
            losses.append(total_epoch_loss)

            # Append training accuracy per epoch
            Y_hat = self.model.forward_pass(data)
            pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
            gt = np.array([np.argmax(tru) for tru in labels.T])
            trn_accuracy.append(np.sum(pred == gt)/data.shape[1]*100)

            # Append validation accuracy per epoch
            Y_hat = self.model.forward_pass(val_data)
            pred = np.array([np.argmax(yhat) for yhat in Y_hat.T])
            gt = np.array([np.argmax(tru) for tru in val_labels.T])
            val_accuracy.append(np.sum(pred == gt)/val_data.shape[1]*100)
            # print(val_accuracy[-1])
            
            if val_accuracy[-1] > max(val_accuracy[:-1]):
                best_phases = self.model.get_all_phases()
                best_trf_matrix = self.model.get_transformation_matrix()

            if show_progress:
                iterator.set_description("ℒ = {:.2f}".format(total_epoch_loss), refresh=True)
        
        print(f'Max Validation Accuracy: {max(val_accuracy):.2f}%')
        trn_accuracy = trn_accuracy[1:]
        val_accuracy = val_accuracy[1:]
        return losses, trn_accuracy, val_accuracy, best_phases, best_trf_matrix

class Dropout():
    ''' Implements dropout for ONN '''
    def __init__(self, p: float = 0.8, inplace: bool = False, evaluation: bool = False):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p: float = p
        self.inplace: bool = inplace
        self.evaluation: bool = evaluation

    def evaluation_mode(self):
            self.evaluation = True

    def forward(self, weights):
        if not self.evaluation:
            binomial = np.random.binomial(1, self.p)
            return weights * binomial.sample(weights.size())
        return weights * self.p

# Knowledge Distillation
class InSituAdamKnowledgeDistill(Optimizer):
    '''
    On-chip training with in-situ backpropagation using adjoint field method and adam optimizer
    '''
    def __init__(self, model: Sequential, loss: Type[Loss], step_size=0.01,
                 beta1=0.9, beta2=0.99, epsilon=1e-8, pKeep=0.8):
        super().__init__(model, loss)

        self.step_size = step_size
        self.initial_step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.pKeep = pKeep

        self.t = 0
        self.m = {}
        self.v = {}
        self.g = {}
        for layer in model.layers:
            if isinstance(layer, OpticalMeshNetworkLayer):
                for component in layer.mesh.all_tunable_components():
                    self.m[component] = np.zeros(component.dof)
                    self.v[component] = np.zeros(component.dof)
                    self.g[component] = np.zeros(component.dof)
        print('\n')

    def update_lib(self, model: Sequential):
        self.m = {}
        self.v = {}
        self.g = {}
        self.model = model
        for layer in self.model.layers:
            if isinstance(layer, OpticalMeshNetworkLayer):
                for component in layer.mesh.all_tunable_components():
                    self.m[component] = np.zeros(component.dof)
                    self.v[component] = np.zeros(component.dof)
                    self.g[component] = np.zeros(component.dof)

    def make_batches_all(self, data: np.ndarray, labels: np.ndarray, teacher_data: np.ndarray = None, batch_size: int = 32, shuffle=True):
        '''
        Prepare batches of a given size from data and labels
        :param data: features vector, shape: (n_features, n_samples)
        :param labels: labels vector, shape: (n_label_dim, n_samples)
        :param batch_size: size of the batch
        :param shuffle: if true, batches will be randomized
        :return: yields a tuple (data_batch, label_batch)
        '''

        n_features, n_samples = data.shape
        np.random.seed(21)
        batch_indices = np.arange(0, n_samples, batch_size)

        if shuffle:
            permutation = np.random.permutation(n_samples)
            data = data[:, permutation]  # this doesn't overwrite data from outside function call
            labels = labels[:, permutation]

            if teacher_data is not None:
                teacher_data = teacher_data[: , permutation]

        for i in batch_indices:
            X = data[:, i:i + batch_size]
            Y = labels[:, i:i + batch_size]
            teacher_sample = teacher_data[:, i:i + batch_size]
            yield X, Y, teacher_sample

    def fit(self, data: np.ndarray, labels: np.ndarray, val_data: np.ndarray, val_labels: np.ndarray, train_cfg: dict):
        '''
        Fit the model using Adam with support for knowledge distillation via teacher model or precomputed soft labels.

        :param data: features vector, shape: (n_features, n_samples)
        :param labels: labels vector, shape: (n_label_dim, n_samples)
        :param val_data: validation data
        :param val_labels: labels for validation
        :param train_cfg: training configurations defined in a dictionary for readibility
        :return: losses, accuracy
        '''

        # Hyperparameters from train_cfg
        epochs = train_cfg.get("epochs", 1000)
        batch_size = train_cfg.get("batch_size", 32)
        show_progress = train_cfg.get("show_progress", True)
        cache_fields = train_cfg.get("cache_fields", False)
        use_partial_vectors = train_cfg.get("use_partial_vectors", False)
        lr_scheduler = train_cfg.get('lr_scheduler', None)

        teacher_model = train_cfg.get("teacher_model", None)
        data_T = train_cfg.get("data_T", None)

        losses = []
        trn_accuracy = [0]
        val_accuracy = [0]

        best_phases = self.model.get_all_phases()
        best_trf_matrix = self.model.get_transformation_matrix()

        if data_T is not None:
            teacher_sample = data_T.T
        else:
            teacher_sample = None
        
        n_features, n_samples = data.shape
        iterator = range(epochs)
        if show_progress: iterator = pbar(iterator)

        for epoch in iterator:
            if lr_scheduler:
                self.step_size = lr_scheduler(epoch, self.initial_step_size)

            total_epoch_loss = 0.0
            batch = 0

            for X, Y, teacher_preds in self.make_batches_all(data, labels, teacher_sample, batch_size):
                batch += 1
                self.t += 1

                # Forward pass - student
                Y_hat = self.model.forward_pass(X, cache_fields=cache_fields, use_partial_vectors=use_partial_vectors)

                # Get teacher soft labels
                if teacher_model is not None:
                    # different forward propagation for ONNs and DNNs
                    if isinstance(teacher_model, neu.Sequential):
                        teacher_preds = teacher_model.forward_pass(X)
                    elif isinstance(teacher_model, Sequential):
                        teacher_preds = teacher_model.forward_pass(X, training=False)
                    else:
                        teacher_model.eval()
                        teacher_preds = teacher_model(X)
                elif data_T is not None:
                    pass
                #     # Slice soft targets from full precomputed data_T
                #     idx_start = (batch - 1) * batch_size
                #     idx_end = idx_start + Y.shape[1]
                #     teacher_preds = data_T[idx_start:idx_end, :].T
                else:
                    raise ValueError("Either teacher_model or data_T must be provided.")
            
                # Compute distillation loss and gradient
                d_loss = self.loss.dL(Y_hat, Y, teacher_preds.copy())
                total_epoch_loss += np.sum(self.loss.L(X_student = Y_hat, y_true = Y, X_teacher=teacher_preds.copy()))

                # 4. Backward pass
                deltas = self.model.backward_pass(d_loss, cache_fields=cache_fields, use_partial_vectors=use_partial_vectors)
                delta_prev = d_loss

                # 5. Adam update
                for layer in reversed(self.model.layers):
                    if isinstance(layer, (OpticalMeshNetworkLayer, )):
                        gradients = layer.mesh.compute_gradients(layer.input_prev, delta_prev,
                                                                cache_fields=cache_fields,
                                                                use_partial_vectors=use_partial_vectors)
                        for cmpt in gradients:
                            self.g[cmpt] = np.mean(gradients[cmpt], axis=-1)
                            self.m[cmpt] = self.beta1 * self.m.get(cmpt, 0) + (1 - self.beta1) * self.g[cmpt]
                            self.v[cmpt] = self.beta2 * self.v.get(cmpt, 0) + (1 - self.beta2) * self.g[cmpt] ** 2
                            mhat = self.m[cmpt] / (1 - self.beta1 ** self.t)
                            vhat = self.v[cmpt] / (1 - self.beta2 ** self.t)

                            grad = -1 * self.step_size * mhat / (np.sqrt(vhat) + self.epsilon)

                            if isinstance(cmpt, PhaseShifter):
                                cmpt.phi = (cmpt.phi + grad[0]) % (2 * pi)

                            elif isinstance(cmpt, (MZI, )):
                                dtheta, dphi = grad
                                cmpt.theta = (cmpt.theta + dtheta) % (2 * pi)
                                cmpt.phi = (cmpt.phi + dphi) % (2 * pi)

                    # Set the backprop signal for the subsequent (spatially previous) layer
                    delta_prev = deltas[layer.__name__]

            # Append loss per epoch
            total_epoch_loss /= n_samples
            losses.append(total_epoch_loss)

            # Append training accuracy per epoch
            if isinstance(self.model, neu.Sequential):
                Y_hat = self.model.forward_pass(data)
            else:
                Y_hat = self.model.forward_pass(data, training = True)
            pred = np.argmax(Y_hat, axis=0)
            gt = np.argmax(labels, axis=0)
            trn_accuracy.append(np.mean(pred == gt) * 100)

            # Append validation accuracy per epoch
            if isinstance(self.model, neu.Sequential):
                Y_hat_val = self.model.forward_pass(val_data)
            else:
                Y_hat_val = self.model.forward_pass(val_data, training = True)
            pred_val = np.argmax(Y_hat_val, axis=0)
            gt_val = np.argmax(val_labels, axis=0)
            val_accuracy.append(np.mean(pred_val == gt_val) * 100)
                
            if val_accuracy[-1] > max(val_accuracy[:-1]):
                best_phases = self.model.get_all_phases()
                best_trf_matrix = self.model.get_transformation_matrix()

            if show_progress:
                iterator.set_description("ℒ = {:.2f} | Acc = {:.2f}".format(total_epoch_loss, val_accuracy[-1]), refresh=True)
        
        print(f'Max Validation Accuracy: {max(val_accuracy):.2f}%')
        return losses, trn_accuracy[1:], val_accuracy[1:], best_phases, best_trf_matrix
    
class InSituAdam_QAT(Optimizer):
    '''
    On-chip training with in-situ backpropagation using adjoint field method and adam optimizer
    '''
    def __init__(self, model: Sequential, loss: Type[Loss], step_size=0.01,
                 beta1=0.9, beta2=0.99, epsilon=1e-8, pKeep=0.8, bit_length = 8, is_diamond = False, train_cfg = None):
        super().__init__(model, loss)
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.pKeep = pKeep

        self.is_diamond = is_diamond

        self.train_cfg = train_cfg if train_cfg is not None else {}
        self.loss_dB=train_cfg.get('loss_dB', 0)
        self.phase_uncert_phi=train_cfg.get('phase_uncert_phi', 0)
        self.phase_uncert_theta=train_cfg.get('phase_uncert_theta', 0)
        
        self.phase_levels = get_phase_levels(b=bit_length)
        set_phases_quantized(self.model, phase_levels=self.phase_levels, is_diamond=self.is_diamond, loss_dB=self.loss_dB,
                            phase_uncert_phi=self.phase_uncert_phi, phase_uncert_theta=self.phase_uncert_theta)


        self.t = 0
        self.m = {}
        self.v = {}
        self.g = {}
        self.old_model = deepcopy(model)
        self.bit_length = bit_length

        for layer in self.model.layers:
            if isinstance(layer, OpticalMeshNetworkLayer):
                for component in layer.mesh.all_tunable_components():
                    self.m[component] = np.zeros(component.dof)
                    self.v[component] = np.zeros(component.dof)
                    self.g[component] = np.zeros(component.dof)
        # print(self.m.keys())
        print('\n')

    def update_lib(self, model: Sequential):
        self.m = {}
        self.v = {}
        self.g = {}
        self.model = model
        for layer in self.model.layers:
            if isinstance(layer, OpticalMeshNetworkLayer):
                for component in layer.mesh.all_tunable_components():
                    self.m[component] = np.zeros(component.dof)
                    self.v[component] = np.zeros(component.dof)
                    self.g[component] = np.zeros(component.dof)
        # print(self.m.keys())
        print('\n')

    def fit(self, data: np.ndarray, labels: np.ndarray, val_data: np.ndarray,
            val_labels: np.ndarray, epochs=1000, batch_size=32,
            show_progress=True, cache_fields=False, use_partial_vectors=False):
        '''
        Fit the model to the labeled data
        :param data: features vector, shape: (n_features, n_samples)
        :param labels: labels vector, shape: (n_label_dim, n_samples)
        :param epochs:
        :param batch_size:
        :param show_progress:
        :param cache_fields: if set to True, will cache fields at the phase shifters on the forward and backward pass
        :param use_partial_vectors: if set to True, the MZI partial matrices will be stored as Nx2 vectors
        :return: losses, accuracy
        '''
        losses = []
        trn_accuracy = [0]
        val_accuracy = [0]

        best_phases = self.model.get_all_phases()
        best_trf_matrix = self.model.get_transformation_matrix()
        n_features, n_samples = data.shape
        iterator = range(epochs)
        if show_progress: iterator = pbar(iterator)

        itr = 0
        for epoch in iterator:

            total_epoch_loss = 0.0
            batch = 0
            for X, Y in self.make_batches(data, labels, batch_size):
                batch += 1
                self.t += 1

                # Propagate the data forward
                Y_hat = self.model.forward_pass(X, cache_fields=cache_fields, use_partial_vectors=use_partial_vectors)
                d_loss = self.loss.dL(Y_hat, Y)
                total_epoch_loss += np.sum(self.loss.L(Y_hat, Y))

                # Compute the backpropagated signals for the model
                deltas = self.model.backward_pass(d_loss, cache_fields=cache_fields,
                                                  use_partial_vectors=use_partial_vectors)
                delta_prev = d_loss  # backprop signal to send in the final layer

                # Compute the foward and adjoint fields at each phase shifter in all tunable layers
                for layer in reversed(self.model.layers):
                    if isinstance(layer, OpticalMeshNetworkLayer):
                        gradients = layer.mesh.compute_gradients(layer.input_prev, delta_prev,
                                                                 cache_fields=cache_fields,
                                                                 use_partial_vectors=use_partial_vectors)

                        for cmpt in gradients:
                            # print("M:", self.m.keys())
                            # print(gradients.keys())
                            self.g[cmpt] = np.mean(gradients[cmpt], axis=-1)
                            self.m[cmpt] = self.beta1 * self.m[cmpt] + (1 - self.beta1) * self.g[cmpt]
                            self.v[cmpt] = self.beta2 * self.v[cmpt] + (1 - self.beta2) * self.g[cmpt] ** 2
                            mhat = self.m[cmpt] / (1 - self.beta1 ** self.t)
                            vhat = self.v[cmpt] / (1 - self.beta2 ** self.t)

                            grad = -1 * self.step_size * mhat / (np.sqrt(vhat) + self.epsilon)

                            # Adjust settings by gradient amount
                            # cliping to the nearest phase value
                            if isinstance(cmpt, PhaseShifter):
                                cmpt.phi += grad[0]
                                if cmpt.phi < 0:
                                    cmpt.phi += 2*pi

                                if cmpt.phi > 2*pi:
                                    cmpt.phi -= 2*pi

                                cmpt.phi = find_nearest(self.phase_levels, cmpt.phi)

                            elif isinstance(cmpt,(MZI, )):
                                dtheta, dphi = grad


                                dtheta, dphi = grad
                                if cmpt.phi + dphi < 0:
                                    cmpt.phi += 2*pi
                                if cmpt.phi + dphi > 2*pi:
                                    cmpt.phi -= 2*pi
                                if cmpt.theta + dtheta < 0:
                                    cmpt.theta += 2*pi
                                if cmpt.theta + dtheta > 2*pi:
                                    cmpt.theta -= 2*pi

                                cmpt.phi += dphi
                                cmpt.theta += dtheta
                                cmpt.phi = find_nearest(self.phase_levels, cmpt.phi)
                                cmpt.theta = find_nearest(self.phase_levels, cmpt.theta)

                    # Set the backprop signal for the subsequent (spatially previous) layer
                    delta_prev = deltas[layer.__name__]

            # Append loss per epoch
            total_epoch_loss /= n_samples
            losses.append(total_epoch_loss)

            accs = []
            for _ in range(20):
                Y_hat = self.model.forward_pass(data)
                pred = np.argmax(Y_hat, axis=0)
                gt = np.argmax(labels, axis=0)
                accs.append(np.mean(pred == gt) * 100)
            trn_accuracy.append(np.mean(accs))

            accs = []
            for _ in range(20): 
                Y_hat_val = self.model.forward_pass(val_data)
                pred_val = np.argmax(Y_hat_val, axis=0)
                gt_val = np.argmax(val_labels, axis=0)
                accs.append(np.mean(pred_val == gt_val) * 100)
            val_accuracy.append(np.mean(accs))

            if val_accuracy[-1] > max(val_accuracy[:-1]):
                best_phases = self.model.get_all_phases()
                best_trf_matrix = self.model.get_transformation_matrix()

            if show_progress:
                iterator.set_description("ℒ = {:.2f} | Acc = {:.2f}".format(total_epoch_loss, val_accuracy[-1]), refresh=True)

            # if itr == 0:
            #     global begin_phase
            #     begin_phase = self.model.get_all_phases()
            # elif itr == epochs - 1:
            #     global end_phase
            #     end_phase = self.model.get_all_phases()
            #print(self.model.get_all_phases())
            itr += 1

        print(f'Max Validation Accuracy: {max(val_accuracy):.2f}%')
        trn_accuracy = trn_accuracy[1:]
        val_accuracy = val_accuracy[1:]
        return losses, trn_accuracy, val_accuracy, best_phases, best_trf_matrix
    
class InSituAdam_QAT_KD(Optimizer):
    def __init__(self, model: Sequential, loss: Type[Loss], step_size=0.01,
                 beta1=0.9, beta2=0.99, epsilon=1e-8, pKeep=0.8,
                 bit_length=8, is_diamond=False, phase_quant=True, train_cfg=None):
        super().__init__(model, loss)
        self.step_size = step_size
        self.initial_step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.pKeep = pKeep
        self.t = 0
        self.m = {}
        self.v = {}
        self.g = {}

        self.phase_quant = phase_quant
        self.bit_length = bit_length
        self.is_diamond = is_diamond
        self.train_cfg = train_cfg

        self.loss_dB=train_cfg.get('loss_dB', 0)
        self.phase_uncert_phi=train_cfg.get('phase_uncert_phi', 0)
        self.phase_uncert_theta=train_cfg.get('phase_uncert_theta', 0)
        
        self.phase_levels = get_phase_levels(b=bit_length) if self.phase_quant else None
        if self.phase_quant:
            set_phases_quantized(self.model, phase_levels=self.phase_levels, is_diamond=self.is_diamond, loss_dB=self.loss_dB,
                             phase_uncert_phi=self.phase_uncert_phi, phase_uncert_theta=self.phase_uncert_theta)

        for layer in model.layers:
            if isinstance(layer, OpticalMeshNetworkLayer):
                for component in layer.mesh.all_tunable_components():
                    self.m[component] = np.zeros(component.dof)
                    self.v[component] = np.zeros(component.dof)
                    self.g[component] = np.zeros(component.dof)

    def make_batches_all(self, data, labels, teacher_data=None, batch_size=32, shuffle=True):
        n_features, n_samples = data.shape
        np.random.seed(21)
        batch_indices = np.arange(0, n_samples, batch_size)

        if shuffle:
            permutation = np.random.permutation(n_samples)
            data = data[:, permutation]
            labels = labels[:, permutation]
            if teacher_data is not None:
                teacher_data = teacher_data[:, permutation]

        for i in batch_indices:
            X = data[:, i:i + batch_size]
            Y = labels[:, i:i + batch_size]
            teacher_sample = teacher_data[:, i:i + batch_size] if teacher_data is not None else None
            yield X, Y, teacher_sample

    def fit(self, data, labels, val_data, val_labels, train_cfg):
        epochs = train_cfg.get("epochs", 1000)
        batch_size = train_cfg.get("batch_size", 32)
        show_progress = train_cfg.get("show_progress", True)
        cache_fields = train_cfg.get("cache_fields", False)
        use_partial_vectors = train_cfg.get("use_partial_vectors", False)
        lr_scheduler = train_cfg.get("lr_scheduler", None)

        teacher_model = train_cfg.get("teacher_model", None)
        data_T = train_cfg.get("data_T", None)

        
        losses = []
        trn_accuracy = [0]
        val_accuracy = [0]

        best_phases = self.model.get_all_phases()
        best_trf_matrix = self.model.get_transformation_matrix()

        if data_T is not None:
            teacher_sample = data_T.T
        else:
            teacher_sample = None

        n_features, n_samples = data.shape
        iterator = range(epochs)
        if show_progress: iterator = pbar(iterator)

        for epoch in iterator:
            if lr_scheduler:
                self.step_size = lr_scheduler(epoch, self.initial_step_size)

            total_epoch_loss = 0.0
            for X, Y, teacher_preds in self.make_batches_all(data, labels, teacher_sample, batch_size):
                self.t += 1
                Y_hat = self.model.forward_pass(X, cache_fields=cache_fields, use_partial_vectors=use_partial_vectors)

                if teacher_model is not None:
                    teacher_preds = teacher_model.forward_pass(X) if hasattr(teacher_model, "forward_pass") else teacher_model(X)
                elif data_T is None:
                    raise ValueError("Either teacher_model or data_T must be provided.")

                d_loss = self.loss.dL(Y_hat, Y, teacher_preds.copy())
                total_epoch_loss += np.sum(self.loss.L(Y_hat, Y, teacher_preds.copy()))

                deltas = self.model.backward_pass(d_loss, cache_fields=cache_fields, use_partial_vectors=use_partial_vectors)
                delta_prev = d_loss

                for layer in reversed(self.model.layers):
                    if isinstance(layer, OpticalMeshNetworkLayer):
                        gradients = layer.mesh.compute_gradients(layer.input_prev, delta_prev, cache_fields, use_partial_vectors)
                        for cmpt in gradients:
                            self.g[cmpt] = np.mean(gradients[cmpt], axis=-1)
                            self.m[cmpt] = self.beta1 * self.m[cmpt] + (1 - self.beta1) * self.g[cmpt]
                            self.v[cmpt] = self.beta2 * self.v[cmpt] + (1 - self.beta2) * self.g[cmpt] ** 2
                            mhat = self.m[cmpt] / (1 - self.beta1 ** self.t)
                            vhat = self.v[cmpt] / (1 - self.beta2 ** self.t)
                            grad = -self.step_size * mhat / (np.sqrt(vhat) + self.epsilon)

                            if isinstance(cmpt, PhaseShifter):
                                cmpt.phi = (cmpt.phi + grad[0]) % (2 * pi)
                                if self.phase_quant:
                                    cmpt.phi = find_nearest(self.phase_levels, cmpt.phi)

                            elif isinstance(cmpt, (MZI, )):
                                dtheta, dphi = grad
                                cmpt.theta = (cmpt.theta + dtheta) % (2 * pi)
                                cmpt.phi = (cmpt.phi + dphi) % (2 * pi)
                                if self.phase_quant:
                                    cmpt.theta = find_nearest(self.phase_levels, cmpt.theta)
                                    cmpt.phi = find_nearest(self.phase_levels, cmpt.phi)

                    delta_prev = deltas[layer.__name__]

            total_epoch_loss /= n_samples
            losses.append(total_epoch_loss)

            accs = []
            for _ in range(20):
                Y_hat = self.model.forward_pass(data)
                pred = np.argmax(Y_hat, axis=0)
                gt = np.argmax(labels, axis=0)
                accs.append(np.mean(pred == gt) * 100)
            trn_accuracy.append(np.mean(accs))

            accs = []
            for _ in range(20): 
                Y_hat_val = self.model.forward_pass(val_data)
                pred_val = np.argmax(Y_hat_val, axis=0)
                gt_val = np.argmax(val_labels, axis=0)
                accs.append(np.mean(pred_val == gt_val) * 100)
            val_accuracy.append(np.mean(accs))

            if val_accuracy[-1] > max(val_accuracy[:-1]):
                # set_phases_quantized(self.model, phase_levels=self.phase_levels, is_diamond=self.is_diamond)
                best_phases = self.model.get_all_phases()
                best_trf_matrix = self.model.get_transformation_matrix()

            if show_progress:
                iterator.set_description("ℒ = {:.2f}| Acc = {:.2f}".format(total_epoch_loss, val_accuracy[-1]), refresh=True)

        print(f'Max Validation Accuracy: {max(val_accuracy):.2f}%')
        return losses, trn_accuracy[1:], val_accuracy[1:], best_phases, best_trf_matrix

    '''
    Example of use:
    train_cfg = {
    "epochs": 1000,
    "batch_size": 64,
    "alpha": 0.7,
    "temperature": 3.0,
    "teacher_model": my_teacher_model,  # OR: "data_T": soft_targets
    "show_progress": True,
    "cache_fields": False
    }

    loss = DistillationLoss(alpha=train_cfg.get("alpha", 0.7), temperature=train_cfg.get("temperature", 3))
    optimizer = InSituAdamLossAware(model, loss)
    optimizer.fit(train_data, train_labels, val_data, val_labels, train_cfg)
    '''