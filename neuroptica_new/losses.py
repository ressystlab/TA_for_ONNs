'''The Losses submodule contains classes for computing common loss functions.'''

import numpy as np

class Loss:
    @staticmethod
    def L(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        '''
        The scalar, real-valued loss function (vectorized over multiple X, T inputs)
        :param X: the output of the network
        :param T: the target output
        :return: loss function for each X
        '''
        raise NotImplementedError("Loss function must be specified in child class")

    @staticmethod
    def dL(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        '''
        The derivative of the loss function dL/dX_L used for backpropagation (vectorized over multiple X)
        :param X: the output of the network
        :param T: the target output
        :return: dL/dX_L for each X
        '''
        raise NotImplementedError("Derivative loss function must be specified in child class")

class MeanSquaredError(Loss):
    @staticmethod
    def L(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        return np.sum(1 / 2 * np.abs(T - X) ** 2, axis=0)

    @staticmethod
    def dL(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        #print(X.shape, T.shape)
        return np.conj(X - T)

class CategoricalCrossEntropy(Loss):
    '''Represents categorical cross entropy with a softmax layer implicitly applied to the outputs'''

    @staticmethod
    def L(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        X_softmax = np.exp(X) / np.sum(np.exp(X), axis=0)
        tol = 1e-10
        X_clip = np.clip(X_softmax, tol, 1 - tol)
        return -np.sum(T * np.log(X_clip), axis=0)

    @staticmethod
    def dL(X: np.ndarray, T: np.ndarray) -> np.ndarray:
        X_softmax = np.exp(X) / np.sum(np.exp(X), axis=0)
        return np.conj(X_softmax - T)

def softmax_with_temperature(X, Temperature):
    return np.exp(X/Temperature) / np.sum(np.exp(X/Temperature), axis=0)

class KLDivergence(Loss):
    '''
    Represents KL divergence with a softmax layer implicitly applied to the outputs
    '''

    @staticmethod
    def L(X_student: np.ndarray, X_teacher: np.ndarray, Temperature: np.float32) -> np.ndarray:
        '''
            Calculate the KL divergence based on teacher and student output logits

            Args:
            X_student: student output logits
            X_teacher: teacher output logits
            Temperature: temperature for softmax
        '''
        # Apply softmax with temperature
        p_teacher = softmax_with_temperature(X_teacher, Temperature)
        p_student = softmax_with_temperature(X_student, Temperature)

        # Compute KL divergence
        epsilon = 1e-10
        kl = np.sum(p_teacher * np.log(p_teacher / (p_student + epsilon) + epsilon), axis = 0)  # epsilon to avoid log(0)
        
        # Scale by temperature^2 as done in distillation loss
        return kl * (Temperature ** 2)

    @staticmethod
    def dL(X_student: np.ndarray, X_teacher: np.ndarray, Temperature: np.float32) -> np.ndarray:
        p_teacher = softmax_with_temperature(X_teacher, Temperature)
        p_student = softmax_with_temperature(X_student, Temperature)

        # Gradient of KL(P_teacher || P_student) w.r.t student logits
        grad = Temperature * (p_student - p_teacher)  # scaled by T
        return grad
    
class DistillationLoss(Loss):
    '''
    Combines standard cross-entropy loss with KL divergence for knowledge distillation
    '''
    def __init__(self, alpha=0.5, temperature=1.0):
        '''
        alpha: weight for KL divergence (soft target), (1 - alpha) for cross-entropy (hard target)
        temperature: temperature for softening the distributions
        '''
        self.alpha = alpha
        self.temperature = temperature
        self.CE_loss = CategoricalCrossEntropy()

    def L(self, X_student: np.ndarray, y_true: np.ndarray, X_teacher: np.ndarray) -> float:
        '''
        X_student: logits from student model
        y_true: one-hot ground-truth labels
        X_teacher: logits from teacher model
        '''
        ce_loss = self.CE_loss.L(X_student, y_true)
        kl_loss = KLDivergence.L(X_student, X_teacher, self.temperature)

        # Combine the losses
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        return total_loss

    def dL(self, X_student: np.ndarray, y_true: np.ndarray, X_teacher: np.ndarray) -> np.ndarray:
        '''
        Compute the gradient of the combined loss w.r.t. student logits
        '''
        ce_grad = self.CE_loss.dL(X_student, y_true)
        kl_grad = KLDivergence.dL(X_student, X_teacher, self.temperature)

        # Combine gradients
        total_grad = (1 - self.alpha) * ce_grad + self.alpha * kl_grad
        return total_grad
    