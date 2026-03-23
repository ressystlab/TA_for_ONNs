import numpy as np
import matplotlib.pyplot as plt

def step_decay(epoch, initial_lr=0.01, drop=0.5, epochs_drop=10):
    """
    Reduces learning rate every `epochs_drop` epochs by a factor `drop`.
    :param epoch: Current epoch number
    :param initial_lr: Initial learning rate
    :param drop: Multiplicative factor of learning rate decay
    :param epochs_drop: Number of epochs after which learning rate is reduced
    :return: New learning rate
    """
    return initial_lr * (drop ** (epoch // epochs_drop))


def exponential_decay(epoch, initial_lr=0.01, decay_rate=0.1, epochs_drop=10):
    """
    Exponentially decays the learning rate.
    :param epoch: Current epoch number
    :param initial_lr: Initial learning rate
    :param decay_rate: Rate of decay
    :return: New learning rate
    """
    return initial_lr * np.exp(-decay_rate * (epoch // epochs_drop))

def polynomial_decay(epoch, initial_lr=0.01, max_epochs=100, power=2):
    """
    Decays learning rate polynomially.
    :param epoch: Current epoch number
    :param initial_lr: Initial learning rate
    :param max_epochs: Total number of epochs for decay
    :param power: Power of the polynomial
    :return: New learning rate
    """
    return initial_lr * (1 - (epoch / max_epochs)) ** power

def inverse_time_decay(epoch, initial_lr=0.01, decay_rate=0.1):
    """
    Decays learning rate inversely with time.
    :param epoch: Current epoch number
    :param initial_lr: Initial learning rate
    :param decay_rate: Rate of decay
    :return: New learning rate
    """
    return initial_lr / (1 + decay_rate * epoch)

def cosine_annealing(epoch, initial_lr=0.01, max_epochs=100):
    """
    Cosine annealing schedule for learning rate decay.
    :param epoch: Current epoch number
    :param initial_lr: Initial learning rate
    :param max_epochs: Total number of epochs
    :return: New learning rate
    """
    return initial_lr * (0.5 * (1 + np.cos(np.pi * epoch / max_epochs)))

def cyclical_lr(epoch, min_lr=0.001, max_lr=0.01, step_size=10):
    """
    Cyclical learning rate.
    :param epoch: Current epoch number
    :param min_lr: Minimum learning rate
    :param max_lr: Maximum learning rate
    :param step_size: Number of epochs per half-cycle
    :return: New learning rate
    """
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    return min_lr + (max_lr - min_lr) * max(0, (1 - x))

# Function to plot learning rate variations
def plot_lr_variations(epochs, initial_lr=0.01, save_path = None):
    x = np.arange(epochs)

    # Generate learning rate values for each method
    step = [step_decay(epoch, initial_lr=initial_lr, drop=0.5, epochs_drop=10) for epoch in x]
    exp = [exponential_decay(epoch, initial_lr=initial_lr, decay_rate=0.1) for epoch in x]
    poly = [polynomial_decay(epoch, initial_lr=initial_lr, max_epochs=epochs, power=2) for epoch in x]
    inv_time = [inverse_time_decay(epoch, initial_lr=initial_lr, decay_rate=0.1) for epoch in x]
    cosine = [cosine_annealing(epoch, initial_lr=initial_lr, max_epochs=epochs) for epoch in x]
    cyclical = [cyclical_lr(epoch, min_lr=0.001, max_lr=initial_lr, step_size=10) for epoch in x]

    # Plot the learning rates
    plt.figure(figsize=(8, 6))
    plt.plot(x, step, label="Step Decay", linestyle="--", color="b")
    plt.plot(x, exp, label="Exponential Decay", linestyle=":", color="g")
    plt.plot(x, poly, label="Polynomial Decay", linestyle="-.", color="r")
    plt.plot(x, inv_time, label="Inverse Time Decay", linestyle="-", color="c")
    plt.plot(x, cosine, label="Cosine Annealing", linestyle="--", color="m")
    plt.plot(x, cyclical, label="Cyclical LR", linestyle=":", color="y")

    # Customize the plot
    plt.title("Learning Rate Variations")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi = 600)
    plt.show()
