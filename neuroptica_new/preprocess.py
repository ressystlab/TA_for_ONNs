import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def normalize_to_power(data, power_sum = 10, epsilon=1e-4):
    # Min-Max normalize each row to [0, 1]
    data_min = np.min(data, axis=1, keepdims=True)
    data_max = np.max(data, axis=1, keepdims=True)
    normalized = (data - data_min) / (data_max - data_min + 1e-8)

    # Shift to avoid zeros: values in [epsilon, 1]
    shifted = normalized * (1 - epsilon) + epsilon

    # Scale so that the max squared value of each data instance equals power_sum/N
    N = data.shape[1]
    scaled = shifted * np.sqrt(power_sum / N)

    return scaled

def dataset_to_numpy(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset))
    images, labels = next(iter(loader))
    images = images.view(images.size(0), -1)  # Flatten to [N, 784]
    return images.numpy(), labels.numpy()