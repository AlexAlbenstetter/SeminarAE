import numpy as np
from river.utils.data_conversion import numpy2dict
from sklearn.utils import resample
import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader


def build_anomaly_dataset(x, y, y_clean=1, frac_anom=0.3, batch_size=1, seed=42):
    x_clean = x[y == y_clean]
    y_anom = y[y != y_clean]
    x_anom = x[y != y_clean]

    n_clean = len(x_clean)
    n_anom = int((frac_anom * n_clean) / (1.0 - frac_anom))

    x_anom = resample(x_anom, replace=False, n_samples=n_anom,
                      stratify=y_anom, random_state=seed)

    x_all = np.concatenate((x_clean, x_anom), axis=0)
    is_anom = np.concatenate(
        (np.zeros(n_clean, dtype=int), np.ones(n_anom, dtype=int)))

    x_all, is_anom = resample(x_all, is_anom, replace=False, random_state=seed)

    return build_dataloader(torch.tensor(x_all), torch.tensor(is_anom), batch_size=batch_size)


def build_dataloader(x, y, batch_size=1, shuffle=False):
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def conv_output_size(input_height, padding, k_size, stride):
    return ((input_height - k_size + 2 * padding) / stride + 1)


def deconv_output_size(input_height, padding, k_size, stride, output_padding=0):
    return (input_height - 1) * stride - 2*padding+(k_size-1)+output_padding+1


class Tensor2Dict():
    @property
    def _supervised(self):
        return False

    def learn_one(self, x):
        pass

    def transform_one(self, x):
        return numpy2dict(torch.flatten(x))
