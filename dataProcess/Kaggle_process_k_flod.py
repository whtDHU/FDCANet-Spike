import json
import os

import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from utils.save_load import load_hickle_file

with open('../SETTINGS_%s.json' % "Kaggle") as f:
    settings = json.load(f)

# 这是真正的指定处
targets = [
    'Dog_1',
    'Dog_2',
    'Dog_3',
    'Dog_4',
    'Dog_5',
]


class KaggleDataset(Dataset):
    def __init__(self, data, labels):
        """
        数据集对象
        :param data: 数据
        :param labels: 标签
        """
        super(KaggleDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


# 数据预处理
def pre_option(patient_index):
    target = targets[patient_index]
    ictal_X, ictal_y = load_hickle_file(os.path.join(settings['cachedir'], f'ictal_{target}'))

    X_train_ictal = np.concatenate(ictal_X[:], axis=0).astype(np.float32)
    y_train_ictal = np.concatenate(ictal_y[:], axis=0).astype(np.float32)
    del ictal_X, ictal_y
    interictal_X, interictal_y = load_hickle_file(os.path.join(settings['cachedir'], f'interictal_{target}'))
    if isinstance(interictal_y, list):
        interictal_X = np.concatenate(interictal_X, axis=0).astype(np.float32)
        interictal_y = np.concatenate(interictal_y, axis=0).astype(np.float32)

    X_train_interictal = None
    y_train_interictal = None
    down_spl = int(np.floor(interictal_y.shape[0] / y_train_ictal.shape[0]))
    if down_spl > 1:
        X_train_interictal = interictal_X[::down_spl]
        y_train_interictal = interictal_y[::down_spl]
    elif down_spl == 1:
        X_train_interictal = interictal_X[:X_train_ictal.shape[0]]
        y_train_interictal = interictal_y[:X_train_ictal.shape[0]]
    print('balancing y_train_ictal.shape, y_train_interictal.shape: ', X_train_ictal.shape,
          y_train_interictal.shape)
    del interictal_X, interictal_y

    y_train_ictal[y_train_ictal == 2] = 1
    all_data = np.concatenate((X_train_ictal, X_train_interictal), axis=0).astype(np.float32)
    del X_train_ictal, X_train_interictal
    all_label = np.concatenate((y_train_ictal, y_train_interictal), axis=0).astype(np.float32)
    del y_train_ictal, y_train_interictal

    # 打乱所有数据和标签
    random_indices = np.random.permutation(all_data.shape[0])
    shuffled_data = all_data[random_indices]
    shuffled_label = [all_label[i] for i in random_indices]
    del all_data
    return shuffled_data, shuffled_label

    # 分割数据为训练集、验证集和测试集的KFold


def cross_validation(patient_index, n_splits=10):
    data, labels = pre_option(patient_index)
    kf = KFold(n_splits=n_splits)

    return data, labels, kf


if __name__ == "__main__":
    # Example usage
    cross_validation(1)
