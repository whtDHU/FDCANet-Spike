import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from load_data import PathSpectogramFolder, patients, loadSpectogramData

OutputPathModels = "./EggModels"


class MITDataset(Dataset):
    def __init__(self, data, labels):
        """
        数据集对象
        :param data: 数据
        :param labels: 标签
        """
        super(MITDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


def load_data(patient_index):
    print('Patient ' + patients[patient_index])
    interictalSpectograms, preictalSpectograms, _ = loadSpectogramData(patient_index)
    print('Spectograms data loaded')
    filesPath = []
    interictalSpectograms = [item for sublist in interictalSpectograms for item in sublist]
    preictalSpectograms = [item for sublist in preictalSpectograms for item in sublist]
    filesPath.extend(interictalSpectograms)
    filesPath.extend(preictalSpectograms)
    print(filesPath)
    all_label = []

    from concurrent.futures import ThreadPoolExecutor

    def load_array(file_path):
        label = []
        array = np.load(PathSpectogramFolder + file_path).astype(np.float32)
        label.extend([1 if 'P' in file_path else 0] * array.shape[0])
        return array, label

    with ThreadPoolExecutor(max_workers=16) as executor:
        results_all = list(executor.map(load_array, filesPath))

    all_data = np.concatenate([item[0] for item in results_all], axis=0).astype(np.float32)
    for item in results_all:
        all_label.extend(item[1])
    random_indices = np.random.permutation(len(all_data))
    shuffled_data = all_data[random_indices]
    shuffled_label = [all_label[i] for i in random_indices]

    return shuffled_data, shuffled_label


def cross_validation(patient_index, n_splits=10):
    data, labels = load_data(patient_index)
    kf = KFold(n_splits=n_splits)

    return data, labels, kf


if __name__ == "__main__":
    # Example usage
    for patient_index in range(len(patients)):
        cross_validation(patient_index)
