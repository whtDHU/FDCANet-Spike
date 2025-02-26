import numpy as np


def get_start_indices(sequences):  # list:[4,...,5,...6,...,4,...,5,...,6] when skip sequence 1,2,3
    indices = [0]  # [0,6564,13128,19692,26256,32820]
    for i in range(1, len(sequences)):
        if (sequences[i - 1] == 6) and (sequences[i] == 4):  # (sequences[i-1] == 6) and (sequences[i] == 1)
            indices.append(i)
    indices.append(len(sequences))  # Get the index of the different seizures starts
    return indices


def group_seizure(X, y, sequences):  # X=list{32820} All samples of preictal
    Xg = []  # grouping X
    yg = []
    start_indices = get_start_indices(sequences)  # 每次癫痫发作的起始索引
    print('start_indices', start_indices)
    print(len(X), len(y))
    for i in range(len(start_indices) - 1):
        Xg.append(
            np.concatenate(X[start_indices[i]:start_indices[i + 1]], axis=0)
        )
        yg.append(
            np.array(y[start_indices[i]:start_indices[i + 1]])
        )
    return Xg, yg
