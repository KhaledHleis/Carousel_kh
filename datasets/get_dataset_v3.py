import math

import numpy as np
import matplotlib.pyplot as plt

from motion_optimal import motion_optimal
from datasets.get_dataset_v2 import get_dataset_v2
from datasets.get_dataset_v1 import get_dataset_v1

def get_dataset_v3(N: int, K: float = 1, u_bar: float = 5, R: float= 10, std_circle: float = 1) -> tuple[np.ndarray, np.ndarray]:
    "Half of the data sample randomly everywhere, half sample centred on the circle"
    half_N = N // 2
    print(f'Generates {half_N} points with dataset_v1...')
    data1_in, data1_out = get_dataset_v1(half_N, K, u_bar, R)
    print(f'Generates {half_N} points with dataset_v2...')
    data2_in, data2_out = get_dataset_v2(half_N, K, u_bar, R, std_circle=std_circle)
    return np.concatenate((data1_in, data2_in), axis=0), np.concatenate((data1_out, data2_out), axis=0)


if __name__ == '__main__':
    inputs, outputs = get_dataset_v3(1000, std_circle=2)
    plt.figure(figsize=(6, 6))
    plt.scatter(inputs[:, 0], inputs[:, 1], s=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Dataset Visualization')
    plt.show()
    print(inputs.shape)
    print(outputs.shape)