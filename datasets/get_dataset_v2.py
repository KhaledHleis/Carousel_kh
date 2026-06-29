import math

import numpy as np
import matplotlib.pyplot as plt

from motion_optimal import motion_optimal

def get_dataset_v2(N: int, K: float = 1, u_bar: float = 5, R: float= 10, std_circle: float = 1) -> tuple[np.ndarray, np.ndarray]:
    inputs = []
    outputs = []
    for _ in range(N):
        theta = np.random.uniform(0, 2 * math.pi)
        r = np.random.normal(R, std_circle)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        inputs.append([x, y, np.cos(theta), np.sin(theta)])
        u1, u2 = motion_optimal(x, y, theta, K, u_bar, R)
        outputs.append([u1, u2])
    return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)

if __name__ == '__main__':
    inputs, outputs = get_dataset_v2(10)
    plt.figure(figsize=(6, 6))
    plt.scatter(inputs[:, 0], inputs[:, 1], s=10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Dataset Visualization')
    plt.show()
    print(inputs.shape)
    print(outputs.shape)