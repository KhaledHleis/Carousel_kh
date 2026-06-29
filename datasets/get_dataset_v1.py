import math
from random import uniform

import numpy as np

from motion_optimal import motion_optimal

def get_dataset_v1(N: int, K: float = 1, u_bar: float = 5, R: float= 10) -> tuple[np.ndarray, np.ndarray]:
    inputs = []
    outputs = []
    limit = 100
    for _ in range(N):
        x = uniform(-limit, limit)
        y = uniform(-limit, limit)
        theta = uniform(0, 2 * math.pi)
        inputs.append(get_input_line(x, y, theta))
        u1, u2 = motion_optimal(x, y, theta, K, u_bar, R)
        outputs.append([u1, u2])
    return np.array(inputs, dtype=np.float32), np.array(outputs, dtype=np.float32)


def get_input_line(x: float, y: float, theta: float) -> list[float]:
    return [x, y, math.cos(theta), math.sin(theta)]


if __name__ == '__main__':
    inputs, outputs = get_dataset_v1(10)
    print(inputs.shape, inputs)
    print(outputs.shape, outputs)


