import math
from random import uniform

import numpy as np


def get_dataset_v1(N: int, K: float, u_bar: float, R: float) -> tuple[np.ndarray, np.ndarray]:
    inputs = []
    outputs = []
    limit = 100
    for _ in range(N):
        x = uniform(-limit, limit)
        y = uniform(-limit, limit)
        theta = uniform(0, 2 * math.pi)
        inputs.append([x, y, math.cos(theta), math.sin(theta)])
        outputs.append([1, -1])
    return np.array(inputs), np.array(outputs)

if __name__ == '__main__':
    inputs, outputs = get_dataset_v1(10, 2, 2, 2)
    print(inputs.shape, inputs)
    print(outputs.shape, outputs)


