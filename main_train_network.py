from torch import from_numpy

from datasets.get_dataset_v1 import get_dataset_v1
from neural_net_pytorch_ import NeuralNetwork


def cost_fn(prediction, true):
    return ((prediction - true) ** 2).sum()


def train_network(inputs, outputs, epochs: int, lr: float, network_name: str):
    model = NeuralNetwork()
    X, y = from_numpy(inputs), from_numpy(outputs)
    model.train_model(X, y, cost_fn=cost_fn, epochs=epochs, lr=lr)
    model.register_to_csv(f"models/{network_name}.csv")

def main_train_network(version: int = 1, N: int = 100000, epochs: int = 100000, lr: float = 0.001):
    # Dataset settings
    version = 1
    K = 4
    u_bar = 20
    R = 10
    # Training settings
    epochs = 100_000
    lr = 0.001
    # Training of the network
    inputs, outputs = get_dataset_v1(N, K, u_bar, R)
    network_name = f'network_v{version}_{N}_{K}_{u_bar}_{R}_{epochs}'
    train_network(inputs, outputs, epochs, lr, network_name)


if __name__ == '__main__':
    main_train_network()
