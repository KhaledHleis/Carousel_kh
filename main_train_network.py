from torch import from_numpy

from datasets.get_dataset_v1 import get_dataset_v1
from neural_net_pytorch_ import NeuralNetwork


def cost_fn(prediction, true):
    return (prediction - true) ** 2


def train_network(inputs, outputs):
    model = NeuralNetwork()
    X = from_numpy(inputs)
    y = from_numpy(outputs)
    model.train_model(X, y, cost_fn=cost_fn, epochs=1)
    model.register_to_csv("network_v1.csv")

def main_train_network():
    inputs, outputs = get_dataset_v1(10, 2, 2, 2)
    train_network(inputs, outputs)


if __name__ == '__main__':
    main_train_network()
