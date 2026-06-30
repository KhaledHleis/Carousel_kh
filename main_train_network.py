from torch import from_numpy
import matplotlib.pyplot as plt

from datasets.get_dataset_v1 import get_dataset_v1
from datasets.get_dataset_v2 import get_dataset_v2
from datasets.get_dataset_v3 import get_dataset_v3
from neural_net_pytorch_ import NeuralNetwork

# both NumPy (MKL) and PyTorch ship their own libiomp5md.dll, and Windows refuses to load it twice. It's not a bug in your code or the simulator.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def cost_fn(prediction, true):
    return ((prediction - true) ** 2).sum()


def train_network(inputs, outputs, epochs: int, lr: float, network_name: str, verbose: bool = False):
    model = NeuralNetwork()
    X, y = from_numpy(inputs), from_numpy(outputs)
    history = model.train_model(X, y, cost_fn=cost_fn, epochs=epochs, lr=lr,verbose=verbose)
    model.register_to_csv(f"models/{network_name}.csv")
    
    plt.figure(figsize=(8, 6))
    plt.plot(history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Loss")
    plt.grid()
    plt.show()

def main_train_network():
    # Dataset settings
    N = 1_000
    # N = 1_000
    version = 3
    K = 0.3
    u_bar = 0.5
    R = 10
    # Training settings
    epochs = 1_000
    lr = 0.005
    # Training of the network
    print('Start generating dataset')
    get_dataset = [get_dataset_v1, get_dataset_v2, get_dataset_v3][version - 1]
    inputs, outputs = get_dataset(N, K, u_bar, R)
    print('Dataset generated !')
    print('Start training pytorch model')
    network_name = f'network_v{version}_{N}_{10 * K}_{10 * u_bar}_{R}_{epochs}'
    train_network(inputs, outputs, epochs, lr, network_name, verbose=True)
    print('pytorch model trained !')



if __name__ == '__main__':
    main_train_network()
