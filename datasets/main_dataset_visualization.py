from datasets.get_dataset_v3 import get_dataset_v3
import matplotlib.pyplot as plt

def main_dataset_visualization():
    # N = 1_000_000
    N = 1_000
    half_N = N//2
    K = 4
    u_bar = 20
    R = 10
    inputs, outputs = get_dataset_v3(N, K, u_bar, R)
    ax  = plt.gca()
    s = 10
    ax.scatter(inputs[:half_N, 0], inputs[:half_N, 1], s=s, color='red', label=f'{half_N} samples from a uniform distribution on the square [-25, 25]')
    ax.scatter(inputs[half_N:, 0], inputs[half_N:, 1], s=s, color='blue', label=f'{half_N} samples from a Gaussian distribution centred on the circle of ray {R}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    lim = 35
    ax.set_ylim([-lim, lim])
    ax.set_xlim([-lim, lim])
    ax.legend(loc='lower left', prop={'size': 8})
    ax.set_title(f'{N} samples for the training set')
    plt.show()
    print(inputs.shape)
    print(outputs.shape)

if __name__ == '__main__':
    main_dataset_visualization()