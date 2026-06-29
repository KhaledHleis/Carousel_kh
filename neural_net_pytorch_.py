import csv

import torch

from torch import nn


class NeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(4, 8, bias=False),

            nn.ReLU(),

            nn.Linear(8, 8, bias=False),

            nn.ReLU(),

            nn.Linear(8, 2, bias=False),

        )

    def forward(self, x):

        '''Forward pass'''

        if x.shape[-1] != 4:
            raise ValueError("Input must have 4 features")

        return self.layers(x)

    def train_model(self, X, y, cost_fn, epochs=100, lr=1e-3, verbose=True):

        '''

        Train the network with a custom cost function.



        Args:

            X:        input tensor, shape (N, 4) (N = number of samples, [pos x, pos y, sin theta, cos theta])

            y:        target tensor, shape (N, 2) (N = number of samples, [u1,u2])

            cost_fn:  callable (predictions, targets) -> scalar loss tensor

            epochs:   number of passes over the data

            lr:       learning rate

            verbose:   if True, print loss every 10 epochs

        Returns:

            list of per-epoch loss values

        '''

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()  # nn.Module's built-in mode switch (enables dropout/BN if present)

        history = []

        for epoch in range(epochs):

            optimizer.zero_grad()

            preds = self(X)

            loss = cost_fn(preds, y)

            loss.backward()

            optimizer.step()

            history.append(loss.item())

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"epoch {epoch:4d} | loss {loss.item():.6f}")

        return history

    def cost_function(self, predictions, targets):

        """cost function for training"""

        pass

    def register_to_csv(self, filename):

        """

        Export the weigts of a PyTorch model (MLP with 4 -> 8 -> 8 -> 2 neurons) to a CSV file for the class NeuralNetwork

        """

        # flatten the weights of the model

        all_weights = []

        # loop on the layers

        for layer in self.modules():

            if isinstance(layer, nn.Linear):
                # PyTorch store the weights like (out_features, in_features)

                # NeuralNetwork would like (in_features, out_features)

                # thus, we need to transpose

                weights_flattened = layer.weight.data.t().reshape(-1).tolist()

                all_weights.extend(weights_flattened)

        # write the csv file

        with open(filename, 'w', newline='') as f:

            writer = csv.writer(f)

            writer.writerow(all_weights)

        return all_weights


if __name__ == "__main__":
    model = NeuralNetwork()

    print(model.register_to_csv("model_weights.csv"))

    # model.train_model(X, y, cost_fn=model.cost_function, epochs=100, lr=1e-3)