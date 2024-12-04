import torch.nn as nn
from typing import Callable
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, *layers, activations: dict[int, Callable] | None = None):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()

        activations = activations if activations else {}

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                act_func = activations.get(i, nn.ReLU())
                self.layers.append(act_func)
        
        self.network = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.network(x)
    
    def train_model(self, x_train, y_train, epochs, learning_rate):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()  # Choose appropriate loss function
        
        for epoch in range(epochs):
            self.train()  # Set model to training mode
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self(x_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")