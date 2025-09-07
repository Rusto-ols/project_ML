import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=2)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.sigmoid(out)
        return out


input_size = X_train.shape[1]
hidden_size = 50
output_size = 1
learning_rate = 0.01
num_epochs = 100

model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train.view(-1, 1))
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        predicted = outputs.round()
        correct = (predicted == Y_train.view(-1, 1)).float().sum()
        accuracy = correct / Y_train.size(0)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss : {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%")

model.eval()
with torch.no_grad():
    outputs = model(X_train)
    predicted = outputs.round()
    correct = (predicted == Y_train.view(-1, 1)).float().sum()
    accuracy = correct / Y_train.size(0)
    print(f"Accuracy on training data: {accuracy.item() * 100:.2f}%")

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predicted = outputs.round()
    correct = (predicted == Y_test.view(-1, 1)).float().sum()
    accuracy = correct / Y_test.size(0)
    print(f"Accuracy on test data: {accuracy.item() * 100:.2f}%")
