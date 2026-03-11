import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

#Rede neural simples
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


#Escolhendo a GPU como device, verificando sua disponibilidade
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hiperparametros
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#Carregando os dados de treino e teste do MNIST e separando-o em lotes de 64 imagens
train_data = dset.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = dset.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

#Instânciando o modelo
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#Escolhendo o critério e o otimizador
criterion = nn.CrossEntropyLoss() #Entropia foi escolhida com base em sua compatibiliade para classificação de categorias
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Adam foi escolhido pela taxa de aprendizado adaptativa


def train_network(model, loader, criterion, optimizer, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, targets) in enumerate(loader):
            # Envia para o device (GPU/CPU)
            data, targets = data.to(device), targets.to(device)

            # Flatten: (batch, 1, 28, 28) -> (batch, 784)
            data = data.view(data.shape[0], -1)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient descent step
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}] concluída.")


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    accuracy = float(num_correct) / float(num_samples) * 100
    print(f"Acurácia: {num_correct}/{num_samples} ({accuracy:.2f}%)")
    model.train()


# Execução
train_network(model, train_loader, criterion, optimizer, device)
check_accuracy(test_loader, model, device)

