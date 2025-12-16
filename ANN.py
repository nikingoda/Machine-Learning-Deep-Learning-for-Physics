import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd

DATA_PATH = "data/ground_truth"
BATCH_SIZE = 32
NUM_EPOCHS = 500

class GetData():
    def __init__(self, data_frame):
        features_np = data_frame.iloc[:, :-1].values.astype(np.float32)
        targets_np = data_frame.iloc[:, -1].values.astype(np.float32)
        
        self.features = torch.tensor(features_np)
        self.targets = torch.tensor(targets_np).reshape(-1, 1)

class SimpleCNN(nn.Module):
    def __init__(self, in_features = 2):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(in_features= in_features, out_features= 400)
        self.fc2 = nn.Linear(in_features= 400, out_features= 400)
        self.fc3 = nn.Linear(in_features= 400, out_features= 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

data_frame = pd.read_csv(DATA_PATH)
num_features = data_frame.shape[1] - 1
dataset = GetData(data_frame= data_frame)
data_loader = DataLoader(
    dataset= dataset,
    batch_size= BATCH_SIZE,
    shuffle=True
)

model = SimpleCNN(in_features= num_features)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device= device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters, lr=0.001)


for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for input, target in data_loader:
        optimizer.zero_grad()
        prediction = model(input)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 20 == 0 or epoch == 0:
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = correct / total
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

print("Training Completed")
torch.save(model.state_dict(), 'trained_ann.pth')