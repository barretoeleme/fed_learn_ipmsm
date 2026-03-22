import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split  # ✅ ADICIONADO

MOTORS = ["2D", "Nabla", "V"]

class RegressionModel(nn.Module):
    
    def __init__(self, input_dim = 20, output_dim = 2, neurons = 160, layers = 4):
        super().__init__()

        modules = []
        
        modules.append(nn.Linear(input_dim, neurons))
        modules.append(nn.ReLU())
        for i in range(layers):
            modules.append(nn.Linear(neurons, neurons))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(neurons, output_dim))
        
        self.linear = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.linear(x)

class MotorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def _motor_from_partition(partition_id: int) -> str:
    return MOTORS[partition_id % len(MOTORS)]

def load_data(partition_id: int, num_partitions: int, batch_size = 128):
    motor = _motor_from_partition(partition_id)

    PATH = str(Path(__file__).resolve().parent.parent.parent / "encoded_dataset" / motor) + "/" 
    
    train_data = pd.read_csv(f"{PATH}encoded_train.csv") 
    test_data = pd.read_csv(f"{PATH}encoded_test.csv")

    target = ['hysteresis', 'joule']

    # ✅ SPLIT 50/50 → eval (client)
    eval_data, _ = train_test_split(test_data, test_size=0.5, random_state=42)

    train_dataset = MotorDataset(train_data.drop(columns = target), train_data[target])
    test_dataset = MotorDataset(eval_data.drop(columns = target), eval_data[target])

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, test_loader

def load_centralized_dataset(batch_size=128):
    all_data = []

    base_path = Path(__file__).resolve().parent.parent.parent / "encoded_dataset"

    for motor in MOTORS:
        path = base_path / motor

        test_data = pd.read_csv(path / "encoded_test.csv")

        # ✅ SPLIT 50/50 → test global (server)
        _, test_split = train_test_split(test_data, test_size=0.5, random_state=42)

        all_data.append(test_split)

    global_test = pd.concat(all_data, axis=0).reset_index(drop=True)

    target = ['hysteresis', 'joule']

    dataset = MotorDataset(global_test.drop(columns=target), global_test[target])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader

def train(model, train_loader, device, epochs = 100, lr = 0.001):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    running_loss = 0.0

    for i in range(epochs):
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred_train = model(X)
            loss = loss_func(pred_train, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    avg_trainloss = running_loss / (epochs * len(train_loader))
    return avg_trainloss


def test(model, test_loader, device):
    model.to(device)
    loss_func = nn.MSELoss()
    model.eval()

    loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss += loss_func(pred, y).item()

    loss = loss / len(test_loader)
    return loss