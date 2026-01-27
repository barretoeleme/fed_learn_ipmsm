import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import datetime
from pathlib import Path

# --------------------------------------------------------- #

class RegressionModel(nn.Module):
    
    def __init__(self, input_dim, output_dim, neurons = 5, layers = 1):
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
        x = self.linear(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU() # The bottleneck layer
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class MotorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
def get_data(motor):
    PATH = str(Path(__file__).resolve().parent.parent / "dataset" / motor) + "/"
    TRAIN_FILE = "_all_scaled_train.csv"
    TEST_FILE = "_all_scaled_test.csv"

    train_data = pd.DataFrame()

    train_data = pd.concat([train_data, pd.read_csv(f'{PATH}idiq{TRAIN_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
    train_data['speed'] = pd.read_csv(f'{PATH}speed{TRAIN_FILE}')['N']
    train_data = pd.concat([train_data, pd.read_csv(f'{PATH}xgeom{TRAIN_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
    train_data['hysteresis'] = pd.read_csv(f'{PATH}hysteresis{TRAIN_FILE}')['total']
    train_data['joule'] = pd.read_csv(f'{PATH}joule{TRAIN_FILE}')['total']

    test_data = pd.DataFrame()

    test_data = pd.concat([test_data, pd.read_csv(f'{PATH}idiq{TEST_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
    test_data['speed'] = pd.read_csv(f'{PATH}speed{TEST_FILE}')['N']
    test_data = pd.concat([test_data, pd.read_csv(f'{PATH}xgeom{TEST_FILE}').drop(columns = "Unnamed: 0")], axis = 1)
    test_data['hysteresis'] = pd.read_csv(f'{PATH}hysteresis{TEST_FILE}')['total']
    test_data['joule'] = pd.read_csv(f'{PATH}joule{TEST_FILE}')['total']

    return train_data, test_data

def to_array_record(state_dict):
    return ArrayRecord(arrays={k: v.detach().cpu().numpy() for k, v in state_dict.items()})

# --------------------------------------------------------- #

# ------------------- Autoencoder Functions ------------------- #

def coder_dataset(coder_train_data, coder_test_data):
    target = ['hysteresis', 'joule']

    coder_train_dataset = MotorDataset(coder_train_data.drop(columns = target), coder_train_data[target])
    coder_test_dataset = MotorDataset(coder_test_data.drop(columns = target), coder_test_data[target])

    return coder_train_dataset, coder_test_dataset


def coder_dataloader(coder_train_dataset, coder_test_dataset, batch_size = 128):
    coder_train_loader = DataLoader(coder_train_dataset, batch_size = batch_size, shuffle = True)
    coder_test_loader = DataLoader(coder_test_dataset, batch_size = batch_size, shuffle = True)

    return coder_train_loader, coder_test_loader


def train_coder(coder_train_loader, coder_test_loader, learning_rate = 1e-3, epochs = 100, latent_dim = 20):
    X_sample, _ = next(iter(coder_train_loader))
    input_dim = X_sample.shape[1]

    device = torch.device("cpu")

    autoencoder_model = Autoencoder(input_dim, latent_dim)
    autoencoder_model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder_model.parameters(), lr = learning_rate)
    epochs = 100

    for epoch in range(epochs):
        autoencoder_model.train()
        for data, _ in coder_train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            outputs = autoencoder_model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
    
    return autoencoder_model

def encoded_dataset(coder, train_data, test_data):
    device = torch.device("cpu")

    target = ['hysteresis', 'joule']

    encoded_train = coder.encoder(
        torch.tensor(train_data.drop(columns=target).values, dtype=torch.float32).to(device)
    ).cpu().detach().numpy()

    encoded_test = coder.encoder(
        torch.tensor(test_data.drop(columns=target).values, dtype=torch.float32).to(device)
    ).cpu().detach().numpy()

    model_train_dataset = MotorDataset(pd.DataFrame(encoded_train), train_data[target])
    model_test_dataset = MotorDataset(pd.DataFrame(encoded_test), test_data[target])

    return model_train_dataset, model_test_dataset

# --------------------------------------------------------- #

# ------------------- Model Functions ------------------- #


def model_dataloader(model_train_dataset, model_test_dataset, batch_size = 128):
    model_train_loader = DataLoader(model_train_dataset, batch_size=batch_size, shuffle=True)
    model_test_loader = DataLoader(model_test_dataset, batch_size=batch_size, shuffle=True)
    
    return model_train_loader, model_test_loader


def model(model_train_loader, model_test_loader):
    X_sample, _ = next(iter(model_train_loader))
    input_dim = X_sample.shape[1]
    output_dim = 2

    model = RegressionModel(input_dim, output_dim, neurons = 10, layers = 2)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

    for a in range(100):
        model.train()
        for X, y in model_train_loader:
            optimizer.zero_grad()
            pred_train = model(X)
            loss = loss_func(pred_train, y)
            loss.backward()
            optimizer.step()

    time = datetime.datetime.now()
    print(f"\tFinished training model at {time}.\n")

    return model