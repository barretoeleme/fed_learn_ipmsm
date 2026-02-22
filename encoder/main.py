import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from flwr.app import ArrayRecord
from pathlib import Path

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim = 20):
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
    PATH = f"../original_dataset/{motor}/"
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

def get_dataset(coder_train_data, coder_test_data):
    target = ['hysteresis', 'joule']

    coder_train_dataset = MotorDataset(coder_train_data.drop(columns = target), coder_train_data[target])
    coder_test_dataset = MotorDataset(coder_test_data.drop(columns = target), coder_test_data[target])

    return coder_train_dataset, coder_test_dataset

def get_dataloader(coder_train_dataset, coder_test_dataset, batch_size = 128):
    coder_train_loader = DataLoader(coder_train_dataset, batch_size = batch_size, shuffle = True)
    coder_test_loader = DataLoader(coder_test_dataset, batch_size = batch_size, shuffle = True)

    return coder_train_loader, coder_test_loader

def train_coder(coder, train_loader, learning_rate = 1e-3, epochs = 100):
    device = torch.device("cpu")
    coder.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(coder.parameters(), lr = learning_rate)

    for epoch in range(epochs):
        print("funcionando")
        coder.train()
        for data, _ in train_loader:
            data = data.to(device)

            optimizer.zero_grad()
            outputs = coder(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
    
    return coder

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

motors = ["2D", "Nabla", "V"]

for motor in motors:
    # data loading
    train_data, test_data = get_data("2D")
    train_dataset, test_dataset = get_dataset(train_data, test_data)
    train_loader, test_loader = get_dataloader(train_dataset, test_dataset)

    # setting input dim
    X_sample, _ = next(iter(train_loader))
    input_dim = X_sample.shape[1]

    # setting up autoencoder
    coder = Autoencoder(input_dim=input_dim)
    coder = train_coder(coder = coder, train_loader = train_loader)
    encoded_train_data, encoded_test_data = encoded_dataset(coder, train_data, test_data)

    X_train_encoded = encoded_train_data.X.numpy()
    y_train_encoded = encoded_train_data.y.numpy()

    X_test_encoded = encoded_test_data.X.numpy()
    y_test_encoded = encoded_test_data.y.numpy()

    latent_dim = X_train_encoded.shape[1]
    latent_columns = [f"latent_{i}" for i in range(latent_dim)]

    df_encoded_train = pd.DataFrame(X_train_encoded, columns=latent_columns)
    df_encoded_train["hysteresis"] = y_train_encoded[:, 0]
    df_encoded_train["joule"] = y_train_encoded[:, 1]

    df_encoded_test = pd.DataFrame(X_test_encoded, columns=latent_columns)
    df_encoded_test["hysteresis"] = y_test_encoded[:, 0]
    df_encoded_test["joule"] = y_test_encoded[:, 1]

    # Salva
    df_encoded_train.to_csv(f"../encoded_dataset/{motor}/encoded_train_2D.csv", index=False)
    df_encoded_test.to_csv(f"../encoded_dataset/{motor}/encoded_test_2D.csv", index=False)

