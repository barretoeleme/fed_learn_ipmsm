import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import datetime

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
    
def get_data():
    MOTOR = "2D"
    PATH = f"../dataset/{MOTOR}/"
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

# --------------------------------------------------------- #

# ------------------- Autoencoder Functions ------------------- #

def coder_dataset():
    train_data, test_data = get_data()

    target = ['hysteresis', 'joule']

    train_dataset = MotorDataset(train_data.drop(columns = target), train_data[target])
    test_dataset = MotorDataset(test_data.drop(columns = target), test_data[target])

    return train_dataset, test_dataset


def coder_dataloader(batch_size = 128):
    train_dataset, test_dataset = coder_dataset()

    coder_train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    coder_test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return coder_train_loader, coder_test_loader


def coder():
    coder_train_loader, coder_test_loader = coder_dataloader()

    X_sample, _ = next(iter(coder_train_loader))
    input_dim = X_sample.shape[1]
    latent_dim = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder_model = Autoencoder(input_dim, latent_dim)
    autoencoder_model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder_model.parameters(), lr=1e-3)
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

# --------------------------------------------------------- #

# ------------------- Model Functions ------------------- #

def model_dataset():
    train_data, test_data = get_data()
    autoencoder_model = coder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target = ['hysteresis', 'joule']

    encoded_train = autoencoder_model.encoder(
        torch.tensor(train_data.drop(columns=target).values, dtype=torch.float32).to(device)
    ).cpu().detach().numpy()

    encoded_test = autoencoder_model.encoder(
        torch.tensor(test_data.drop(columns=target).values, dtype=torch.float32).to(device)
    ).cpu().detach().numpy()

    model_train_dataset = MotorDataset(pd.DataFrame(encoded_train), train_data[target])
    model_test_dataset = MotorDataset(pd.DataFrame(encoded_test), test_data[target])

    return model_train_dataset, model_test_dataset


def model_dataloader():
    model_train_dataset, model_test_dataset = model_dataset()

    batch_size = 128

    model_train_loader = DataLoader(model_train_dataset, batch_size=batch_size, shuffle=True)
    model_test_loader = DataLoader(model_test_dataset, batch_size=batch_size, shuffle=True)
    
    return model_train_loader, model_test_loader


def model():
    model_train_loader, model_test_loader = model_dataloader()

    input_dim = model_train_loader.shape[1]
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

    y_pred_list = []
    y_test_list = []

    model.eval()

    with torch.no_grad():
        for X, y in model_test_loader:
            pred_test = model(X)
            y_pred_list.append(pred_test)
            y_test_list.append(y)

    y_pred = torch.cat(y_pred_list)
    y_test = torch.cat(y_test_list)

    hys_score = r2_score(y_test[:, 0], y_pred[:, 0])
    hys_mse = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    hys_mape = mean_absolute_percentage_error(y_test[:, 0], y_pred[:, 0])

    jou_score = r2_score(y_test[:, 1], y_pred[:, 1])
    jou_mse = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    jou_mape = mean_absolute_percentage_error(y_test[:, 1], y_pred[:, 1])

    print(f"\tSpecs:")
    print(f"\t\thys_score: {hys_score}, hys_mse: {hys_mse}, hys_mape: {hys_mape}.\n")
    print(f"\t\tjou_score: {jou_score}, jou_mse: {jou_mse}, jou_mape: {jou_mape}.\n\n")