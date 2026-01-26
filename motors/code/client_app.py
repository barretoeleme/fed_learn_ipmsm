import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

import task

def train():
    train_data, test_data = task.get_data("2D")
    coder_train_dataset, coder_test_dataset = task.coder_dataset(train_data, test_data)
    coder_train_dataloader, coder_test_dataloader = task.coder_dataloader(coder_train_dataset, coder_test_dataset)
    coder = task.train_coder(coder_train_dataloader, coder_test_dataloader)
    model_train_dataset, model_test_dataset = task.encoded_dataset(coder, train_data, test_data)
    model_train_dataloader, model_test_dataloader = task.model_dataloader(model_train_dataset, model_test_dataset)
    model = task.model(model_train_dataloader, model_test_dataloader)
