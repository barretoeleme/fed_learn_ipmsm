import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

import task

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    print("TRAIN STARTED")
    
    train_data, test_data = task.get_data("2D")
    
    coder_train_dataset, coder_test_dataset = task.coder_dataset(train_data, test_data)
    coder_train_dataloader, coder_test_dataloader = task.coder_dataloader(coder_train_dataset, coder_test_dataset)
    coder = task.train_coder(coder_train_dataloader, coder_test_dataloader)
    
    model_train_dataset, model_test_dataset = task.encoded_dataset(coder, train_data, test_data)
    model_train_dataloader, model_test_dataloader = task.model_dataloader(model_train_dataset, model_test_dataset)
    model = task.model(model_train_dataloader, model_test_dataloader)

    model_record = ArrayRecord(model.state_dict())
    coder_record = ArrayRecord(coder.state_dict())

    content = RecordDict({"model": model_record, "coder": coder_record,})
    
    return Message(content=content, reply_to=msg)

@app.evaluate
def evaluate(msg: Message, context: Context):
    y_pred_list = []
    y_test_list = []

    train_data, test_data = task.get_data("2D")
    
    model_train_dataset, model_test_dataset = task.encoded_dataset(coder, train_data, test_data)
    model_train_dataloader, model_test_dataloader = task.model_dataloader(model_train_dataset, model_test_dataset)
    model = task.model(model_train_dataloader, model_test_dataloader)

    model.eval()

    with torch.no_grad():
        for X, y in model_test_dataloader:
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

    metrics = {
        "hys_score" : hys_score,
        "hys_mse" : hys_mse,
        "hys_mape" : hys_mape,
        "jou_score" : jou_score,
        "jou_mse" : jou_mse,
        "jou_mape" : jou_mape,
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics" : metric_record})
    return Message(content=content, reply_to=msg)