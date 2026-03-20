
"""motors: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.task import RegressionModel, load_data
from src.task import test as test_fn
from src.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    model = RegressionModel()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    train_loader, _ = load_data(partition_id, num_partitions, batch_size)

    train_loss = train_fn(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=context.run_config["local-epochs"],
        lr=msg.content["config"]["lr"],
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(train_loader.dataset),
    }
    metric_record = MetricRecord(metrics)

    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    model = RegressionModel()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    _, test_loader = load_data(partition_id, num_partitions, batch_size)

    eval_loss = test_fn(model, test_loader, device)

    metrics = {"eval_loss": eval_loss, "num-examples": len(test_loader.dataset)}
    metric_record = MetricRecord(metrics)

    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)