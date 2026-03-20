import os 
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

"""motors: A Flower / PyTorch app."""

import torch
import pandas as pd
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src.task import RegressionModel, test, load_centralized_dataset

app = ServerApp()

server_losses = []

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    global_model = RegressionModel()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    rounds = sorted(result.train_metrics_clientapp.keys())

    train_losses = []
    eval_losses = []

    for r in rounds:
        train_losses.append(float(result.train_metrics_clientapp[r]["train_loss"]))
        eval_losses.append(float(result.evaluate_metrics_clientapp[r]["eval_loss"]))

    server_losses_cut = server_losses[1:]

    df = pd.DataFrame({
        "round": rounds,
        "server_loss": server_losses_cut,
        "train_loss": train_losses,
        "eval_loss": eval_losses,
    })

    df.to_csv("federated_results.csv", index=False)

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    model = RegressionModel()
    model.load_state_dict(arrays.to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss = test(model, test_dataloader, device)

    server_losses.append(test_loss)

    return MetricRecord({"loss": test_loss})