"""
fed_learn_ipmsm: Flower ServerApp
"""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from src import task

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    num_rounds: int = context.run_config["num-server-rounds"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    lr: float = context.run_config["learning-rate"]

    print("Starting Flower server")
    print(f"Rounds: {num_rounds}, LR: {lr}")

    input_dim = 20
    output_dim = 2

    global_model = task.RegressionModel(
        input_dim=input_dim,
        output_dim=output_dim,
        neurons=1,
        layers=1,
    )

    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    print("\nSaving final global model...")
    final_state_dict = result.arrays.to_torch_state_dict()
    torch.save(final_state_dict, "final_regression_model.pt")
