"""
fed_learn_ipmsm: Flower ServerApp
"""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

import task  # seu task.py

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # -------- run config --------
    num_rounds: int = context.run_config["num-server-rounds"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    lr: float = context.run_config["learning-rate"]

    print("Starting Flower server")
    print(f"Rounds: {num_rounds}, LR: {lr}")

    # -------- build global model --------
    # input_dim precisa bater com o encoder output
    input_dim = 20      # latent_dim do seu autoencoder
    output_dim = 2      # hysteresis, joule

    global_model = task.RegressionModel(
        input_dim=input_dim,
        output_dim=output_dim,
        neurons=10,
        layers=2,
    )

    # Initialize weights
    arrays = ArrayRecord(
        arrays={
            k: v.detach().cpu().numpy()
            for k, v in global_model.state_dict().items()
        }
    )

    # -------- strategy --------
    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
    )

    # -------- start federated learning --------
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # -------- save final model --------
    print("\nSaving final global model...")
    final_state_dict = result.arrays.to_torch_state_dict()
    torch.save(final_state_dict, "final_regression_model.pt")
