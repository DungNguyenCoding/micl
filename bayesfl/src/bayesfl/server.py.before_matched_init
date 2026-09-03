"""Flower/Ray simulation assembly."""

from __future__ import annotations

from pathlib import Path

from flwr.app import Context
from flwr.clientapp import ClientApp
from flwr.server import ServerAppComponents, ServerConfig
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation

from bayesfl.client import BayesFLNumPyClient
from bayesfl.config import ExperimentConfig
from bayesfl.data.datasets import load_test_loader
from bayesfl.evaluation import CentralEvaluator
from bayesfl.models.factory import build_model, count_bayesian_random_variables
from bayesfl.posterior.packing import ParameterLayout, initial_fola_state, model_to_ndarrays
from bayesfl.strategies.research_strategy import ResearchStrategy
from bayesfl.runtime_utils import seed_everything


def run_flower_simulation(
    cfg: ExperimentConfig,
    *,
    partition_path: Path,
    partition_metadata: dict,
    run_dir: Path,
    logger,
) -> None:
    # Make the initial server state reproducible across runs.
    seed_everything(cfg.runtime.seed)
    initial_model = build_model(cfg)
    layout = ParameterLayout.from_model(initial_model)
    if cfg.method == "fola":
        initial_arrays = initial_fola_state(initial_model, cfg.fola.initial_precision)
    else:
        initial_arrays = model_to_ndarrays(initial_model)

    if cfg.method == "bbb":
        d = count_bayesian_random_variables(initial_model)
        logger.info(
            "BBB Bayesian dimension=%d resolved_kl_weight=%.12g",
            d,
            cfg.resolved_kl_weight(d),
        )
        if cfg.data.dataset == "cifar10" and d != 851_514:
            raise RuntimeError(f"CIFAR-10 Bayesian dimension must be 851,514, got {d:,}")

    test_loader = load_test_loader(cfg)
    evaluator = CentralEvaluator(cfg, test_loader, run_dir, logger=logger)
    strategy = ResearchStrategy(
        cfg=cfg,
        layout=layout,
        initial_arrays=initial_arrays,
        run_dir=run_dir,
        logger=logger,
    )
    strategy.evaluate_fn = lambda rnd, arrays, _config: evaluator.evaluate(rnd, arrays)

    average_client_size = float(partition_metadata["mean_size"])

    def client_fn(context: Context):
        client_id = int(context.node_config["partition-id"])
        return BayesFLNumPyClient(
            client_id=client_id,
            cfg=cfg,
            partition_path=partition_path,
            average_client_size=average_client_size,
        ).to_client()

    client_app = ClientApp(client_fn=client_fn)

    def server_fn(context: Context):
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=cfg.training.rounds),
        )

    server_app = ServerApp(server_fn=server_fn)
    backend_config = {
        "client_resources": {
            "num_cpus": float(cfg.runtime.client_num_cpus),
            "num_gpus": float(cfg.runtime.client_num_gpus),
        },
        "init_args": {
            "include_dashboard": False,
            "log_to_driver": True,
        },
    }
    logger.info(
        "Starting Flower/Ray: clients=%d per_round=%d rounds=%d resources/client=(%.2f CPU, %.3f GPU)",
        cfg.federation.num_clients,
        cfg.federation.clients_per_round,
        cfg.training.rounds,
        cfg.runtime.client_num_cpus,
        cfg.runtime.client_num_gpus,
    )
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.federation.num_clients,
        backend_name="ray",
        backend_config=backend_config,
        verbose_logging=cfg.runtime.verbose_flower,
    )
