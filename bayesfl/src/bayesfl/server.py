"""Flower/Ray simulation assembly."""

from __future__ import annotations

from pathlib import Path

from flwr.app import Context
from flwr.clientapp import ClientApp
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation

from bayesfl.client import BayesFLNumPyClient
from bayesfl.budget_server import BudgetServer
from bayesfl.experiment_state import RunState
from bayesfl.logging_utils import save_resolved_config
from bayesfl.config import ExperimentConfig
from bayesfl.data.datasets import load_test_loader
from bayesfl.evaluation import CentralEvaluator
from bayesfl.models.factory import count_bayesian_random_variables, initialize_model
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
    resume: bool = False,
) -> None:
    seed_everything(cfg.runtime.seed)
    initial_model = initialize_model(cfg)
    layout = ParameterLayout.from_model(initial_model)
    if cfg.method == "fola":
        initial_arrays = initial_fola_state(initial_model, cfg.fola.initial_precision)
    else:
        initial_arrays = model_to_ndarrays(initial_model)

    if cfg.method == "bbb":
        d = count_bayesian_random_variables(initial_model)
        logger.info(
            "BBB Bayesian dimension=%d kl_reference_dimension=%s resolved_kl_weight=%.12g",
            d,
            str(cfg.bbb.kl_reference_dimension or d),
            cfg.resolved_kl_weight(d),
        )
        expected = None
        if cfg.data.dataset == "cifar10":
            if cfg.model.name == "resnet56_gn8":
                expected = 851_514
            elif cfg.model.name == "paper_basiccnn":
                expected = 878_538
        if expected is not None and d != expected:
            raise RuntimeError(
                f"{cfg.model.name} Bayesian dimension must be {expected:,}, got {d:,}"
            )

    if cfg.sparse_enabled:
        # The uploaded deterministic architectures have no persistent buffers.
        # Do not silently drop future BatchNorm running statistics or aliased
        # distinct parameter objects from the sparse coordinate manifest.
        if list(initial_model.named_buffers()):
            raise ValueError("Sparse transport requires an explicit buffer policy for this new model")
        storages = [p.untyped_storage().data_ptr() for p in initial_model.parameters()]
        if len(set(storages)) != len(storages):
            raise ValueError("Aliased distinct parameter storage needs a dedicated sparse layout adapter")
    state = RunState(cfg, layout, initial_arrays, run_dir,
                     partition_sha256=partition_metadata["sha256"], resume=resume)
    initial_arrays = state.current
    if resume:
        save_resolved_config(cfg, run_dir)
    test_loader = load_test_loader(cfg)
    evaluator = CentralEvaluator(cfg, test_loader, run_dir, logger=logger, state=state)
    strategy = ResearchStrategy(
        cfg=cfg,
        layout=layout,
        initial_arrays=initial_arrays,
        run_dir=run_dir,
        logger=logger,
        state=state,
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
            server=BudgetServer(client_manager=SimpleClientManager(), strategy=strategy),
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
