"""Real local or Flower/Ray execution using baseline trainers and evaluator."""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from bayesfl.communication import atomic_json
from bayesfl.logging_utils import create_run_paths, setup_logging, save_resolved_config, save_environment
from bayesfl.data.datasets import prepare_partition, load_test_loader
from bayesfl.data.partition import load_partition, _hash_partitions
from bayesfl.models.factory import initialize_model
from bayesfl.posterior.packing import ParameterLayout, initial_fola_state, model_to_ndarrays
from bayesfl.runtime_utils import seed_everything
from bayesfl.evaluation import CentralEvaluator
from bayesfl.wireless.channel import Channel
from bayesfl.wireless.controller import Controller
from bayesfl.wireless.state import WirelessState, run_lock
from bayesfl.wireless.transport import LocalTransport, FlowerTransport, flower_client


def run_experiment(cfg, *, config_path: Path, resume: Path | None = None):
    cfg.validate()
    seed_everything(cfg.runtime.seed)
    partition_path, metadata = prepare_partition(cfg)
    parts = load_partition(partition_path)
    if _hash_partitions(parts) != metadata["sha256"]:
        raise ValueError("Partition data does not match its saved content hash")
    samples = [int(len(p)) for p in parts]
    if any(n <= 0 for n in samples):
        raise ValueError("Wireless all-client execution requires nonempty clients")
    model = initialize_model(cfg)
    if list(model.named_buffers()):
        raise ValueError("Wireless v2 requires an explicit buffer policy for buffered models")
    ptrs = [p.untyped_storage().data_ptr() for p in model.parameters()]
    if len(set(ptrs)) != len(ptrs):
        raise ValueError("Aliased model parameters need a separate wire-layout adapter")
    layout = ParameterLayout.from_model(model)
    initial = initial_fola_state(model, cfg.fola.initial_precision) if cfg.method == "fola" else model_to_ndarrays(model)
    del model
    if resume is None:
        run_dir, log_path = create_run_paths(cfg)
    else:
        run_dir = resume.expanduser().resolve()
        if not (run_dir / "wireless_resume/latest.json").is_file():
            raise ValueError("Wireless resume requires a wireless-v2 checkpoint, not a legacy run")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = Path(cfg.output.logs_dir).resolve() / f"{run_dir.name}_wireless_resume_{stamp}.log"
        log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = setup_logging(log_path)
    with run_lock(run_dir):
        trace_path = None
        if cfg.wireless.channel.mode == "trace":
            trace_path = run_dir / "wireless/channel_trace.csv"
            if resume is None:
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(cfg.wireless.channel.trace_csv).expanduser(), trace_path)
            elif not trace_path.is_file():
                raise ValueError("Missing pinned channel trace in resumed run")
        channel = Channel(cfg.wireless, cfg.federation.num_clients, trace_path=trace_path)
        state = WirelessState(cfg, layout, initial, run_dir, partition_sha256=metadata["sha256"],
                              trace_sha256=channel.trace_sha256, resume=resume is not None)
        if resume is not None:
            previous = json.loads((run_dir / "partition_metadata.json").read_text())
            if previous["sha256"] != metadata["sha256"]:
                raise ValueError("Resumed partition metadata changed")
            shutil.copy2(run_dir / "resolved_config.yaml", run_dir / f"resolved_config.before_wireless_resume_{stamp}.yaml")
        else:
            shutil.copy2(config_path, run_dir / "source_config.yaml")
            save_environment(run_dir)
            atomic_json(run_dir / "partition_metadata.json", metadata)
        save_resolved_config(cfg, run_dir)
        logger.info("Wireless output: %s | backend=%s | modeled fixed slot %.6fs", run_dir, cfg.wireless.backend, cfg.wireless.round_target_s)
        evaluator = CentralEvaluator(cfg, load_test_loader(cfg), run_dir, logger=logger, state=state)
        if cfg.wireless.backend == "local":
            transport = LocalTransport(cfg, partition_path, metadata["mean_size"])
            Controller(cfg, state, channel, transport, samples, logger).run(evaluator.evaluate)
        else:
            run_flower(cfg, state, channel, evaluator, samples, partition_path, metadata, logger)
        logger.info("Wireless simulation completed: %s; rounds=%d reason=%s", run_dir, state.round_id, state.stop_reason())
    return run_dir


def run_flower(cfg, state, channel, evaluator, samples, partition_path, metadata, logger, *, trainer=None):
    from flwr.app import Context
    from flwr.clientapp import ClientApp
    from flwr.server import Server, ServerAppComponents, ServerConfig
    from flwr.server.client_manager import SimpleClientManager
    from flwr.server.history import History
    from flwr.serverapp import ServerApp
    from flwr.simulation import run_simulation
    from flwr.common import ndarrays_to_parameters

    class WirelessServer(Server):
        def fit(self, num_rounds, timeout):
            started = time.perf_counter()
            history = History()
            transport = FlowerTransport(self._client_manager, cfg)
            def evaluate(r, arrays):
                loss, metrics = evaluator.evaluate(r, arrays)
                history.add_loss_centralized(server_round=r, loss=loss)
                history.add_metrics_centralized(server_round=r, metrics=metrics)
                return loss, metrics
            Controller(cfg, state, channel, transport, samples, logger).run(evaluate)
            self.parameters = ndarrays_to_parameters(state.current)
            return history, time.perf_counter()-started

    def client_fn(context: Context):
        return flower_client(cfg, partition_path, metadata["mean_size"], context, trainer=trainer)

    def server_fn(context: Context):
        return ServerAppComponents(server=WirelessServer(client_manager=SimpleClientManager()),
                                   config=ServerConfig(num_rounds=cfg.training.rounds))

    run_simulation(server_app=ServerApp(server_fn=server_fn), client_app=ClientApp(client_fn=client_fn),
        num_supernodes=cfg.federation.num_clients, backend_name="ray",
        backend_config={"client_resources": {"num_cpus":float(cfg.runtime.client_num_cpus),
                                            "num_gpus":float(cfg.runtime.client_num_gpus)},
                        "init_args": {"include_dashboard":False, "log_to_driver":True}},
        verbose_logging=cfg.runtime.verbose_flower)
