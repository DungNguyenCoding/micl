"""Local and Flower transports share the exact scalar/packet phase contract."""
from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from bayesfl.wireless.client import ClientEngine, MemoryStore, FlowerStore


@dataclass
class Reply:
    arrays: list[np.ndarray]
    num_examples: int
    metrics: dict


class ExchangeError(RuntimeError):
    def __init__(self, message, replies):
        super().__init__(message)
        self.replies = replies


class LocalTransport:
    """Sequential device emulation. Modeled preparation is still parallel max."""
    def __init__(self, cfg, partition_path, average_client_size, *, trainer=None):
        self.clients = [ClientEngine(cfg, i, partition_path, average_client_size, MemoryStore(), trainer=trainer)
                        for i in range(cfg.federation.num_clients)]

    def exchange(self, parameters, instructions, round_id):
        replies = {}
        try:
            for cid, client in enumerate(self.clients):
                arrays, n, metrics = client.handle(parameters, instructions[cid])
                replies[cid] = Reply(arrays, n, metrics)
        except Exception as exc:
            raise ExchangeError(str(exc), replies) from exc
        return replies


class FlowerTransport:
    def __init__(self, manager, cfg):
        from flwr.common import Code, GetPropertiesIns
        self.cfg = cfg
        k = cfg.federation.num_clients
        if not manager.wait_for(num_clients=k, timeout=int(cfg.wireless.rpc_timeout_s)):
            raise RuntimeError("Timed out waiting for all wireless clients")
        self.clients = {}
        for proxy in manager.all().values():
            response = proxy.get_properties(GetPropertiesIns({}), timeout=cfg.wireless.rpc_timeout_s, group_id=0)
            cid = response.properties.get("client_id")
            if response.status.code != Code.OK or type(cid) is not int or not 0 <= cid < k or cid in self.clients:
                raise ValueError("Invalid/duplicate wireless logical identity")
            self.clients[cid] = proxy
        if set(self.clients) != set(range(k)):
            raise ValueError("Missing client identity")

    def exchange(self, parameters, instructions, round_id):
        from flwr.common import Code, FitIns, ndarrays_to_parameters, parameters_to_ndarrays
        packed = ndarrays_to_parameters(parameters)
        replies, errors = {}, []
        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = {executor.submit(proxy.fit, FitIns(packed, instructions[cid]),
                                       self.cfg.wireless.rpc_timeout_s, round_id): cid
                       for cid, proxy in self.clients.items()}
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    r = future.result()
                    arrays = parameters_to_ndarrays(r.parameters)
                    replies[cid] = Reply(arrays, r.num_examples, dict(r.metrics))
                    if r.status.code != Code.OK:
                        errors.append(f"client {cid}: {r.status.message}")
                except Exception as exc:
                    errors.append(f"client {cid}: {exc}")
        if errors:
            raise ExchangeError("; ".join(errors), replies)
        return replies


def flower_client(cfg, partition_path, average_client_size, context, *, trainer=None):
    from flwr.client import NumPyClient
    cid = int(context.node_config["partition-id"])
    engine = ClientEngine(cfg, cid, partition_path, average_client_size, FlowerStore(context.state), trainer=trainer)

    class WirelessNumPyClient(NumPyClient):
        def get_properties(self, config):
            return {"client_id": cid}

        def get_parameters(self, config):
            raise RuntimeError("Wireless initial model is server-owned; no get_parameters upload")

        def fit(self, parameters, config):
            return engine.handle(parameters, config)

        def evaluate(self, parameters, config):
            return 0.0, 0, {}

    return WirelessNumPyClient().to_client()
