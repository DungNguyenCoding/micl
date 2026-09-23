"""Legacy Flower Server loop with true-round and strict-budget stopping.

Uses the project's existing Server/Strategy/NumPyClient API, not Message API
aggregation. Do not implement budget stopping by running empty fake rounds.
"""
from __future__ import annotations

import time
from flwr.server import Server
from flwr.server.history import History


class BudgetServer(Server):
    def fit(self, num_rounds: int, timeout: float | None):
        state = self.strategy.state
        history = History()
        start = time.perf_counter()
        try:
            self.parameters = self._get_initial_parameters(server_round=state.round_id, timeout=timeout)
            if state.round_id == 0 and not (state.run_dir / "resume" / "latest.json").exists():
                result = self.strategy.evaluate(0, parameters=self.parameters)
                if result is not None:
                    history.add_loss_centralized(server_round=0, loss=result[0])
                    history.add_metrics_centralized(server_round=0, metrics=result[1])
            for server_round in range(state.round_id + 1, num_rounds + 1):
                if state.stop_reason() != "running":
                    break
                result = self.fit_round(server_round=server_round, timeout=timeout)
                if result is None or result[0] is None:
                    raise RuntimeError("A dispatched round produced no aggregated model")
                self.parameters, fit_metrics, _ = result
                history.add_metrics_distributed_fit(server_round=server_round, metrics=fit_metrics)
                result = self.strategy.evaluate(server_round, parameters=self.parameters)
                if result is not None:
                    history.add_loss_centralized(server_round=server_round, loss=result[0])
                    history.add_metrics_centralized(server_round=server_round, metrics=result[1])
                # Evaluation is centralized; fraction_evaluate remains zero.
                self.evaluate_round(server_round=server_round, timeout=timeout)
            state.finalize()
            self.strategy.logger.info("Stopped: reason=%s completed_rounds=%d array_bytes=%d",
                                      state.stop_reason(), state.round_id, state.budget_used)
            return history, time.perf_counter() - start
        except Exception as exc:
            state.finalize(failed=str(exc))
            raise
