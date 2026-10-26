# Algorithm and implementation contract — current interpretation

Scope: FedAvg, full-posterior FOLA (figure label **Full-posterior BayesFL**), and post-local-training sparse FOLA (figure label **FedKLS** for KL(global || local)). BBB and `online_recurrence` are retained alternatives, not current paper experiments. This is a description of the documented implementation, not independent certification of exact Paper-00 or Paper-02 reproduction.

## FedAvg

Clients start from the broadcast deterministic network, minimize mean cross-entropy with local SGD, then the server averages parameters with `pi_k=n_k/sum_j n_j`. n_k is assigned data count, not E*n_k. Optional clipping is part of the experimental configuration. Stage-16 is the later strict/unclipped control; Stage-23 is the selected **clipped** MNIST comparison. Do not call every run strict Paper-00 just because the method field is `fedavg`.

## FOLA `paper_reference`

An ordinary deterministic network stores the posterior mean; raw float32 omega/precision arrays accompany its named parameters. The inspected code and source handoff describe the following round-t update (t starts at 1):

```text
mu_local starts from mu_global
omega_local starts from omega_global
For each minibatch b across all E epochs:
    task_loss = mean cross_entropy
    g_task = gradient(task_loss)
    omega_local += (actual_batch_size/client_size) * g_task**2
    prior_loss = (0.5/t) * sum(omega_global*(theta-mu_global)**2)
    objective = task_loss + prior_lambda*prior_loss
    backprop objective; optionally clip objective gradient; SGD step
```

The task gradient-square accumulator is computed before gradient clipping. It is **not** an explicitly computed per-example Fisher or Hessian. Do not replace the accumulator with the separate `online_recurrence` formula or apply that branch's variance-floor semantics to these runs.

For the documented paper-reference reducer:

```text
omega_new = sum_k pi_k*omega_effective_k
mu_new = sum_k pi_k*omega_effective_k*mu_effective_k / (omega_new+epsilon)
```

The epsilon is operational, not merely a plotting safeguard. Float64 accumulation and float32 transport are documented. Initialization and precision settings belong to the selected YAML, not a universal paper default. New source modifications must first inspect the current code and regression tests.

## FedKLS / sparse random protocol

A client snapshots the incoming global posterior, runs the complete local fit, and selects exactly `ceil(r*d)` coordinates over the whole model. Selection does not prune/freezes weights, train a subnet, add dropout rescaling, or apply error feedback.

For valid positive Gaussian variances:

```text
KL(global || local) = .5*[log(v_local/v_global)
                         + (v_global+(mu_local-mu_global)**2)/v_local - 1]
```

Equivalently, with positive precisions tau:

```text
.5*[log(tau_global/tau_local) + tau_local/tau_global
    + tau_local*(mu_global-mu_local)**2 - 1]
```

The global distribution is the **pre-local-training round broadcast**; the local distribution is the **post-fit posterior**. `kl_local_global` reverses the entire KL argument order and is a historical ablation, not the current proposed selector. Ties in the documented selector use lower global coordinate indices.

Random selection is uniform without replacement at the same exact coordinate count, with independent seed/client/true-round mask randomness; it pays the same bitmap bytes. A seed-derived random mask does not remove bitmap cost in this protocol.

Each sparse upload contains three arrays:

```text
uint8[ceil(d/8)] bitmap, little bit order
float32[m] selected means, ascending global indices
float32[m] selected raw precisions, same order
```

Scores are not uploaded. The server validates the packet and reconstructs omitted coordinates from the exact broadcast. It must not read hidden full local posteriors. Sample weights are normalized once across accepted clients, not once per coordinate. After the common dense reducer, coordinates omitted by **every** participating client are restored exactly to broadcast mean and precision; touched coordinates retain baseline epsilon behavior. The documented failure policy rejects incomplete/invalid rounds rather than silently changing the client set.

## Zero precision and score-only floors

CIFAR's documented initialization is raw omega=0. A zero-precision state is not a proper finite-variance Gaussian. `floor_for_score` uses a positive surrogate only to calculate KL scores; it must not silently floor the trained, transmitted, reconstructed or aggregated raw omega. `strict` rejects nonpositive scoring precision. Read and preserve the saved `precision_policy` and `score_precision_floor`; do not infer that floor policies calibrate uncertainty.

## Boundaries

The current study does not implement a physical wireless channel, analog AirComp, adaptive client keep ratios, sparse downlink, secure aggregation, privacy noise, quantization or permanent model pruning. Future wireless development is a proposed next research direction, not an existing feature. Reverse KL, BBB and alternate FOLA modes remain useful research controls but must not be relabeled as the final experiment.

Current byte model: [COMMUNICATION_ACCOUNTING.md](COMMUNICATION_ACCOUNTING.md). Original explanations, including obsolete formulas, are preserved byte-for-byte in [history](history/README.md).
