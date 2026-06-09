# Fig. 2 simulation: BS coverage distance / active-radius experiment

This addition does **not** change the previous Fig. 1 script (`main.py`).
It adds a separate entry point:

```bash
python main_fig2.py ...
```

## Experiment meaning

For Fig. 2, the code uses the following interpretation requested in the chat:

1. Generate/place `K=300` physical devices once inside the maximum BS coverage disk, e.g. `coverage_m=550`.
2. For each active radius in `fig2_active_radii`, keep only devices whose distance to the BS is inside that radius.
3. Run the proposed BS-assisted OTA-FL method without TCI.

So with:

```yaml
coverage_m: 550
fig2_active_radii: [550, 300, 50]
```

we get:

- `r_cvge = 550m`: all generated devices participate.
- `r_cvge = 300m`: only devices within 300m participate.
- `r_cvge = 50m`: only devices within 50m participate.

The code saves `fig2_active_device_summary.csv`, so you can verify how many devices are active for each radius.

## Run command

```bash
nohup python main_fig2.py \
  num_rounds=1000 \
  fig2_m0=160 \
  fig2_active_radii='[550,300,50]' \
  num_devices=300 \
  num_flower_clients=24 \
  client_cpus=1 \
  client_gpus=0 \
  coverage_m=550 \
  local_epochs=1 \
  batch_size=32 \
  lr=0.02 \
  eval_every=1 \
  split_seed=42 \
  runtime_seed=42 \
  output_dir=outputs/fig2_cov550_active_radii \
  > fig2_cov550_active_radii.log 2>&1 &
```

Monitor:

```bash
tail -f fig2_cov550_active_radii.log
watch -n 60 'date; ls -lh outputs/fig2_cov550_active_radii'
```

Expected outputs:

```text
outputs/fig2_cov550_active_radii/data_split_summary.csv
outputs/fig2_cov550_active_radii/fig2_active_device_summary.csv
outputs/fig2_cov550_active_radii/rcvge_550m_m0_160_history.csv
outputs/fig2_cov550_active_radii/rcvge_300m_m0_160_history.csv
outputs/fig2_cov550_active_radii/rcvge_50m_m0_160_history.csv
outputs/fig2_cov550_active_radii/fig2_coverage_accuracy.png
```

## Note

This script implements only the proposed/Alg.2 curves. It does not add the TCI benchmark curves.
