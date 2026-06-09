
# Device-distance visualization add-on

This version adds a radar-style plot of the generated 1D device distances without
changing the simulation logic.

The simulation still uses only the distance from each edge device to the BS.
The visualization assigns a deterministic random angle to each distance only for
plotting.

## Fig. 1 output

Running `main.py` now additionally saves:

```text
device_distribution_fig1.png
device_distribution_fig1_devices.csv
device_distribution_fig1_active_summary.csv
```

inside the selected `output_dir`.

## Fig. 2 output

Running `main_fig2.py` now additionally saves:

```text
device_distribution_fig2.png
device_distribution_fig2_devices.csv
device_distribution_fig2_active_summary.csv
```

inside the selected `output_dir`.

For Fig. 2, the plot highlights the active radius circles, e.g. 550 m, 300 m,
and 50 m. The plot also draws concentric guide rings every 50 m.

## Device labels

By default, device index labels are only enabled when `num_devices <= 50`,
because labeling all 300 devices makes the figure crowded. To force labels,
change this line in `main.py` or `main_fig2.py`:

```python
label_devices=(sim_cfg.num_devices <= 50)
```

to:

```python
label_devices=True
```
