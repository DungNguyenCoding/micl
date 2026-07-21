#!/usr/bin/env bash
set -euo pipefail

RUN="${RUN:-outputs/aircomp_bayesfl_mnist_paper_default_seed42}"
OUT="${OUT:-plots/aircomp_bayesfl_mnist_paper_default_seed42}"
python aircomp_plots.py --run "${RUN}" --output_dir "${OUT}"
