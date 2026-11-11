#!/usr/bin/env python3
"""Terminal-only summary; no training, plots or report files."""
import argparse
from pathlib import Path
from bayesfl.wireless.reporting import inspect_run, print_results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,action='append',required=True)
    p.add_argument('--time-budget-seconds',type=float)
    a=p.parse_args()
    print_results([inspect_run(d,time_budget_s=a.time_budget_seconds) for d in a.run_dir])
if __name__=='__main__':main()
