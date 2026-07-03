"""
Cross-platform reproduction script for MMF-EMSNet Scenarios 1, 2, and 3.
Runs the training and evaluation using the parameters from best HPO findings.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Configuration variables
PYTHON_EXEC = r"C:\Users\awan2\miniconda3\envs\gizviz\python.exe"
DATASET_PATH = "Dataset/NPZ/dataset_16.npz"
EPOCHS = 100
NUM_SAMPLES_PER_CLASS = 5

SCENARIO_CONFIGS = [
    {
        "name": "Scenario 1 (Binary: 0 vs 4)",
        "scenario": "1",
        "residual": "True",
        "dsm_mode": "dsm_only",
        "optimizer": "RMSprop",
        "lr": "0.001",
        "batch_size": "64",
        "output_dir": "results_reproduced/scenario_1",
    },
    {
        "name": "Scenario 2 (Binary: 0-3 vs 4)",
        "scenario": "2",
        "residual": "False",
        "dsm_mode": "dsm_uncertainty",
        "optimizer": "RMSprop",
        "lr": "0.0001",
        "batch_size": "256",
        "output_dir": "results_reproduced/scenario_2",
    },
    {
        "name": "Scenario 3 (Multiclass: 0-4)",
        "scenario": "3",
        "residual": "True",
        "dsm_mode": "dsm_uncertainty",
        "optimizer": "Nadam",
        "lr": "0.001",
        "batch_size": "128",
        "output_dir": "results_reproduced/scenario_3",
    },
]


def check_python_env():
    """Verify that the configured python executable exists and works."""
    python_path = Path(PYTHON_EXEC)
    if not python_path.exists():
        # Fallback to current sys.executable if custom one is missing
        print(f"Warning: Configured python path '{PYTHON_EXEC}' not found.")
        print(f"Falling back to current interpreter: '{sys.executable}'")
        return sys.executable
    return str(python_path)


def run_command(cmd: list[str], name: str):
    print("\n" + "=" * 80)
    print(f"Starting execution of: {name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80 + "\n")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream stdout/stderr in real time
    if process.stdout:
        for line in process.stdout:
            print(line, end="")
            
    rc = process.wait()
    if rc != 0:
        print(f"\n[ERROR] Command failed with exit code: {rc}")
        sys.exit(rc)
    print(f"\n[SUCCESS] Completed execution of: {name}")


def main():
    python_bin = check_python_env()
    
    print("=" * 80)
    print("MMF-EMSNet Scenario Reproduction Runner")
    print(f"Python interpreter:   {python_bin}")
    print(f"Dataset path:         {DATASET_PATH}")
    print(f"Training Epochs:      {EPOCHS}")
    print(f"Vis samples/class:    {NUM_SAMPLES_PER_CLASS}")
    print("=" * 80)

    for config in SCENARIO_CONFIGS:
        cmd = [
            python_bin,
            "run_custom_train_infer.py",
            "--dataset", DATASET_PATH,
            "--scenario", config["scenario"],
            "--residual", config["residual"],
            "--dsm-mode", config["dsm_mode"],
            "--optimizer", config["optimizer"],
            "--lr", config["lr"],
            "--batch-size", config["batch_size"],
            "--epochs", str(EPOCHS),
            "--num-samples-per-class", str(NUM_SAMPLES_PER_CLASS),
            "--output-dir", config["output_dir"],
        ]
        
        run_command(cmd, config["name"])

    print("\n" + "=" * 80)
    print("Reproduction runs for all scenarios successfully completed!")
    print("Outputs stored under results_reproduced/")
    print("=" * 80)


if __name__ == "__main__":
    main()
