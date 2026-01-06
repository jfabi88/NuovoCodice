import argparse
import yaml
import torch
from pathlib import Path
import shutil
from hgatr.train.train import train
from datetime import datetime

from hgatr.ga.primitives import blade_operator

def main():
    for i in range(8):
        parser = argparse.ArgumentParser(description="Train HGATR model")
        parser.add_argument(
            "--dataset", "-d",
            type=str,
            required=True,
            help="Dataset's name"
        )
        parser.add_argument(
            "--config", "-c",
            type=str,
            required=True,
            help="YAML file"
        )
        args = parser.parse_args()

        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        CONFIGS_DIR = PROJECT_ROOT / "configs"
        RUNS_DIR = PROJECT_ROOT / "runs"

        dataset_name = args.dataset
        print(f"Dataset: {dataset_name}")
        config_path = CONFIGS_DIR / args.config

        if not config_path.exists():
            raise FileNotFoundError(f"File not found: {config_path}")

        with open(config_path, "r") as f:
            parameters = yaml.safe_load(f)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"test-{dataset_name}-{run_id}"

        run_dir = RUNS_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"

        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(config_path, run_dir / "parameters.yaml")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blade = blade_operator().to(device)  # sostituisci con il tuo blade

        train(
            parameters,
            blade,
            dataset_name,
            device,
            run_dir,
        )

if __name__ == "__main__":
    main()
