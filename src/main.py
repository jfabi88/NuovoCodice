import argparse
import yaml
import torch
from pathlib import Path
import shutil
from hgatr.train.train import train
from datetime import datetime
import os
from hgatr.ga.primitives import blade_operator

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def main():
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
    parser.add_argument(
        "--gpu", "-g",
        type=str,
        required=True,
        help="Index of GPU to use"
    )
    parser.add_argument(
        "--loop", "-l",
        type=str,
        required=True,
        help="Number of training"
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        required=True,
        help="Directory to save runs"
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    CONFIGS_DIR = PROJECT_ROOT / "configs"
    RUNS_DIR = PROJECT_ROOT / "runs" / args.save

    dataset_name = args.dataset
    print(f"Dataset: {dataset_name}")
    config_path = CONFIGS_DIR / args.config

    if not config_path.exists():
        raise FileNotFoundError(f"File not found: {config_path}")

    with open(config_path, "r") as f:
        parameters = yaml.safe_load(f)

    for i in range(int(args.loop)):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"test-{dataset_name}-{run_id}"

        run_dir = RUNS_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"

        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(config_path, run_dir / "parameters.yaml")

        device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
        #device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
        blade = blade_operator().to(device)  # sostituisci con il tuo blade
        
        train(
            parameters,
            blade,
            dataset_name,
            device,
            int(args.gpu),
            run_dir,
        )

if __name__ == "__main__":
    main()
