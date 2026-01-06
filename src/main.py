import argparse
import yaml
from pathlib import Path
import shutil
from hgatr.train.train import train
from datetime import datetime

from hgatr.ga.primitives import blade_operator

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
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    CONFIGS_DIR = PROJECT_ROOT / "configs"

    dataset_name = args.dataset
    config_path = CONFIGS_DIR / args.config

    if not config_path.exists():
        raise FileNotFoundError(f"File not found: {config_path}")

    with open(config_path, "r") as f:
        parameters = yaml.safe_load(f)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pre_name = Path(dataset_name).with_stem
    name = f"test-{pre_name}-{run_id}"

    PROJECT_ROOT = Path(__file__).resolve().parents[0]
    BASE_DIR = PROJECT_ROOT / "runs"
    run_dir = BASE_DIR / name
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, run_dir / "parameters.yaml")

    device = "cpu"  # o "cpu"
    blade = blade_operator().to(device)  # sostituisci con il tuo blade

    train(
        parameters,
        blade,
        dataset_name,
        device,
    )

if __name__ == "__main__":
    main()
