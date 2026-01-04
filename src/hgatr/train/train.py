from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path

from hgatr.model.net import HGatr
from lighting import HGATr_LIGHT
from hgatr.dataset.dataset import create_datasets
from hgatr.dataset.loader import load_dataset

from datetime import datetime
from pathlib import Path

def train(parameters, blade, dataset_name, device):

    data, gt, info = load_dataset(dataset_name)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"test-{info['pre_name']}-{run_id}"

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    BASE_DIR = PROJECT_ROOT / "runs"
    run_dir = BASE_DIR / name
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset, val_dataset = create_datasets(
        data,
        parameters["h_ch"],
        parameters["ver_ch"],
        parameters["vol_ch"],
        parameters["split"],
        gt,
        parameters["window_size"]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=parameters["batch_size"],
        shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    hgatr = HGatr(
        in_channels=parameters["in_channels"],
        out_channels=parameters["out_channels"],
        blade=blade,
        blade_len=blade.shape[0],
        hidden_dim=parameters["hidden_dim"],
        n_heads=parameters["n_heads"],
        n_classes=info["n_classes"],
        crop_size=parameters["window_size"] // 4,
        positional_dim=parameters["positional_dim"],
        mv_in_channels=parameters["mv_in_channels"],
        blocks=parameters["blocks"],
        dropout_gatr=parameters["dropout_gatr"],
        dropout_final=parameters["dropout_final"],
        window_size=parameters["window_size"],
    ).to(device)

    hgatr_light = HGATr_LIGHT(
        model=hgatr,
        n_classes=info["n_classes"],
        learning_rate=parameters["lr"],
        data_class_names=info["data_class_names"],
        weights=None,
    )

    checkpoint_loss = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="hci",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    csv_logger = CSVLogger(save_dir=str(logs_dir), name="")

    trainer_hgatr = Trainer(
        logger=csv_logger,
        max_epochs=parameters["max_epochs"],
        callbacks=[checkpoint_loss]
    )

    trainer_hgatr.fit(hgatr_light, train_dataloader, val_dataloader)

    hgatr_l = HGatr(
        in_channels=parameters["in_channels"],
        out_channels=parameters["out_channels"],
        blade=blade,
        blade_len=blade.shape[0],
        hidden_dim=parameters["hidden_dim"],
        n_heads=parameters["n_heads"],
        n_classes=info["n_classes"],
        crop_size=parameters["window_size"] // 4,
        positional_dim=parameters["positional_dim"],
        mv_in_channels=parameters["mv_in_channels"],
        blocks=parameters["blocks"],
        dropout_gatr=parameters["dropout_gatr"],
        dropout_final=parameters["dropout_final"],
        window_size=parameters["window_size"],
    )

    hgatr_l_light = HGATr_LIGHT.load_from_checkpoint(
        checkpoint_path=checkpoints_dir / "hci.ckpt",
        model=hgatr_l,
        n_classes=info["n_classes"],
        learning_rate=parameters["lr"],
        data_class_names=info["data_class_names"],
        weights=None,
    )

    trainer_hgatr.test(hgatr_l_light, dataloaders=test_dataloader, verbose=False)

    print(hgatr_l_light.test_results)
