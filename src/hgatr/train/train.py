from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
import torch

from hgatr.model.net import HGatr
from hgatr.train.lighting import HGATr_LIGHT
from hgatr.dataset.dataset import create_datasets
from hgatr.dataset.loader import load_dataset

from datetime import datetime
from pathlib import Path


def collate_fn(batch):
    windows_batch, labels = zip(*batch)

    w1 = [w[0] for w in windows_batch]
    w2 = [w[1] for w in windows_batch]
    w3 = [w[2] for w in windows_batch]

    labels_tensor = torch.stack(labels)

    return [w1, w2, w3], labels_tensor



def train(parameters, blade, dataset_name, device, run_dir):

    data, gt, info = load_dataset(dataset_name)

    train_dataset, test_dataset, val_dataset = create_datasets(
        data,
        parameters["h_ch"],
        parameters["ver_ch"],
        parameters["vol_ch"],
        parameters["split"],
        torch.from_numpy(gt),
        parameters["window_size"]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=parameters["batch_size"],
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, persistent_workers=False, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, persistent_workers=False, collate_fn=collate_fn)

    hgatr = HGatr(
        in_channels=parameters["in_channels"],
        out_channels=parameters["out_channels"],
        blade=blade,
        hidden_dim=parameters["hidden_dim"],
        n_heads=parameters["n_heads"],
        crop_size=parameters["window_size"] // 4,
        positional_dim=parameters["positional_dim"],
        mv_in_channels=parameters["mv_in_channels"],
        blocks=parameters["blocks"],
        dropout_gatr=parameters["dropout_gatr"],
        dropout_final=parameters["dropout_final"],
        window_size=parameters["window_size"],
        device=device,
    ).to(device)

    hgatr_light = HGATr_LIGHT(
        model=hgatr,
        n_classes=info["n_classes"],
        learning_rate=parameters["lr"],
        data_class_names=info["data_class_names"],
        weights=None,
    )

    checkpoint_loss = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="hci",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    csv_logger = CSVLogger(save_dir=str(run_dir / "logs"), name="")

    trainer_hgatr = Trainer(
        logger=csv_logger,
        devices=1,
        max_epochs=parameters["max_epochs"],
        callbacks=[checkpoint_loss],
        enable_progress_bar=False,
    )
    
    trainer_hgatr.fit(hgatr_light, train_dataloader, val_dataloader)

    hgatr_l = HGatr(
        in_channels=parameters["in_channels"],
        out_channels=parameters["out_channels"],
        blade=blade,
        hidden_dim=parameters["hidden_dim"],
        n_heads=parameters["n_heads"],
        crop_size=parameters["window_size"] // 4,
        positional_dim=parameters["positional_dim"],
        mv_in_channels=parameters["mv_in_channels"],
        blocks=parameters["blocks"],
        dropout_gatr=parameters["dropout_gatr"],
        dropout_final=parameters["dropout_final"],
        window_size=parameters["window_size"],
        device=device,
    )

    hgatr_l_light = HGATr_LIGHT.load_from_checkpoint(
        checkpoint_path=str(run_dir / "checkpoints" / "hci.ckpt"),
        model=hgatr_l,
        n_classes=info["n_classes"],
        learning_rate=parameters["lr"],
        data_class_names=info["data_class_names"],
        weights=None,
    )

    trainer_hgatr.test(hgatr_l_light, dataloaders=test_dataloader, verbose=False)
    hgatr_l_light.save_test_results(run_dir)