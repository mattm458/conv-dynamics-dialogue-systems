import os
from os import path

import lightning.pytorch as pl
import numpy as np
import torch
from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader

from data.dataloader import collate_fn
from data.dataset import ConversationDataset
from model.config import get_model
from util.cv import get_52_cv_ids, get_cv_ids
from util.lightning_logs import get_best_checkpoint, get_highest_version


def _do_stats(
    training_config,
    model_config,
    val_ids,
    features,
    results_dir,
    device,
    i,
    embeddings_dir,
    conversation_data_dir,
):
    name = f"{training_config['name']}_{i}"

    latest_version = get_highest_version(
        os.listdir(path.join(results_dir, "lightning_logs", name))
    )

    best_checkpoint_filename = get_best_checkpoint(
        os.listdir(
            path.join(
                results_dir,
                "lightning_logs",
                name,
                f"version_{latest_version}",
                "checkpoints",
            )
        )
    )

    best_checkpoint_path = path.join(
        results_dir,
        "lightning_logs",
        name,
        f"version_{latest_version}",
        "checkpoints",
        best_checkpoint_filename,
    )

    torch.set_float32_matmul_precision("high")

    # Load the datasets and dataloaders
    val_dataset = ConversationDataset(
        val_ids,
        embeddings_dir=embeddings_dir,
        conversation_data_dir=conversation_data_dir,
        features=features,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=4,
        multiprocessing_context=get_context("loky"),
    )

    # Create a new instance of the model based on the config
    model = get_model(
        model_config=model_config,
        training_config=training_config,
        feature_names=features,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        precision=16,
        devices=[device],
        auto_lr_find=False,
        logger=None,
        enable_progress_bar=False,  # enable_progress_bar,
        max_epochs=1000,
    )

    results = trainer.predict(
        model,
        dataloaders=val_dataloader,
        ckpt_path=best_checkpoint_path,
    )

    return


def do_stats(
    dataset_config,
    training_config,
    model_config,
    results_dir,
    device,
    method,
    n_jobs,
    embeddings_dir,
    conversation_data_dir,
):
    ids = get_cv_ids(dataset_config)
    cv_ids = get_52_cv_ids(ids)

    for i, (_, val_ids) in enumerate(cv_ids):
        _do_stats(
            training_config,
            model_config,
            val_ids,
            features=dataset_config["features"],
            results_dir=results_dir,
            device=device,
            i=i,
            embeddings_dir=embeddings_dir,
            conversation_data_dir=conversation_data_dir,
        )
        break

    return
