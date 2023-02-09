import datetime
import os
from os import path

import lightning.pytorch as pl
import numpy as np
import torch
from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader

from data.dataloader import collate_fn
from data.dataset import ConversationDataset
from model.config import get_model


def _do_cross_validate_52(
    training_config,
    model_config,
    train_ids,
    val_ids,
    features,
    results_dir,
    device,
    i,
    enable_progress_bar=False,
):
    torch.set_float32_matmul_precision("high")

    name = f"{training_config['name']}_{i}"

    model_checkpoint = ModelCheckpoint(
        save_top_k=5,
        monitor="validation_loss",
        mode="min",
        filename="checkpoint-{epoch}-{validation_loss_l1:.5f}",
    )

    early_stopping = EarlyStopping(
        monitor="validation_loss",
        patience=training_config["early_stopping_patience"],
        verbose=False,
        mode="min",
    )

    logger = TensorBoardLogger(path.join(results_dir, "lightning_logs"), name=name)

    train_dataset = ConversationDataset(
        train_ids,
        embeddings_dir="/home/mmcneil/datasets/fisher_corpus/fisher-embeddings",
        conversation_data_dir="/home/mmcneil/datasets/fisher_corpus/fisher-ipu-data",
        features=features,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=4,
        multiprocessing_context=get_context("loky"),
    )
    val_dataset = ConversationDataset(
        val_ids,
        embeddings_dir="/home/mmcneil/datasets/fisher_corpus/fisher-embeddings",
        conversation_data_dir="/home/mmcneil/datasets/fisher_corpus/fisher-ipu-data",
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

    model = get_model(
        model_config=model_config,
        training_config=training_config,
        feature_names=features,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        precision=16,
        devices=[device],
        callbacks=[model_checkpoint, early_stopping],
        auto_lr_find=False,
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        max_epochs=1,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    with open(path.join(results_dir, name), "w") as outfile:
        outfile.write(str(early_stopping.best_score.item()))


def cross_validate_52(
    dataset_config,
    training_config,
    model_config,
    device,
    n_jobs,
    results_dir=None,
):
    ids = open(dataset_config["train"]).read().split("\n")
    ids += open(dataset_config["val"]).read().split("\n")
    ids = np.array(ids)

    if results_dir is None:
        results_dir = f"results_{training_config['name']} {datetime.datetime.now()}"

    os.mkdir(results_dir)

    cv_ids = []
    for i in range(5):
        kfold = KFold(n_splits=2, shuffle=True, random_state=i)
        for train_idx, val_idx in kfold.split(ids):
            cv_ids.append((train_idx, val_idx))

    Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_do_cross_validate_52)(
            training_config=training_config,
            model_config=model_config,
            train_ids=ids[train_idx],
            val_ids=ids[val_idx],
            features=dataset_config["features"],
            results_dir=results_dir,
            device=device,
            i=i,
            enable_progress_bar=n_jobs == 1,
        )
        for i, (train_idx, val_idx) in enumerate(cv_ids)
    )
