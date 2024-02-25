import datetime
import os
from os import path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader

from cdmodel.data.dataloader import collate_fn, get_collate_fn
from cdmodel.data.dataset_manifest_1 import ConversationDataset
from cdmodel.data.dataset_legacy import LegacyConversationDataset
from cdmodel.model.config import get_model
from cdmodel.util.cv import get_52_cv_ids, get_cv_ids
from cdmodel.util.gpu_distributed_multiprocessing import run_distributed


def _do_cross_validate_52(
    training_config,
    model_config,
    train_ids,
    val_ids,
    features,
    results_dir,
    device,
    i,
    embeddings_dir,
    conversation_data_dir,
    enable_progress_bar=False,
):
    name = f"{training_config['name']}_{i}"

    # If this specific fold has already completed, we don't have anything to do. We know a
    # fold is complete because when it is finished, we create a file with the fold's name
    # in the results directory. Check for that file below:
    results_dir_contents = os.listdir(results_dir)
    if name in results_dir_contents:
        print(f"Fold {name} already completed, skipping...")
        return

    # By default, we aren't loading an existing checkpoint. However, if a previous training run
    # was interrupted for this fold, we'll use checkpoint_path to specify the checkpoint to resume
    # from.
    checkpoint_path = None
    if "lightning_logs" in results_dir_contents and name in os.listdir(
        path.join(results_dir, "lightning_logs")
    ):
        # List the contents of the Lightning fold directory
        fold_dir = path.join(results_dir, "lightning_logs", name)

        # Contents of the directory have the name "version_<num>". Find the largest
        # version number.
        latest_run = max([int(x.split("_")[1]) for x in os.listdir(fold_dir)])

        # The models are configured to save checkpoints of the 5 best results.
        # Find the latest checkpoint by sorting them by their epoch number.
        checkpoint_dir = path.join(
            results_dir, "lightning_logs", name, f"version_{latest_run}", "checkpoints"
        )
        latest_checkpoint = sorted(
            os.listdir(checkpoint_dir),
            key=lambda x: int(
                x.replace("checkpoint-epoch=", "").split("-")[0]
            ),  # Example filename: checkpoint-epoch=222-validation_loss_l1=0.05359.ckpt
            reverse=True,
        )[0]

        # Assemble the path to the checkpoint and save it for later
        checkpoint_path = path.join(
            results_dir,
            "lightning_logs",
            name,
            f"version_{latest_run}",
            "checkpoints",
            latest_checkpoint,
        )

        print("Resuming training from checkpoint", checkpoint_path)

    torch.set_float32_matmul_precision("high")

    # Set up Lightning callbacks and logger
    model_checkpoint = ModelCheckpoint(
        save_top_k=5,
        monitor="validation_loss",
        mode="min",
        filename="checkpoint-{epoch}-{validation_loss:.5f}",
    )
    early_stopping = EarlyStopping(
        monitor="validation_loss",
        patience=training_config["early_stopping_patience"],
        verbose=False,
        mode="min",
    )
    logger = TensorBoardLogger(path.join(results_dir, "lightning_logs"), name=name)

    # Load the datasets and dataloaders
    train_dataset = LegacyConversationDataset(
        train_ids,
        embeddings_dir=embeddings_dir,
        conversation_data_dir=conversation_data_dir,
        features=features,
        agent_assignment=agent_assignment,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=1,  # 4,
        multiprocessing_context="fork",
    )
    val_dataset = LegacyConversationDataset(
        val_ids,
        embeddings_dir=embeddings_dir,
        conversation_data_dir=conversation_data_dir,
        features=features,
        agent_assignment=agent_assignment,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=1,  # 4,
        multiprocessing_context="fork",
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
        callbacks=[model_checkpoint, early_stopping],
        auto_lr_find=False,
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        max_epochs=1000,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=checkpoint_path,
    )

    with open(path.join(results_dir, name), "w") as outfile:
        outfile.write(str(early_stopping.best_score.item()))


def cross_validate_52(
    dataset_config,
    training_config,
    model_config,
    device,
    n_jobs,
    embeddings_dir,
    conversation_data_dir,
    resume=None,
):
    ids = get_cv_ids(dataset_config)

    if resume is None:
        results_dir = f"results_{training_config['name']} {datetime.datetime.now()}"
        os.mkdir(results_dir)
    else:
        results_dir = resume

        if training_config["name"] not in results_dir:
            raise Exception(
                f"Attempted to resume training on an incorrect configuration! Configuration name: {training_config['name']}, directory name: {resume}"
            )

    cv_ids = get_52_cv_ids(ids)

    args = [
        {
            "training_config": training_config,
            "model_config": model_config,
            "train_ids": ids[train_idx],
            "val_ids": ids[val_idx],
            "features": dataset_config["features"],
            "results_dir": results_dir,
            "i": i,
            "embeddings_dir": embeddings_dir,
            "conversation_data_dir": conversation_data_dir,
            "enable_progress_bar": False,  # n_jobs == 1,
        }
        for i, (train_idx, val_idx) in enumerate(cv_ids)
    ]

    run_distributed(
        _do_cross_validate_52, args=args, devices=[0, 1], processes_per_device=2
    )

    # Parallel(n_jobs=n_jobs, backend="loky")(
    #     delayed(_do_cross_validate_52)(
    #         training_config=training_config,
    #         model_config=model_config,
    #         train_ids=ids[train_idx],
    #         val_ids=ids[val_idx],
    #         features=dataset_config["features"],
    #         results_dir=results_dir,
    #         device=device,
    #         i=i,
    #         embeddings_dir=embeddings_dir,
    #         conversation_data_dir=conversation_data_dir,
    #         enable_progress_bar=n_jobs == 1,
    #     )
    #     for i, (train_idx, val_idx) in enumerate(cv_ids)
    # )


def standard_train(
    dataset_config,
    training_config,
    model_config,
    device,
    embeddings_dir,
    conversation_data_dir,
    resume=None,
):
    if resume is None:
        results_dir = f"results_{training_config['name']} {datetime.datetime.now()}"
        os.mkdir(results_dir)
    else:
        results_dir = resume

        if training_config["name"] not in results_dir:
            raise Exception(
                f"Attempted to resume training on an incorrect configuration! Configuration name: {training_config['name']}, directory name: {resume}"
            )

    i = 0

    if "train" and "val" in dataset_config:
        train_ids = np.array(
            [x for x in open(dataset_config["train"]).read().split("\n") if len(x) > 0]
        )
        val_ids = np.array(
            [x for x in open(dataset_config["val"]).read().split("\n") if len(x) > 0]
        )
    else:
        train_path = path.join(
            dataset_config["dir"],
            f"train-{dataset_config['subset']}.csv",
        )
        train_ids = [x for x in open(train_path).read().split("\n") if len(x) > 0]

        val_path = path.join(
            dataset_config["dir"],
            f"val-{dataset_config['subset']}.csv",
        )
        val_ids = [x for x in open(val_path).read().split("\n") if len(x) > 0]

    dir = dataset_config["dir"]
    subset = dataset_config["subset"]
    features = dataset_config["features"]
    gender = dataset_config["additional_inputs"]["gender"]
    speaker_identity = dataset_config["additional_inputs"]["speaker_identity"]
    speaker_identity_them = dataset_config["speaker_identity_them"]
    speaker_identity_always_us = dataset_config["speaker_identity_always_us"]
    zero_pad = dataset_config["zero_pad"]
    agent_assignment = dataset_config["agent_assignment"]
    spectrogram_agent = dataset_config["spectrogram_agent"]
    spectrogram_partner = dataset_config["spectrogram_partner"]
    speaker_id_encode_override = dataset_config["speaker_id_encode_override"]

    speaker_ids = pd.read_csv(
        path.join(dir, f"speaker-ids-{subset}.csv"), index_col="speaker_id"
    )["idx"].to_dict()

    spectrogram_dir = None
    if "spectrogram_dir" in dataset_config:
        spectrogram_dir = dataset_config["spectrogram_dir"]

    name = f"{training_config['name']}_{i}"

    # If this specific fold has already completed, we don't have anything to do. We know a
    # fold is complete because when it is finished, we create a file with the fold's name
    # in the results directory. Check for that file below:
    results_dir_contents = os.listdir(results_dir)
    if name in results_dir_contents:
        print(f"Run {name} already completed, skipping...")
        return

    # By default, we aren't loading an existing checkpoint. However, if a previous training run
    # was interrupted for this fold, we'll use checkpoint_path to specify the checkpoint to resume
    # from.
    checkpoint_path = None
    if "lightning_logs" in results_dir_contents and name in os.listdir(
        path.join(results_dir, "lightning_logs")
    ):
        # List the contents of the Lightning fold directory
        fold_dir = path.join(results_dir, "lightning_logs", name)

        # Contents of the directory have the name "version_<num>". Find the largest
        # version number.
        latest_run = max([int(x.split("_")[1]) for x in os.listdir(fold_dir)])

        # The models are configured to save checkpoints of the 5 best results.
        # Find the latest checkpoint by sorting them by their epoch number.
        checkpoint_dir = path.join(
            results_dir, "lightning_logs", name, f"version_{latest_run}", "checkpoints"
        )
        latest_checkpoint = sorted(
            os.listdir(checkpoint_dir),
            key=lambda x: int(
                x.replace("checkpoint-epoch=", "").split("-")[0]
            ),  # Example filename: checkpoint-epoch=222-validation_loss_l1=0.05359.ckpt
            reverse=True,
        )[0]

        # Assemble the path to the checkpoint and save it for later
        checkpoint_path = path.join(
            results_dir,
            "lightning_logs",
            name,
            f"version_{latest_run}",
            "checkpoints",
            latest_checkpoint,
        )

        print("Resuming training from checkpoint", checkpoint_path)

    torch.set_float32_matmul_precision("high")

    # Set up Lightning callbacks and logger
    model_checkpoint = ModelCheckpoint(
        save_top_k=5,
        monitor="validation_loss",
        mode="min",
        filename="checkpoint-{epoch}-{validation_loss:.5f}",
    )
    early_stopping = EarlyStopping(
        monitor="validation_loss",
        patience=training_config["early_stopping_patience"],
        verbose=False,
        mode="min",
    )
    logger = TensorBoardLogger(path.join(results_dir, "lightning_logs"), name=name)

    # Load the datasets and dataloaders
    Dataset = ConversationDataset
    if "version" in dataset_config and dataset_config["version"] == "legacy":
        Dataset = LegacyConversationDataset

    speaker_id_encode_override_ids = None
    if speaker_id_encode_override is not None:
        speaker_id_encode_override_ids = list(
            pd.read_csv(speaker_id_encode_override, header=None)[0]
        )

    train_dataset = Dataset(
        dir=dir,
        ids=train_ids,
        features=features,
        speaker_ids=speaker_ids,
        gender=gender,
        speaker_identity=speaker_identity,
        speaker_identity_them=speaker_identity_them,
        speaker_identity_always_us=speaker_identity_always_us,
        zero_pad=zero_pad,
        agent_assignment=agent_assignment,
        spectrogram_agent=spectrogram_agent,
        spectrogram_partner=spectrogram_partner,
        spectrogram_dir=spectrogram_dir,
        speaker_id_encode_override=speaker_id_encode_override_ids,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        collate_fn=get_collate_fn(
            has_gender=gender,
            has_speaker_identity=speaker_identity,
            has_spectrogram_agent=spectrogram_agent,
            has_spectrogram_partner=spectrogram_partner,
        ),
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=8,
        persistent_workers=True,
    )
    val_dataset = Dataset(
        dir=dir,
        ids=val_ids,
        features=features,
        speaker_ids=speaker_ids,
        gender=gender,
        speaker_identity=speaker_identity,
        speaker_identity_them=speaker_identity_them,
        speaker_identity_always_us=speaker_identity_always_us,
        zero_pad=zero_pad,
        agent_assignment=agent_assignment,
        spectrogram_agent=spectrogram_agent,
        spectrogram_partner=spectrogram_partner,
        spectrogram_dir=spectrogram_dir,
        speaker_id_encode_override=speaker_id_encode_override_ids,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        collate_fn=get_collate_fn(
            has_gender=gender,
            has_speaker_identity=speaker_identity,
            has_spectrogram_agent=spectrogram_agent,
            has_spectrogram_partner=spectrogram_partner,
        ),
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=8,
        persistent_workers=True,
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
        callbacks=[model_checkpoint, early_stopping],
        logger=logger,
        enable_progress_bar=True,
        max_epochs=1000,
        gradient_clip_val=1.0,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=checkpoint_path,
    )
