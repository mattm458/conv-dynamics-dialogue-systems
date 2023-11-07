import os
from os import path

import lightning.pytorch as pl
import pandas as pd
import torch
from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context
from lightning import pytorch as pl
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

from cdmodel.data.dataloader import collate_fn
from cdmodel.data.dataset import ConversationDataset
from cdmodel.model.config import get_model
from cdmodel.util.cv import get_52_cv_ids, get_cv_ids
from cdmodel.util.lightning_logs import get_best_checkpoint, get_highest_version


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
    name = training_config["name"]
    name_idx = f"{name}_{i}"

    if name_idx not in os.listdir(path.join(results_dir, "lightning_logs")):
        return {
            "name": name,
            "fold": i,
            "l1_smooth_loss_autoregress": None,
            "mse_loss_autoregress": None,
        }

    # if name_idx not in os.listdir(results_dir):
    #     return

    latest_version = get_highest_version(
        os.listdir(path.join(results_dir, "lightning_logs", name_idx))
    )

    best_checkpoint_filename = get_best_checkpoint(
        os.listdir(
            path.join(
                results_dir,
                "lightning_logs",
                name_idx,
                f"version_{latest_version}",
                "checkpoints",
            )
        )
    )

    best_checkpoint_path = path.join(
        results_dir,
        "lightning_logs",
        name_idx,
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
        enable_progress_bar=True,
    )

    results = trainer.predict(
        model,
        dataloaders=val_dataloader,
        ckpt_path=best_checkpoint_path,
    )

    y_hats = [x["y_hat"] for x in results]
    ys = [x["y"] for x in results]
    predicts = [x["predict"] for x in results]

    longest_y_hat = max([x.shape[1] for x in y_hats])
    longest_y = max([x.shape[1] for x in ys])
    longest_predict = max([x.shape[1] for x in predicts])

    y_hats_pad = [F.pad(x, (0, 0, 0, longest_y_hat - x.shape[1])) for x in y_hats]
    ys_pad = [F.pad(x, (0, 0, 0, longest_y - x.shape[1])) for x in ys]
    predicts_pad = [F.pad(x, (0, longest_predict - x.shape[1])) for x in predicts]

    y_hats_cat = torch.cat(y_hats_pad, dim=0)
    ys_cat = torch.cat(ys_pad, dim=0)
    predicts_cat = torch.cat(predicts_pad, dim=0)

    output = {
        "name": name,
        "fold": i,
        "l1_smooth_loss_autoregress": F.smooth_l1_loss(
            y_hats_cat[predicts_cat], ys_cat[:, 1:][predicts_cat]
        ).item(),
        "mse_loss_autoregress": F.mse_loss(
            y_hats_cat[predicts_cat], ys_cat[:, 1:][predicts_cat]
        ).item(),
    }

    for j, feature in enumerate(features):
        output[f"{feature}_l1_smooth_loss_autoregress"] = F.smooth_l1_loss(
            y_hats_cat[predicts_cat][:, j], ys_cat[:, 1:][predicts_cat][:, j]
        ).item()

    if "y_hat_tf" in results[0]:
        y_hats_tf = [x["y_hat_tf"] for x in results]
        longest_y_hat_tf = max([x.shape[1] for x in y_hats_tf])
        y_hats_pad_tf = [
            F.pad(x, (0, 0, 0, longest_y_hat_tf - x.shape[1])) for x in y_hats_tf
        ]
        y_hats_cat_tf = torch.cat(y_hats_pad_tf, dim=0)

        output = output | {
            "l1_smooth_loss": F.smooth_l1_loss(
                y_hats_cat_tf[predicts_cat], ys_cat[:, 1:][predicts_cat]
            ).item(),
            "mse_loss": F.mse_loss(
                y_hats_cat_tf[predicts_cat], ys_cat[:, 1:][predicts_cat]
            ).item(),
        }

        for j, feature in enumerate(features):
            output[f"{feature}_l1_smooth_loss"] = F.smooth_l1_loss(
                y_hats_cat_tf[predicts_cat][:, j], ys_cat[:, 1:][predicts_cat][:, j]
            ).item()

    return output


def do_stats(
    dataset_config,
    training_config,
    model_config,
    results_dir,
    device,
    n_jobs,
    embeddings_dir,
    conversation_data_dir,
):
    ids = get_cv_ids(dataset_config)
    cv_ids = get_52_cv_ids(ids)

    pd.DataFrame(
        Parallel(n_jobs=1, backend="loky")(
            delayed(_do_stats)(
                training_config,
                model_config,
                val_ids=ids[val_idx],
                features=dataset_config["features"],
                results_dir=results_dir,
                device=0,
                i=i,
                embeddings_dir=embeddings_dir,
                conversation_data_dir=conversation_data_dir,
            )
            for i, (_, val_idx) in enumerate(cv_ids)
        )
    ).set_index("name").to_csv(f"{training_config['name']}.csv")
