import os
from os import path

import pandas as pd
import torch
from lightning import pytorch as pl
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from cdmodel.data.dataloader_manifest_1 import collate_fn
from cdmodel.data.dataset_manifest_1 import ConversationDataset
from cdmodel.model.config import get_model
from cdmodel.model.sequential_manifest_v1 import SequentialConversationModel


def __load_set_ids(dataset_dir: str, dataset_subset: str, set: str) -> list[int]:
    with open(path.join(dataset_dir, f"{set}-{dataset_subset}.csv")) as infile:
        return [int(x) for x in infile.readlines() if len(x) > 0]


def do_test(
    dataset_config: dict,
    training_config: dict,
    model_config: dict,
    checkpoint_path: str,
    device,
    dataset_dir: str,
    out_dir: str,
):
    try:
        os.mkdir(out_dir)
    except:
        pass

    features = dataset_config["features"]

    # Load the speaker IDs for the given subset
    speaker_ids = pd.read_csv(
        path.join(dataset_dir, f"speaker-ids-all.csv"),
        index_col="speaker_id",
    )["idx"].to_dict()

    train_ids = __load_set_ids(
        dataset_dir=dataset_dir, dataset_subset="all", set="train"
    )
    val_ids = __load_set_ids(dataset_dir=dataset_dir, dataset_subset="all", set="val")
    test_ids = __load_set_ids(dataset_dir=dataset_dir, dataset_subset="all", set="test")

    all_ids = train_ids + val_ids + test_ids

    torch.set_float32_matmul_precision("high")

    model = get_model(
        dataset_config=dataset_config,
        model_config=model_config,
        training_config=training_config,
        feature_names=features,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cuda:0")

    model.load_state_dict(checkpoint["state_dict"])

    dataset = ConversationDataset(
        dataset_dir=dataset_dir,
        conv_ids=all_ids,
        speaker_ids=speaker_ids,
        features=dataset_config["features"],
        zero_pad=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=8,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        precision="16-mixed",
        devices=[0],
        logger=None,
        enable_progress_bar=True,
    )

    output = trainer.predict(model=model, dataloaders=[dataloader])

    att_path = path.join(out_dir, "output", "attention")
    try:
        os.makedirs(att_path)
    except:
        pass

    for (
        our_features_pred,
        our_scores_all,
        our_history_mask,
        their_scores_all,
        their_history_mask,
        batch,
        predict_next,
    ) in output:
        for i in range(len(batch.conv_id)):
            conv_id = batch.conv_id[i]

            att_conv_path = path.join(att_path, str(conv_id))
            try:
                os.makedirs(att_conv_path)
            except:
                pass

            # Save output features
            our_features_pred_i = our_features_pred[i, predict_next[i]]
            torch.save(our_features_pred_i, path.join(att_conv_path, "y_hat.pt"))

            their_scores_i = their_scores_all[i]
            our_scores_i = our_scores_all[i]

            for f_i, f in enumerate(features):
                att_conv_feature_path = path.join(att_conv_path, f)

                try:
                    os.makedirs(att_conv_feature_path)
                except:
                    pass

                their_scores_f_pred_timesteps = their_scores_i[predict_next[i], f_i]

                their_scores_f: list[Tensor] = []
                for pred_timestep, their_scores_t in enumerate(
                    their_scores_f_pred_timesteps
                ):
                    scores_truncated = their_scores_t[their_history_mask[i, :-1]]
                    scores_truncated = scores_truncated[scores_truncated > 0.0]
                    if not torch.isclose(scores_truncated.sum(), torch.tensor(1.0)):
                        print(their_scores_t)
                        print(their_scores_t[their_history_mask[i, :-1]])
                        print(scores_truncated)
                        raise Exception("wut")
                    their_scores_f.append(scores_truncated)
                torch.save(
                    their_scores_f, path.join(att_conv_feature_path, "partner.pt")
                )

                our_scores_f_pred_timesteps = our_scores_i[predict_next[i], f_i]
                our_scores_f: list[Tensor] = []
                for pred_timestep, our_scores_t in enumerate(
                    our_scores_f_pred_timesteps
                ):
                    scores_truncated = our_scores_t[our_history_mask[i, :-1]]
                    scores_truncated = scores_truncated[scores_truncated > 0.0]

                    our_scores_f.append(scores_truncated)

                torch.save(our_scores_f, path.join(att_conv_feature_path, "agent.pt"))
