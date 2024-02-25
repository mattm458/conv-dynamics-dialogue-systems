import os
from os import path

import pandas as pd
import torch
import ujson
from lightning import pytorch as pl
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from cdmodel.consts import SPEAKER_ROLE_AGENT_IDX, SPEAKER_ROLE_PARTNER_IDX
from cdmodel.data.dataloader_manifest_1 import collate_fn
from cdmodel.data.dataset_manifest_1 import ConversationDataset
from cdmodel.model.config import get_model


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
        conv_ids=all_ids[:10],
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

    out_path = path.join(out_dir)
    try:
        os.makedirs(out_path)
    except:
        pass

    with open(path.join(out_path, "model_config.json"), "w") as outfile:
        ujson.dump(model_config, outfile, indent=2)
    with open(path.join(out_path, "training_config.json"), "w") as outfile:
        ujson.dump(training_config, outfile, indent=2)
    with open(path.join(out_path, "dataset_config.json"), "w") as outfile:
        ujson.dump(dataset_config, outfile, indent=2)

    for (
        agent_features_pred,
        agent_scores,
        partner_scores,
        combined_scores,
        batch,
        predict_next,
        speaker_role_idx,
    ) in tqdm(output, desc="Processing output"):
        partner_history_mask = (speaker_role_idx == SPEAKER_ROLE_PARTNER_IDX)[:, :-1]
        agent_history_mask = (speaker_role_idx == SPEAKER_ROLE_AGENT_IDX)[:, :-1]

        for i in range(len(batch.conv_id)):
            conv_id = batch.conv_id[i]
            conv_path = path.join(out_path, str(conv_id))
            predict_next_i = predict_next[i]

            partner_history_mask_i = partner_history_mask[i]
            agent_history_mask_i = agent_history_mask[i]

            os.makedirs(conv_path, exist_ok=True)

            # Save output features
            agent_features_pred_i = agent_features_pred[i, predict_next_i]
            torch.save(agent_features_pred_i, path.join(conv_path, "y_hat.pt"))

            for decoder_i in range(model_config["args"]["num_decoders"]):
                att_conv_feature_path = path.join(conv_path, f"decoder_{decoder_i}")
                os.makedirs(att_conv_feature_path, exist_ok=True)

                if model_config["args"]["attention_style"] in {
                    "dual",
                    "single_partner",
                }:
                    their_scores_f_pred_timesteps = partner_scores[
                        i, decoder_i, predict_next_i
                    ][:, partner_history_mask_i]
                    their_scores_f: list[Tensor] = []
                    for their_scores_t in their_scores_f_pred_timesteps:
                        their_scores_f.append(their_scores_t[their_scores_t > 0.0])
                    torch.save(
                        their_scores_f, path.join(att_conv_feature_path, "partner.pt")
                    )

                if model_config["args"]["attention_style"] in {"dual"}:
                    our_scores_f_pred_timesteps = agent_scores[
                        i, decoder_i, predict_next_i
                    ][:, agent_history_mask_i]
                    our_scores_f: list[Tensor] = []
                    for our_scores_t in our_scores_f_pred_timesteps:
                        our_scores_f.append(our_scores_t[our_scores_t > 0.0])
                    torch.save(
                        our_scores_f, path.join(att_conv_feature_path, "agent.pt")
                    )

                if model_config["args"]["attention_style"] in {"single_both"}:
                    combined_scores_f_pred_timesteps = combined_scores[
                        i, decoder_i, predict_next_i
                    ]
                    combined_scores_f: list[Tensor] = []
                    for combined_scores_t in combined_scores_f_pred_timesteps:
                        combined_scores_f.append(
                            combined_scores_t[combined_scores_t > 0.0]
                        )
                    torch.save(
                        combined_scores_f,
                        path.join(att_conv_feature_path, "combined.pt"),
                    )
