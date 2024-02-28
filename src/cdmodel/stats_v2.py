import os
from os import path
from typing import Final

import numpy as np
import pandas as pd
import torch
import ujson
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.special import kl_div
from torch import Tensor
from tqdm import tqdm

from cdmodel.consts import NORM_BY_CONV_SPEAKER_POSTFIX

pd.options.mode.copy_on_write = True

FEATURES: Final[list[str]] = [
    "pitch_mean",
    "pitch_range",
    "intensity_mean_vcd",
    "jitter",
    "shimmer",
    "nhr_vcd",
    "rate",
]

FEATURES_NORM = [f"{f}_{NORM_BY_CONV_SPEAKER_POSTFIX}" for f in FEATURES]
FEATURES_PRED = [f"{f}_pred" for f in FEATURES]


def get_max_idx(scores: Tensor) -> int:
    return int(torch.argmax(scores).item())


def get_k_idx(scores: Tensor) -> tuple[list[float], list[int]]:
    values, topk = torch.topk(scores, k=min(len(scores), 3))
    return [x.item() for x in values], [x.item() for x in topk]


def quantile_diffusion_score(scores: Tensor, q: float = 0.5) -> float | None:
    if len(scores) < 1:
        return None

    return -((scores.max() - scores.quantile(q)) / (scores.max() - scores.min())).item()


def segment(conv_df: DataFrame, scores_all):
    partner = conv_df.iloc[0].side
    agent = "A" if partner == "B" else "B"

    turns = [x for _, x in conv_df.iterrows()]

    att_idx = -1
    turn_count = -1
    partner_count = -1

    segmentations = []
    partner_turns = {}

    # Determine base anchor turn
    for turn_i, turn in enumerate(turns):
        turn_count += 1

        if turn.side == agent:
            att_idx += 1

            scores = scores_all[att_idx]
            if len(scores) == 0:
                continue

            max_idx = get_max_idx(scores)
            topk_values, topk_idx = get_k_idx(scores)

            segmentations.append(
                (
                    turn_i,
                    partner_turns[max_idx],
                    [partner_turns[x] for x in topk_idx],
                    topk_values,
                    quantile_diffusion_score(scores),
                )
            )
        elif turn.side == partner:
            partner_count += 1
            partner_turns[partner_count] = turn_count

    # Group the segmentations
    segment_groups = []
    segment_group = []
    last_anchor = None
    for turn_i, max_idx, topk_idx, topk_score, qs in segmentations:
        if last_anchor is None or max_idx == last_anchor:
            segment_group.append((turn_i, max_idx, topk_idx, topk_score, qs))
        else:
            segment_groups.append((last_anchor, segment_group))
            segment_group = [(turn_i, max_idx, topk_idx, topk_score, qs)]

        last_anchor = max_idx

    # Add leftover segments
    if len(segment_group) > 0:
        segment_groups.append((max_idx, segment_group))

    # Merge the segmentation groups
    segments_merged: list[tuple] = []
    merge = []
    last_anchor = None
    for anchor, segments in segment_groups:
        if last_anchor is None or anchor == last_anchor:
            merge.extend(segments)
            last_anchor = anchor
        elif anchor != last_anchor:
            # If the new segment is really long, cancel the current merge
            if len(segments) >= 3:
                segments_merged.append((last_anchor, merge))
                merge = segments
                last_anchor = anchor
            # Alternatively, if the segment is not all topk previous anchor,
            # cancel the current merge
            elif not all([last_anchor in topk for _, _, topk, _, _ in segments]):
                segments_merged.append((last_anchor, merge))
                merge = segments
                last_anchor = anchor
            # Otherwise, continue the merge
            else:
                merge.extend(segments)

    # Clean up leftover merges
    if len(merge) > 0:
        segments_merged.append((last_anchor, merge))

    return segments_merged


def draw_att_plot(scores: list[Tensor], out_dir: str, cid: int) -> None:
    fig, axs = plt.subplots(len(scores), figsize=(25, 25))
    plt.subplots_adjust(wspace=0, hspace=0)
    for ax, att_score in zip(axs, scores):
        ax.axis("off")
        ax.imshow(
            att_score.unsqueeze(0),
            interpolation="nearest",
            vmin=0 if len(att_score) == 1 else None,
            vmax=1 if len(att_score) == 1 else None,
            aspect="auto",
        )

    plt.savefig(path.join(out_dir, f"{cid}.png"))
    plt.close()


def do_stats(results_dir: str, dataset_dir: str) -> None:
    with open(path.join(results_dir, "model_config.json")) as infile:
        model_config: dict = ujson.load(infile)

    num_decoders: Final[int] = model_config["args"]["num_decoders"]
    attention_style: str = model_config["args"]["attention_style"]

    conv_ids = [
        int(x)
        for x in os.listdir(results_dir)
        if path.isdir(path.join(results_dir, x)) and x != "export" and x != "att_plots"
    ]

    df = pd.read_csv(path.join(dataset_dir, "data-norm.csv"), engine="pyarrow")

    predictions: Final[dict[int, Tensor]] = {}

    partner_att_scores: Final[dict[int, list[list[Tensor]]]] = {}
    agent_att_scores: Final[dict[int, list[list[Tensor]]]] = {}

    for i, conv_id in tqdm(
        enumerate(conv_ids), total=len(conv_ids), desc="Loading results data"
    ):
        if attention_style in {"single_partner", "dual"}:
            partner_att_scores[conv_id] = [
                torch.load(
                    path.join(
                        results_dir,
                        str(conv_id),
                        f"decoder_{i}",
                        "partner.pt",
                    )
                )
                for i in range(num_decoders)
            ]

        if attention_style in {"dual"}:
            agent_att_scores[conv_id] = [
                torch.load(
                    path.join(
                        results_dir,
                        str(conv_id),
                        f"decoder_{i}",
                        "agent.pt",
                    )
                )
                for i in range(num_decoders)
            ]

        predictions[conv_id] = torch.load(
            path.join(
                results_dir,
                str(conv_id),
                "y_hat.pt",
            )
        )

    for f_i, f in enumerate(range(num_decoders)):
        conv_df_all = []
        for CID in tqdm(conv_ids, desc=f"Processing decoder {f}"):
            conv_df = df[df.id == CID]
            try:
                segments_conv = segment(conv_df, partner_att_scores[CID][f_i])
            except Exception as e:
                continue

            partner = conv_df.iloc[0].side
            agent = "A" if partner == "B" else "B"

            partner_df = conv_df[conv_df.side == partner]
            agent_df = conv_df[conv_df.side == agent]

            anchors = np.full((len(conv_df)), -1, dtype=int)
            top1 = np.full((len(conv_df)), -1, dtype=int)
            top2 = np.full((len(conv_df)), -1, dtype=int)
            top3 = np.full((len(conv_df)), -1, dtype=int)

            top1_score = np.full((len(conv_df)), np.nan)
            top2_score = np.full((len(conv_df)), np.nan)
            top3_score = np.full((len(conv_df)), np.nan)

            for anchor, segments in segments_conv:
                for turn_i, max_idx, topk_idx, topk_score, qs in segments:
                    anchors[turn_i] = anchor

                    if len(topk_idx) >= 1:
                        top1[turn_i] = topk_idx[0]
                        top1_score[turn_i] = topk_score[0]
                    if len(topk_idx) >= 2:
                        top2[turn_i] = topk_idx[1]
                        top2_score[turn_i] = topk_score[1]

                    if len(topk_idx) >= 3:
                        top3[turn_i] = topk_idx[2]
                        top3_score[turn_i] = topk_score[2]

            conv_df["anchors"] = anchors
            conv_df["topk_1"] = top1
            conv_df["topk_2"] = top2
            conv_df["topk_3"] = top3
            conv_df["topk_score_1"] = top1_score
            conv_df["topk_score_2"] = top2_score
            conv_df["topk_score_3"] = top3_score
            conv_df["turn_id"] = np.arange(len(conv_df))
            conv_df["agent"] = conv_df.side == agent
            conv_df["diffusion_quantile"] = qs

            features_pred = np.full((len(conv_df), 7), np.nan)
            features_pred[conv_df.agent] = predictions[CID]

            conv_df = pd.concat(
                [
                    conv_df,
                    DataFrame(
                        features_pred,
                        columns=FEATURES_PRED,
                        index=conv_df.index,
                    ),
                ],
                axis=1,
            )

            conv_df_all.append(
                conv_df[
                    FEATURES
                    + FEATURES_NORM
                    + FEATURES_PRED
                    + [
                        "id",
                        "side",
                        "agent",
                        "da_category",
                        "anchors",
                        "topk_1",
                        "topk_2",
                        "topk_3",
                        "topk_score_1",
                        "topk_score_2",
                        "topk_score_3",
                        "turn_id",
                        "transcript",
                        "speaker_id",
                        "diffusion_quantile",
                    ]
                ]
            )
            plot_out_dir = path.join(results_dir, "att_plots", f"decoder_{f}")
            os.makedirs(plot_out_dir, exist_ok=True)

            draw_att_plot(partner_att_scores[CID][f], out_dir=plot_out_dir, cid=CID)

        df_out = pd.concat(conv_df_all)

        out_path = path.join(results_dir, "export")
        os.makedirs(out_path, exist_ok=True)
        df_out.to_csv(path.join(out_path, f"decoder_{f}.csv"))
