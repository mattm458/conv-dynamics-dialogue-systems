import os
from os import path
from typing import Final, Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import stats
from torch import Tensor
from tqdm import tqdm

from cdmodel.analysis.attention.anchors import Anchor, find_anchors

Scores = list[Tensor]
RoleScores = dict[str, Scores]
FeatureRoleScores = dict[str, RoleScores]

Anchors = list[Anchor | None]
RoleAnchors = dict[str, Anchors]
FeatureRoleAnchors = dict[str, RoleAnchors]


def load_attention_conversation_dual(
    conv_dir: str, features: list[str], att_style: str, results_dir: str
) -> tuple[FeatureRoleScores, FeatureRoleAnchors]:
    feature_role_scores = {}
    feature_role_anchors = {}

    for feature in features:
        feature_agent_scores = torch.load(path.join(conv_dir, feature, "agent.pt"))
        feature_partner_scores = torch.load(path.join(conv_dir, feature, "partner.pt"))

        feature_role_scores[feature] = {
            "agent": feature_agent_scores,
            "partner": feature_partner_scores,
        }
        feature_role_anchors[feature] = {
            "agent": find_anchors(feature_agent_scores),
            "partner": find_anchors(feature_partner_scores),
        }

    return feature_role_scores, feature_role_anchors


def load_attention_all(
    features: list[str], att_style: str, results_dir: str, limit: Optional[int] = None
) -> tuple[dict[int, FeatureRoleScores], dict[int, FeatureRoleAnchors]]:
    att_dir: Final[str] = path.join(results_dir, "output", "attention")

    conv_dirs: list[str] = os.listdir(att_dir)
    if limit is not None:
        conv_dirs = conv_dirs[:limit]

    feature_role_scores_all: dict = {}
    feature_role_anchors_all: dict = {}

    if att_style == "dual":
        for conv_dir in tqdm(conv_dirs, desc="Loading attention"):
            feature_role_scores_conv, feature_role_anchors_conv = (
                load_attention_conversation_dual(
                    path.join(att_dir, conv_dir), features, att_style, results_dir
                )
            )

            feature_role_scores_all[int(conv_dir)] = feature_role_scores_conv
            feature_role_anchors_all[int(conv_dir)] = feature_role_anchors_conv

        return feature_role_scores_all, feature_role_anchors_all

    raise NotImplementedError(f"Unimplemented attention style '{att_style}'")


def diffusion_score(scores: Tensor, q: float) -> float:
    return -(
        (scores.max() - scores.quantile(q)) / (scores.max() - scores.min() + 0.1e-10)
    ).item()


def analyze_att_diffusion(
    att_all: dict[int, FeatureRoleScores], q: float
) -> dict[str, dict[str, list[float]]]:
    feature_role_metrics_median: dict[str, dict[str, list[float]]] = {}

    for feature_role_scores in tqdm(
        att_all.values(), desc="Computing anchor diffusion scores"
    ):
        for feature, role_scores in feature_role_scores.items():
            if feature not in feature_role_metrics_median:
                feature_role_metrics_median[feature] = {}

            for role, scores in role_scores.items():
                if role not in feature_role_metrics_median[feature]:
                    feature_role_metrics_median[feature][role] = []

                for predict_timestep in scores:
                    if len(predict_timestep) < 3:
                        continue

                    feature_role_metrics_median[feature][role].append(
                        diffusion_score(predict_timestep, q)
                    )

    return feature_role_metrics_median


def analyze_att_anchor_diffusion(
    anchors_all: dict[int, FeatureRoleAnchors], q: float
) -> dict[str, dict[str, list[tuple[float, int]]]]:
    feature_role_anchor_diffusion: dict[str, dict[str, list[tuple[float, int]]]] = {}

    for feature_role_anchors in tqdm(
        anchors_all.values(), desc="Computing diffusion scores"
    ):
        for feature, role_anchors in feature_role_anchors.items():
            if feature not in feature_role_anchor_diffusion:
                feature_role_anchor_diffusion[feature] = {}

            for role, anchors in role_anchors.items():
                if role not in feature_role_anchor_diffusion[feature]:
                    feature_role_anchor_diffusion[feature][role] = []

                for anchor in anchors:
                    if anchor is None:
                        continue

                    anchor_timesteps: list[Tensor] = [
                        t.scores for t in anchor.timesteps if len(t.scores) > 3
                    ]

                    if len(anchor_timesteps) == 0:
                        continue

                    mean_diffusion_score = float(
                        np.mean([diffusion_score(x, q) for x in anchor_timesteps])
                    )
                    feature_role_anchor_diffusion[feature][role].append(
                        (mean_diffusion_score, len(anchor_timesteps))
                    )

    return feature_role_anchor_diffusion


def graph_feature_role_diffusion(
    feature_role_diffusion: dict[str, dict[str, list[float]]], out_dir: str
):
    feature_role_metrics_out_dir: str = path.join(out_dir, "feature_role_diffusion")
    try:
        os.mkdir(feature_role_metrics_out_dir)
    except FileExistsError:
        print(f"Output directory {feature_role_metrics_out_dir} exists, overwriting...")

    for feature, role_metrics in tqdm(
        feature_role_diffusion.items(), desc="Writing diffusion score graphs"
    ):
        for role, metrics in role_metrics.items():
            plt.hist(metrics, label=role)

        plt.title(feature)
        plt.legend()
        plt.savefig(path.join(feature_role_metrics_out_dir, f"{feature}_diffusion.png"))
        plt.clf()


def graph_anchor_length_hist(anchors_all: dict[int, FeatureRoleAnchors], out_dir: str):
    feature_role_anchor_out_dir: str = path.join(out_dir, "anchors")
    try:
        os.mkdir(feature_role_anchor_out_dir)
    except FileExistsError:
        print(f"Output directory {feature_role_anchor_out_dir} exists, overwriting...")

    lengths = []

    feature_role_anchors: FeatureRoleAnchors
    for feature_role_anchors in anchors_all.values():
        role_anchors: RoleAnchors
        for role_anchors in feature_role_anchors.values():
            anchors: Anchors
            for anchors in role_anchors.values():
                anchor: Anchor | None
                for anchor in anchors:
                    if anchor is None:
                        continue
                    lengths.append(len(anchor.timesteps))

    plt.hist(lengths)
    plt.savefig(path.join(feature_role_anchor_out_dir, f"anchor_lengths.png"))
    plt.clf()


def graph_feature_role_anchor_density(
    feature_role_anchor_metrics: dict[str, dict[str, list[tuple[float, int]]]],
    out_dir: str,
):
    feature_role_anchor_out_dir: str = path.join(out_dir, "feature_role_anchor")
    try:
        os.mkdir(feature_role_anchor_out_dir)
    except FileExistsError:
        print(f"Output directory {feature_role_anchor_out_dir} exists, overwriting...")

    scores_all = []
    lengths_all = []

    feature_role_corr: dict = {}

    for feature, role_anchor_metrics in tqdm(
        feature_role_anchor_metrics.items(), desc="Writing density score graphs"
    ):

        for role, score_lenghts in role_anchor_metrics.items():
            scores = []
            lengths = []
            for score, length in score_lenghts:
                scores.append(score)
                scores_all.append(score)

                lengths.append(length)
                lengths_all.append(length)

            stats_result = stats.pearsonr(lengths, scores)
            feature_role_corr[(feature, role)] = [
                stats_result.statistic,
                stats_result.pvalue,
            ]

            plt.scatter(lengths, scores, label=role)

        plt.title(feature)
        plt.legend()
        plt.savefig(path.join(feature_role_anchor_out_dir, f"{feature}_anchor.png"))
        plt.clf()

    stats_result = stats.pearsonr(lengths_all, scores_all)
    feature_role_corr[("all", "all")] = [
        stats_result.statistic,
        stats_result.pvalue,
    ]

    plt.scatter(lengths_all, scores_all)
    plt.title("All anchors")
    plt.savefig(path.join(feature_role_anchor_out_dir, f"all_anchor.png"))
    plt.clf()

    df = pd.DataFrame.from_dict(feature_role_corr, orient="index")
    df.columns = pd.Index(["r", "p"])
    df["p_adj"] = stats.false_discovery_control(df.p)
    df.to_csv(path.join(feature_role_anchor_out_dir, "correlations.csv"))
    print(df)


def csv_feature_role_metrics(
    feature_role_metrics: dict[str, dict[str, list[float]]], out_dir: str
):
    metrics_dict: dict = {}

    for feature, role_metrics in feature_role_metrics.items():
        metrics_dict[feature] = []
        for role, metrics in role_metrics.items():
            metrics_dict[feature].append(np.mean(metrics))

    pd.DataFrame.from_dict(
        metrics_dict, orient="index", columns=list(role_metrics.keys())
    ).to_csv(path.join(out_dir, "feature_role_diffusion", "means.csv"))


def analyze(features: list[str], results_dir: str, out_dir: str):
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        print(f"Output directory {out_dir} exists, overwriting...")

    att_all: dict[int, FeatureRoleScores]
    anchors_all: dict[int, FeatureRoleAnchors]
    att_all, anchors_all = load_attention_all(features, "dual", results_dir, limit=None)
    feature_role_metrics_median = analyze_att_diffusion(att_all, q=0.5)
    feature_role_anchor_diffusion = analyze_att_anchor_diffusion(anchors_all, q=0.5)

    graph_anchor_length_hist(anchors_all, out_dir=out_dir)
    graph_feature_role_diffusion(feature_role_metrics_median, out_dir=out_dir)
    graph_feature_role_anchor_density(feature_role_anchor_diffusion, out_dir=out_dir)
    csv_feature_role_metrics(feature_role_metrics_median, out_dir=out_dir)
