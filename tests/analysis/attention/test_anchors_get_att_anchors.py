import pytest
import torch

from cdmodel.analysis.attention.anchors import Anchor, get_att_anchors


@pytest.fixture
def att_scores_end_1():
    return [
        torch.tensor([1.0]),
        torch.tensor([0.1, 0.9]),
        torch.tensor([0.05, 0.9, 0.05]),
        torch.tensor([0.1, 0.1, 0.1, 0.7]),
        torch.tensor([0.1, 0.1, 0.1, 0.6, 0.1]),
        torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.5]),
    ]


@pytest.fixture
def anchors_end_1():
    return [
        Anchor(0, [0]),
        Anchor(1, [1, 2]),
        Anchor(3, [3, 4]),
        Anchor(5, [5]),
    ]


@pytest.fixture
def att_scores_end_2():
    return [
        torch.tensor([1.0]),
        torch.tensor([0.1, 0.9]),
        torch.tensor([0.05, 0.9, 0.05]),
        torch.tensor([0.1, 0.1, 0.1, 0.7]),
        torch.tensor([0.1, 0.1, 0.1, 0.6, 0.1]),
        torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.5]),
        torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1]),
    ]


@pytest.fixture
def anchors_end_2():
    return [
        Anchor(0, [0]),
        Anchor(1, [1, 2]),
        Anchor(3, [3, 4]),
        Anchor(5, [5, 6]),
    ]


def test_get_att_anchors_end_1(att_scores_end_1, anchors_end_1):
    anchors_out = get_att_anchors(att_scores=att_scores_end_1)

    assert anchors_out == anchors_end_1


def test_get_att_anchors_end_2(att_scores_end_2, anchors_end_2):
    anchors_out = get_att_anchors(att_scores=att_scores_end_2)

    assert anchors_out == anchors_end_2
