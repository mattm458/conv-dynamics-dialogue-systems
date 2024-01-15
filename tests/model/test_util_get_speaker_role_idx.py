import pytest
import torch

from cdmodel.model.util import get_speaker_role_idx
from cdmodel.consts import SPEAKER_ROLE_AGENT_IDX as A, SPEAKER_ROLE_PARTNER_IDX as P


@pytest.fixture
def speaker_identity_idx():
    return torch.tensor(
        [
            [1, 2, 1, 2, 1, 2, 0],
            [3, 4, 3, 4, 3, 4, 0],
            [5, 6, 5, 6, 5, 6, 6],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def speaker_identity_idx_zero_pad():
    return torch.tensor(
        [
            [0, 1, 2, 1, 2, 1, 2, 0],
            [0, 3, 4, 3, 4, 3, 4, 0],
            [0, 5, 6, 5, 6, 5, 6, 6],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def agent_identity_idx():
    return torch.tensor([1, 4, 5], dtype=torch.long)


@pytest.fixture
def partner_identity_idx():
    return torch.tensor([2, 3, 6], dtype=torch.long)


@pytest.fixture
def speaker_role_idx():
    return torch.tensor(
        [
            [A, P, A, P, A, P, 0],
            [P, A, P, A, P, A, 0],
            [A, P, A, P, A, P, P],
        ]
    )


@pytest.fixture
def speaker_role_idx_zero_pad():
    return torch.tensor(
        [
            [0, A, P, A, P, A, P, 0],
            [0, P, A, P, A, P, A, 0],
            [0, A, P, A, P, A, P, P],
        ]
    )


def test_get_speaker_role_idx(
    speaker_identity_idx,
    agent_identity_idx,
    partner_identity_idx,
    speaker_role_idx,
):
    speaker_role_idx_output = get_speaker_role_idx(
        speaker_identity_idx=speaker_identity_idx,
        agent_identity_idx=agent_identity_idx,
        partner_identity_idx=partner_identity_idx,
    )

    assert torch.equal(speaker_role_idx_output, speaker_role_idx)


def test_get_speaker_role_idx_zero_pad(
    speaker_identity_idx_zero_pad,
    agent_identity_idx,
    partner_identity_idx,
    speaker_role_idx_zero_pad,
):
    speaker_role_idx_output = get_speaker_role_idx(
        speaker_identity_idx=speaker_identity_idx_zero_pad,
        agent_identity_idx=agent_identity_idx,
        partner_identity_idx=partner_identity_idx,
    )

    assert torch.equal(speaker_role_idx_output, speaker_role_idx_zero_pad)
