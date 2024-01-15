import pytest
import torch

from cdmodel.model.util import get_role_idx


@pytest.fixture
def speaker_id_idx():
    return torch.tensor(
        [
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [5, 6, 5, 6, 5, 6],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def speaker_id_idx_zero_pad():
    return torch.tensor(
        [
            [0, 1, 2, 1, 2, 1, 2],
            [0, 3, 4, 3, 4, 3, 4],
            [0, 5, 6, 5, 6, 5, 6],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def speaker_id_idx_1():
    return torch.tensor([1, 3, 5], dtype=torch.long)


@pytest.fixture
def speaker_id_idx_2():
    return torch.tensor([2, 4, 6], dtype=torch.long)


@pytest.fixture
def speaker_id_rand():
    return torch.tensor([0.1, 0.4, 0.7])


@pytest.fixture
def speaker_id_rand_idx_1():
    return torch.tensor([2, 4, 5], dtype=torch.long)


@pytest.fixture
def speaker_id_rand_idx_2():
    return torch.tensor([1, 3, 6], dtype=torch.long)


def test_fails_not_implemented(speaker_id_idx):
    with pytest.raises(NotImplementedError) as e:
        get_role_idx(
            speaker_id_idx=speaker_id_idx, role_assignment="asdf", zero_pad=False
        )

    assert str(e.value) == 'Role assignment method "asdf" not supported'


def test_fails_rand_missing_generator(speaker_id_idx):
    with pytest.raises(Exception) as e:
        get_role_idx(
            speaker_id_idx=speaker_id_idx, role_assignment="random", zero_pad=False
        )

    assert (
        str(e.value)
        == "If using random role assignment, a Generator object must be given"
    )


def test_get_agent_partner_idx_first(
    speaker_id_idx, speaker_id_idx_1, speaker_id_idx_2
):
    agent_idx, partner_idx = get_role_idx(
        speaker_id_idx=speaker_id_idx, role_assignment="first", zero_pad=False
    )

    assert torch.equal(agent_idx, speaker_id_idx_1)
    assert torch.equal(partner_idx, speaker_id_idx_2)


def test_get_agent_partner_idx_second(
    speaker_id_idx, speaker_id_idx_1, speaker_id_idx_2
):
    agent_idx, partner_idx = get_role_idx(
        speaker_id_idx=speaker_id_idx, role_assignment="second", zero_pad=False
    )

    assert torch.equal(agent_idx, speaker_id_idx_2)
    assert torch.equal(partner_idx, speaker_id_idx_1)


def test_get_agent_partner_idx_random(
    speaker_id_idx,
    speaker_id_rand,
    speaker_id_rand_idx_1,
    speaker_id_rand_idx_2,
    mocker,
):
    generator = torch.Generator()
    rand = mocker.patch.object(torch, "rand", return_value=speaker_id_rand)

    batch_size = speaker_id_idx.shape[0]

    agent_idx, partner_idx = get_role_idx(
        speaker_id_idx=speaker_id_idx,
        role_assignment="random",
        generator=generator,
        zero_pad=False,
    )

    assert torch.equal(agent_idx, speaker_id_rand_idx_1)
    assert torch.equal(partner_idx, speaker_id_rand_idx_2)
    rand.assert_called_once_with(
        batch_size, generator=generator, device=generator.device
    )


def test_get_agent_partner_idx_first_zero_pad(
    speaker_id_idx_zero_pad, speaker_id_idx_1, speaker_id_idx_2
):
    agent_idx, partner_idx = get_role_idx(
        speaker_id_idx=speaker_id_idx_zero_pad, role_assignment="first", zero_pad=True
    )

    assert torch.equal(agent_idx, speaker_id_idx_1)
    assert torch.equal(partner_idx, speaker_id_idx_2)


def test_get_agent_partner_idx_second_zero_pad(
    speaker_id_idx_zero_pad, speaker_id_idx_1, speaker_id_idx_2
):
    agent_idx, partner_idx = get_role_idx(
        speaker_id_idx=speaker_id_idx_zero_pad, role_assignment="second", zero_pad=True
    )

    assert torch.equal(agent_idx, speaker_id_idx_2)
    assert torch.equal(partner_idx, speaker_id_idx_1)


def test_get_agent_partner_idx_random_zero_pad(
    speaker_id_idx_zero_pad,
    speaker_id_rand,
    speaker_id_rand_idx_1,
    speaker_id_rand_idx_2,
    mocker,
):
    generator = torch.Generator()
    rand = mocker.patch.object(torch, "rand", return_value=speaker_id_rand)

    batch_size = speaker_id_idx_zero_pad.shape[0]

    agent_idx, partner_idx = get_role_idx(
        speaker_id_idx=speaker_id_idx_zero_pad,
        role_assignment="random",
        generator=generator,
        zero_pad=True,
    )

    assert torch.equal(agent_idx, speaker_id_rand_idx_1)
    assert torch.equal(partner_idx, speaker_id_rand_idx_2)
    rand.assert_called_once_with(
        batch_size, generator=generator, device=generator.device
    )
