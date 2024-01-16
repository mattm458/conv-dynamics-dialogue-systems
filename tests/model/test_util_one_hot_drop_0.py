import pytest
import torch

from cdmodel.model.util import one_hot_drop_0


@pytest.fixture
def idx_tensor():
    return torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def one_hot_tensor():
    return torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def f_one_hot_tensor():
    return torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        ],
        dtype=torch.long,
    )


def test_one_hot_drop_0(idx_tensor, one_hot_tensor):
    encoded = one_hot_drop_0(idx_tensor)

    assert torch.equal(encoded, one_hot_tensor)


def test_one_hot_drop_0_num_classes(idx_tensor, f_one_hot_tensor, mocker):
    f_one_hot = mocker.patch.object(
        torch.nn.functional, "one_hot", return_value=f_one_hot_tensor
    )
    one_hot_drop_0(idx_tensor, num_classes=30)

    f_one_hot.assert_called_once_with(idx_tensor, num_classes=30)


def test_one_hot_drop_0_num_classes_default(idx_tensor, f_one_hot_tensor, mocker):
    f_one_hot = mocker.patch.object(
        torch.nn.functional, "one_hot", return_value=f_one_hot_tensor
    )
    one_hot_drop_0(idx_tensor)

    f_one_hot.assert_called_once_with(idx_tensor, num_classes=-1)
