import argparse


def train_subparser(subparsers):
    train_subparser = subparsers.add_parser("train", help="Train the model")

    train_subparser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Resume training from the given checkpoint.",
        default=None,
    )

    train_subparser.add_argument(
        "--training-data",
        type=str,
        required=True,
        help="Path to the training data",
    )

    train_subparser.add_argument(
        "--validation-data",
        type=str,
        required=True,
        help="Path to the validation data",
    )

    train_subparser.add_argument(
        "--device",
        type=int,
        required=True,
        help="The GPU to use for training",
    )

    train_subparser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="The number of dataloader workers",
    )


def test_subparser(subparsers):
    test_subparser = subparsers.add_parser("test", help="Evaluate a trained model")

    test_subparser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The checkpoint for a trained model to evaluate",
    )

    test_subparser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to the test data",
    )

    test_subparser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Where to save output.",
    )


def torchscript_subparser(subparsers):
    torchscript_subparser = subparsers.add_parser(
        "torchscript", help="Export the model to TorchScript"
    )

    torchscript_subparser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="A trained checkpoint to export with the model.",
    )

    torchscript_subparser.add_argument(
        "--filename",
        type=str,
        required=False,
        help="The TorchScript model filename",
        default=None,
    )


parser = argparse.ArgumentParser(description="Train a model")

parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="A JSON file containing model configurations and hyperparameters.",
)

subparsers = parser.add_subparsers(required=True, dest="mode")

train_subparser(subparsers)
test_subparser(subparsers)
torchscript_subparser(subparsers)

args = parser.parse_args()
