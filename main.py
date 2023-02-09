#! python

import json

import click
import pandas as pd

from train import cross_validate_52

# from util.args import args


@click.group()
@click.pass_context
@click.option("--config", type=str, required=True)
@click.option("--device", type=int, required=False, default=0)
def main(ctx, config, device):
    with open(config) as infile:
        config = json.load(infile)

    ctx.obj["config"] = config
    ctx.obj["device"] = device
    pass


@main.command()
@click.pass_context
@click.option(
    "--method",
    required=True,
    default="standard",
    type=click.Choice(["standard", "cv"]),
)
@click.option("--n-jobs", required=False, default=2, type=int)
def train(ctx, method, n_jobs):
    if method == "cv":
        cross_validate_52(
            dataset_config=ctx.obj["config"]["dataset"],
            training_config=ctx.obj["config"]["training"],
            model_config=ctx.obj["config"]["model"],
            device=ctx.obj["device"],
            n_jobs=n_jobs,
            results_dir=None,
        )
    elif method == "standard":
        raise Exception("Standard training method not implemented!")
    pass


@main.command()
@click.pass_context
def test(ctx):
    pass


@main.command()
@click.pass_context
def preprocess(ctx):
    pass


@main.command()
@click.pass_context
def torchscript(ctx):
    pass


if __name__ == "__main__":
    main(obj={})

# if __name__ == "__main__":
#     if args.mode == "preprocess":
#         print("Preprocessing")
#         from data.preprocessing import preprocess

#         preprocess(args.dataset, args.dataset_dir, args.embedding_out_dir)
#     elif args.mode == "model":
#         if args.model_mode == "train":
#             print("Training")
#         elif args.model_mode == "test":
#             print("Testing")
#         elif args.model_mode == "torchscript":
#             print("Exporting torchscript")
