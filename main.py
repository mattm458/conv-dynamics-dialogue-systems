#! python

import json

import click

from stats import do_stats
from train import cross_validate_52, standard_train


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
@click.option("--resume", required=False, type=str)
@click.option("--embeddings-dir", required=True, type=str)
@click.option("--conversation-data-dir", required=True, type=str)
def train(ctx, method, n_jobs, resume, embeddings_dir, conversation_data_dir):
    if method == "cv":
        cross_validate_52(
            dataset_config=ctx.obj["config"]["dataset"],
            training_config=ctx.obj["config"]["training"],
            model_config=ctx.obj["config"]["model"],
            device=ctx.obj["device"],
            n_jobs=n_jobs,
            resume=resume,
            embeddings_dir=embeddings_dir,
            conversation_data_dir=conversation_data_dir,
        )
    elif method == "standard":
        standard_train(
            dataset_config=ctx.obj["config"]["dataset"],
            training_config=ctx.obj["config"]["training"],
            model_config=ctx.obj["config"]["model"],
            device=ctx.obj["device"],
            n_jobs=n_jobs,
            resume=resume,
            embeddings_dir=embeddings_dir,
            conversation_data_dir=conversation_data_dir,
        )
    pass


@main.command(help="Produce the statistics and graphs used in the paper.")
@click.pass_context
@click.option("--results-dir", required=True, type=str)
@click.option("--n-jobs", required=False, default=2, type=int)
@click.option("--embeddings-dir", required=True, type=str)
@click.option("--conversation-data-dir", required=True, type=str)
def stats(ctx, results_dir, n_jobs, embeddings_dir, conversation_data_dir):
    do_stats(
        dataset_config=ctx.obj["config"]["dataset"],
        training_config=ctx.obj["config"]["training"],
        model_config=ctx.obj["config"]["model"],
        results_dir=results_dir,
        device=ctx.obj["device"],
        n_jobs=n_jobs,
        embeddings_dir=embeddings_dir,
        conversation_data_dir=conversation_data_dir,
    )
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
