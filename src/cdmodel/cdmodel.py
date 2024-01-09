import json
from os import path
from typing import Optional

import click
from click import Context


@click.group()
@click.pass_context
@click.option("--config", type=str, required=False)
@click.option("--device", type=int, required=False, default=0)
def main(ctx: Context, config: Optional[str], device: int):
    if config is not None:
        with open(config) as infile:
            ctx.obj["config"] = json.load(infile)

    ctx.obj["device"] = device


@main.command()
@click.pass_context
@click.option("--dataset-dir", required=True, type=str)
@click.option(
    "--method",
    required=True,
    default="standard",
    type=click.Choice(["standard", "cv"]),
)
@click.option("--n-jobs", required=False, default=2, type=int)
@click.option("--out-dir", required=False, type=str)
def train(
    ctx: Context,
    dataset_dir: str,
    method: str,
    n_jobs: int,
    out_dir: str,
):
    if ctx.obj["config"] is None:
        raise Exception(
            "Training requires a configuration to be specified with --config"
        )

    if method == "cv":
        raise NotImplementedError()
        # from cdmodel.train_manifest_v1 import cross_validate_52

        # cross_validate_52(
        #     dataset_config=ctx.obj["config"]["dataset"],
        #     training_config=ctx.obj["config"]["training"],
        #     model_config=ctx.obj["config"]["model"],
        #     device=ctx.obj["device"],
        #     n_jobs=n_jobs,
        #     resume=resume,
        #     embeddings_dir=embeddings_dir,
        #     conversation_data_dir=conversation_data_dir,
        # )
    elif method == "standard":
        from cdmodel.train_manifest_v1 import standard_train

        standard_train(
            dataset_dir=dataset_dir,
            dataset_config=ctx.obj["config"]["dataset"],
            training_config=ctx.obj["config"]["training"],
            model_config=ctx.obj["config"]["model"],
            device=ctx.obj["device"],
            out_dir_override=out_dir,
        )


@main.command(help="Produce the statistics and graphs used in the paper.")
@click.pass_context
@click.option("--results-dir", required=True, type=str)
@click.option("--n-jobs", required=False, default=2, type=int)
@click.option("--embeddings-dir", required=True, type=str)
@click.option("--conversation-data-dir", required=True, type=str)
def stats(
    ctx: Context,
    results_dir: str,
    n_jobs: int,
    embeddings_dir: str,
    conversation_data_dir: str,
):
    from cdmodel.stats import do_stats

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
@click.option("--dataset", required=True, type=str)
@click.option("--dataset-dir", required=True, type=str)
@click.option("--out-dir", required=False, type=str)
@click.option("--debug", is_flag=True, default=False)
@click.pass_context
def preprocess(
    ctx: Context, dataset: str, dataset_dir: str, out_dir: Optional[str], debug: bool
):
    from cdmodel.preprocessing import preprocess

    preprocess(
        dataset_name=dataset,
        dataset_dir=path.normpath(dataset_dir),
        out_dir=out_dir,
        n_jobs=8,
        debug=debug,
    )
    pass


@main.command()
@click.pass_context
def test(ctx):
    pass


@main.command()
@click.pass_context
def torchscript(ctx):
    pass


def start():
    main(obj={})


if __name__ == "__main__":
    start()
