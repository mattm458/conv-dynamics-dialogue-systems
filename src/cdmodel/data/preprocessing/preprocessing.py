from os import mkdir


def preprocess(dataset, dataset_dir, embedding_out_dir):
    try:
        mkdir(embedding_out_dir)
    except:
        print(f"Warning: Embedding directory {embedding_out_dir} already exists")

    runner = None

    if dataset == "fisher":
        from data.preprocessing.datasets import fisher

        runner = fisher.run

    if runner is None:
        print(f"No preprocessor found for dataset {dataset}, exiting...")
        return

    runner(dataset_dir, embedding_out_dir)
