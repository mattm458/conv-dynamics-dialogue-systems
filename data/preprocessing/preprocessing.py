def preprocess(dataset, dataset_dir):
    runner = None

    if dataset == "fisher":
        from data.preprocessing.datasets import fisher

        runner = fisher.run

    if runner is None:
        print(f"No preprocessor found for dataset {dataset}, exiting...")
        return

    runner(dataset_dir)
