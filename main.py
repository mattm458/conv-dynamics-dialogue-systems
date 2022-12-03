#! python

from util.args import args

if __name__ == "__main__":
    if args.mode == "preprocess":
        print("Preprocessing")
        from data.preprocessing import preprocess

        preprocess(args.dataset, args.dataset_dir, args.embedding_out_dir)
    elif args.mode == "model":
        if args.model_mode == "train":
            print("Training")
        elif args.model_mode == "test":
            print("Testing")
        elif args.model_mode == "torchscript":
            print("Exporting torchscript")
