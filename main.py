#! python

from util.args import args

if __name__ == "__main__":
    if args.mode == "train":
        print("Training")
    elif args.mode == "test":
        print("Testing")
    elif args.mode == "torchscript":
        print("Exporting torchscript")
