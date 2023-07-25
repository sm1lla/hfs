from experiments.experiments_eager import classification_experiments
import argparse


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-e", "--experiments", type=str, nargs="+", help="Experiments to run"
    )
    argParser.add_argument("--wandb", action=argparse.BooleanOptionalAction)

    args = argParser.parse_args()

    if args.experiments == None:
        classification_experiments(use_wandb=args.wandb)
    else:
        classification_experiments(use_wandb=args.wandb, experiments=args.experiments)


if __name__ == "__main__":
    main()
