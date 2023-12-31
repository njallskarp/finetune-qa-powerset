import argparse


def parse_args() -> argparse.Namespace:
    """
    Parser command argument for finetuning BERT such as

        returns argparse.Namespace
    """

    parser = argparse.ArgumentParser("Finetune QA on Domain Powerset")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs")
    parser.add_argument(
        "--bert_name", type=str, default="bert-base-uncased", help="name of bert model"
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["w", "v", "n"],
        help="domains to include",
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    """Finetunes BERT on Domain Powerset

    Args:
        args (argparse.Namespace): the arguments parsed from CLI
    """

    # TODO: set up training and call training pipeline


if __name__ == "__main__":
    args = parse_args()
