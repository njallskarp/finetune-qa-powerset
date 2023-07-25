from collections.abc import Generator
from itertools import chain, combinations

import ruquad_labeling
from datasets import QaDataset
from tokenizers import Encoding
from torch.utils.data import ConcatDataset, DataLoader
from transformers.convert_slow_tokenizer import Tokenizer

from definitions.declarations import AnswerKey, LoaderDict, QaData


def __add_token_positions(
    encodings: Encoding, tokenizer: Tokenizer, answers: list[AnswerKey]
) -> None:
    """
    Code reused from open source colab notebook:
    https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb
    """

    start_pos, end_pos = [], []

    for i, answer in enumerate(answers):
        start = encodings.char_to_token(i, answer["answer_start"])
        end = encodings.char_to_token(i, answer["answer_end"])

        if start is None:
            start = tokenizer.model_max_length
        if end is None:
            end = encodings.char_to_token(i, answer["answer_end"] - 1)
        if end is None:
            end = tokenizer.model_max_length

        start_pos.append(start)
        end_pos.append(end)

    encodings.update({"start_positions": start_pos, "end_positions": end_pos})


def __correct_span_errors(dataset: dict[str, list[QaData]]) -> None:
    """
    Code reused from open source colab notebook and adapted:
    https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb
    """
    data = []
    train_data = dataset["train"]
    test_data = dataset["test"]
    data.extend(train_data)
    data.extend(test_data)

    for info in data:
        answer = info["answer_info"]
        text = info["paragraph"]

        real_answer = answer["text"]
        start_idx = answer["answer_start"]
        # Get the real end index
        end_idx = start_idx + len(real_answer)
        # Deal with the problem of 1 or 2 more characters
        if text[start_idx:end_idx] == real_answer:
            answer["answer_end"] = end_idx
        # When the real answer is more by one character
        elif text[start_idx - 1 : end_idx - 1] == real_answer:
            answer["answer_start"] = start_idx - 1
            answer["answer_end"] = end_idx - 1
        # When the real answer is more by two characters
        elif text[start_idx - 2 : end_idx - 2] == real_answer:
            answer["answer_start"] = start_idx - 2
            answer["answer_end"] = end_idx - 2


def __make_qa_datasets(
    dataset: dict[str, dict[str, list[QaData]]], tokenizer: Tokenizer
) -> dict[str, dict[str, QaDataset]]:
    qa_datasets: dict[str, QaDataset] = {}
    dataset = ruquad_labeling.get_data()

    for domain in dataset.keys():
        train_texts = [qa["paragraph"] for qa in dataset[domain]["train"]]
        train_questions = [qa["question"] for qa in dataset[domain]["train"]]
        train_answers = [qa["answer_info"] for qa in dataset[domain]["train"]]

        test_texts = [qa["paragraph"] for qa in dataset[domain]["test"]]
        test_questions = [qa["question"] for qa in dataset[domain]["test"]]
        test_answers = [qa["answer_info"] for qa in dataset[domain]["test"]]

        __correct_span_errors(dataset[domain])

        train_encodings = tokenizer(
            train_texts, train_questions, truncation=True, padding=True
        )
        test_encodings = tokenizer(
            test_texts, test_questions, truncation=True, padding=True
        )

        __add_token_positions(train_encodings, tokenizer, train_answers)
        __add_token_positions(test_encodings, tokenizer, test_answers)

        train_dataset = QaDataset(train_encodings, is_train=True)
        test_dataset = QaDataset(test_encodings, is_train=False)

        qa_datasets[domain] = {"train": train_dataset, "test": test_dataset}

    return qa_datasets


def __domain_powersets(domains: list[str]) -> set[tuple[str]]:
    return set(
        chain.from_iterable(
            combinations(domains, r) for r in range(1, len(domains) + 1)
        )
    )


def get_powerset_dataloaders(
    tokenizer: Tokenizer, batch_size: int
) -> Generator[LoaderDict, None, None]:
    raw_data = ruquad_labeling.get_data()
    qa_datasets = __make_qa_datasets(raw_data, tokenizer)

    data_domains = list(raw_data.keys())
    powerset = __domain_powersets(data_domains)

    for domains in powerset:
        train_concat_datasets = [qa_datasets[domain]["train"] for domain in domains]
        test_concat_datasets = [qa_datasets[domain]["test"] for domain in domains]

        test_raw_data = [raw_data[domain]["test"] for domain in domains]
        test_raw_data_flat: list[QaData] = list(chain.from_iterable(test_raw_data))

        train_qa_dataset = ConcatDataset(train_concat_datasets)
        test_qa_dataset = ConcatDataset(test_concat_datasets)

        train_loader = DataLoader(train_qa_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_qa_dataset, batch_size=batch_size, shuffle=True)



        yield {
            "train": train_loader,
            "test": test_loader,
            "data": test_raw_data_flat,
            "domains": domains
        }
