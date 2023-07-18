import json
import os

import requests
from dotenv import dotenv_values

# from definitions.declarations import AnswerKey
# from ..definitions.declarations import AnswerKey, CleanRUResponse, RUResponse
from definitions.declarations import AnswerKey, CleanRUResponse, RUResponse

PROJECT_IDS = [1, 4, 5]


def __api_key() -> str:
    env_vars = dotenv_values()
    api_key: str = env_vars.get("API_KEY", "")
    return api_key


def __records(project_id: int, api_key: str) -> list[CleanRUResponse]:
    url = (
        f"https://labeling.gameqa.app/api/projects/{project_id}/export?exportType=JSON"
    )
    headers = {"Authorization": f"Token {api_key}"}
    res = requests.get(url, headers=headers)
    records: list[RUResponse] = json.loads(res.text)
    clean_records: list[CleanRUResponse] = [
        {"annotations": record.get("annotations", []), "meta": record.get("meta", {})}
        for record in records
    ]
    return clean_records


def __all_project_records() -> list[CleanRUResponse]:
    key = __api_key()
    all_project_records = []

    for id in PROJECT_IDS:
        p_records = __records(id, key)
        all_project_records.extend(p_records)

    return all_project_records


def __invalid_record(record: CleanRUResponse) -> bool:
    if len(record["annotations"]) == 0:
        return True
    annotation = record["annotations"][0]["result"]
    if not annotation:
        return True
    annotation_val = annotation[0]["value"]
    if "labels" not in annotation_val:
        return True
    label = annotation_val["labels"][0]
    if label == "Archive":
        return True
    paragraph = record["meta"]["paragraph"]
    if len(paragraph.split(" ")) > 300:
        return True

    return False


def get_data() -> (
    tuple[
        tuple[list[str], list[str], list[AnswerKey]],
        tuple[list[str], list[str], list[AnswerKey]],
    ]
):
    test_texts, train_texts = [], []
    test_questions, train_questions = [], []
    test_answers, train_answers = [], []

    seen_qs = set()
    seen_as = set()

    records = __all_project_records()

    for record in records:
        if __invalid_record(record):
            continue

        annotation = record["annotations"][0]["result"][0]["value"]

        start: int = annotation["start"]
        end: int = annotation["end"]
        paragraph = record["meta"]["paragraph"]

        answer_key = (paragraph, start, end)
        question = record["meta"]["question"]

        if question in seen_qs or answer_key in seen_as:
            continue

        seen_qs.add(question)
        seen_as.add(answer_key)

        split = record["meta"]["split"]
        answer: str = paragraph[start:end]
        answer_info: AnswerKey = {
            "answer_end": end,
            "answer_start": start,
            "text": answer,
        }

        if split == "train":
            train_texts.append(paragraph)
            train_questions.append(question)
            train_answers.append(answer_info)

        if split == "test":
            test_texts.append(paragraph)
            test_questions.append(question)
            test_answers.append(answer_info)

    DEST = "./datafiles/ruquad_1_unstandardized.zip"
    URL = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/311"

    os.system(f"curl --output {DEST} {URL}")

    return (
        (train_texts, train_questions, train_answers),
        (test_texts, test_questions, test_answers),
    )


print(get_data())
