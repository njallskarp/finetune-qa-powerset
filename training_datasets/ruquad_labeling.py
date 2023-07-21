import json
import os

import requests
from dotenv import dotenv_values

from definitions.declarations import AnswerKey, CleanRUResponse, QaData, RUResponse

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

    for i in PROJECT_IDS:
        p_records = __records(i, key)
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


def __sources(records: list[CleanRUResponse]) -> set[str]:
    sources = set()

    for rec in records:
        source = rec["meta"]["source"]
        sources.add(source)

    return sources


def __structure(sources: set[str]) -> dict[str, dict[str, list[QaData]]]:
    structure: dict[str, dict[str, list[QaData]]] = {}
    for src in sources:
        structure[src] = {"train": [], "test": []}
    return structure


def get_data() -> dict[str, dict[str, list[QaData]]]:
    seen_qs = set()
    seen_as = set()

    records = __all_project_records()
    sources = __sources(records)
    data = __structure(sources)

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
        source = record["meta"]["source"]

        answer: str = paragraph[start:end]
        answer_info: AnswerKey = {
            "answer_start": start,
            "answer_end": end,
            "text": answer,
        }

        qaData: QaData = {
            "paragraph": paragraph,
            "question": question,
            "answer_info": answer_info,
        }

        data[source][split].append(qaData)

    DEST = "./datafiles/ruquad_1_unstandardized.zip"
    URL = "https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/311"

    os.system(f"curl --output {DEST} {URL}")

    return data


print(get_data())
