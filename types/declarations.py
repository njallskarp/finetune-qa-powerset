from typing import TypedDict


class Answers(TypedDict):
    """
    Type for answers dictionary
    """
    answers_end: int
    answers_start: int
    text: str

class Metrics(TypedDict):
    """the scores reported for each record
    and the dataset as a whole
    """
    f1: float
    recall: float
    precision: float
