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


class Value(TypedDict):
    start: int
    end: int
    text: str
    labels: list[str]


class Result(TypedDict):
    value: Value
    id: str
    from_name: str
    to_name: str
    type: str
    origin: str


class Prediction(TypedDict):
    id: int
    model_version: str
    created_ago: str
    result: list[Result]
    score: float
    cluster: str
    neighbors: str
    mislabeling: float
    created_at: str
    updated_at: str
    task: int


class Annotation(TypedDict):
    id: int
    completed_by: int
    result: list[Result]
    was_cancelled: bool
    ground_truth: bool
    created_at: str
    updated_at: str
    lead_time: float
    prediction: Prediction
    result_count: int
    unique_id: str
    last_action: str
    task: int
    project: int
    updated_by: int
    parent_prediction: int
    parent_annotation: str
    last_created_by: str


class MetaInfo(TypedDict):
    question: str
    paragraph: str
    start: int
    end: int
    span: str
    type: str
    question_id: str
    answer_id: str
    source: str
    article_id: str
    section: str
    split: str


class RUResponse(TypedDict):
    id: int
    annotations: list[Annotation]
    file_upload: str
    drafts: list[str]
    predictions: list[int]
    data: dict[str, str]
    meta: MetaInfo
    created_at: str
    updated_at: str
    inner_id: int
    total_annotations: int
    cancelled_annotations: int
    total_predictions: int
    comment_count: int
    unresolved_comment_count: int
    last_comment_updated_at: str
    project: int
    updated_by: int
    comment_authors: list[str]


class CleanRUResponse(TypedDict):
    annotations: list[Annotation]
    meta: MetaInfo


class AnswerKey(TypedDict):
    answer_end: int
    answer_start: int
    text: str
