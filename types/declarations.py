from typing import TypedDict


class Answers(TypedDict): 
    """
    Type for answers dictionary
    """
    answers_end: int 
    answers_start: int
    text: str
