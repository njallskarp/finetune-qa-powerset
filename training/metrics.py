import collections
import re
import string

import numpy as np
from tokenizers import Tokenizer
from transformers import RobertaForQuestionAnswering, pipeline

from definitions.declarations import Metrics

""" 
The code in this cell is adapted from the official SQuAD validation script

https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
"""


def __normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        """Removes 'a', 'an', 'the' from the input string

        Args:
            text (str): input string

        Returns:
            str: string without article
        """
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        """Splits string and joins it back together
        to fix artefacts caused by normalization

        Args:
            text (str): possibly erroneous string

        Returns:
            str: fixed string
        """
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        """Removes all punctuation from text

        Args:
            text (str): text with punctuation

        Returns:
            str: text without punctuation
        """
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        """Syntactic

        Args:
            text (str): input text

        Returns:
            str: text with all letters in lower case
        """
        return text.lower()

    # return normalized answer
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def __get_tokens(s: str) -> list[str]:
    """Turns a text into a list of tokens after
      normalizing the text

    Args:
        s (str): input string

    Returns:
        list[str]: list of tokens
    """
    if not s:
        return []
    return __normalize_answer(s).split()


def __span_comparison_helper(
    a_gold: str, a_pred: str
) -> tuple[int, list[str], list[str]]:
    """A reusable function that accepts a ground truth string (gold standard)
    and a predicted string and returns information that is used to calculate metrics
    such as number of same tokens, the list of predicted tokens and list of gold tokens

    Args:
        a_gold (str): groun truth text
        a_pred (str): predicted text

    Returns:
        tuple[int, list[str], list[str]]: (number of
          similar tokens, list of predicted
          tokens, list of gold tokens)
    """
    gold_toks = __get_tokens(a_gold)
    pred_toks = __get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    return num_same, pred_toks, gold_toks


def __recall(num_same: int, gold_toks: list[str]) -> float:
    """Calculates recall based on token info

    Args:
        num_same (int): number of similar tokens in grount truth and predicted string
        gold_toks (list[str]): list of gold tokens

    Returns:
        _type_: recall score
    """
    if len(gold_toks) == 0 or num_same == 0:
        return int(gold_toks == num_same)
    return 1.0 * num_same / len(gold_toks)


def __precision(num_same: int, pred_toks: list[str]) -> float:
    """Calculates precision based on token info

    Args:
        num_same (int): number of similar tokens in grount truth and predicted string
        pred_toks (list[str]): list of predicted tokens

    Returns:
        float: precision score
    """
    if len(pred_toks) == 0:
        return 0
    return 1.0 * num_same / len(pred_toks)


def __f1(num_same: int, pred_toks: list[str], gold_toks: list[str]) -> float:
    """Calculates f1-score based on token info

    Args:
        num_same (int): number of similar tokens in grount truth and predicted string
        pred_toks (list[str]): list of predicted tokens
        gold_toks (list[str]): list of gold tokens

    Returns:
        float: f1-score
    """
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)

    p = __precision(num_same, pred_toks)
    r = __recall(num_same, gold_toks)

    # f1 score formula
    EPS = 1e-8
    return 2 * (p * r) / (p + r + EPS)


def recall(a_gold: str, a_pred: str) -> float:
    """calcualtes recall for two strings

    Args:
        a_gold (str): ground truth answer
        a_pred (str): predicted answer

    Returns:
        float: recall score
    """
    num_same, _, gold_toks = __span_comparison_helper(a_gold, a_pred)
    return __recall(num_same, gold_toks)


def precision(a_gold: str, a_pred: str) -> float:
    """calcualtes precision for two strings

    Args:
        a_gold (str): ground truth answer
        a_pred (str): predicted answer

    Returns:
        float: precision score
    """
    num_same, pred_toks, _ = __span_comparison_helper(a_gold, a_pred)
    return __precision(num_same, pred_toks)


def f1(a_gold: str, a_pred: str) -> float:
    """calcualtes f1 for two strings

    Args:
        a_gold (str): ground truth answer
        a_pred (str): predicted answer

    Returns:
        float: f1 score
    """

    args = __span_comparison_helper(a_gold, a_pred)
    return __f1(*args)


def evaluate_model(
    model: RobertaForQuestionAnswering,
    tokenizer: Tokenizer,
    val_texts: list[str],
    val_queries: list[str],
    val_answers: list[dict],
) -> Metrics:
    """Function that evaluates a model
    on a validation dataset

    Args:
        model (RobertaForQuestionAnswering): The model to evaluate
        tokenizer (Tokenizer): tokenizer for model
        val_texts (list[str]): list of paragraphs for context
        val_queries (list[str]): list of questions
        val_answers (list[dict]): list of answer dictionaries

    Returns:
        MetricsDict: F1, recall, and precision in dictionary
    """

    # initialize pipeline
    nlp = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0)

    # list of dictionaries with scores
    results: list[Metrics] = []

    # iterate through records
    for context, question, answer in zip(val_texts, val_queries, val_answers):
        # get predicted text
        answer_pred = nlp({"question": question, "context": context})["answer"]

        # calculate meta info for metric calulations
        num_same, pred_toks, gold_toks = __span_comparison_helper(
            answer["text"], answer_pred
        )

        # clatulate scores
        p = __precision(num_same, pred_toks)
        r = __recall(num_same, gold_toks)
        f1 = __f1(num_same, pred_toks, gold_toks)

        # append metrics
        results.append({"precision": p, "recall": r, "f1": f1})

    avg_p = np.mean([_["precision"] for _ in results])
    avg_r = np.mean([_["recall"] for _ in results])
    avg_f = 2 * avg_p * avg_r / (avg_p + avg_r)

    return {"precision": avg_p, "recall": avg_r, "f1": avg_f}
