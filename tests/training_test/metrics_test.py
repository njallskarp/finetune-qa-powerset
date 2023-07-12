# BEGIN: abpxx6d04wxr
# This file is used to test the metrics.py file in the training folder.

from training.metrics import (
    f1,
    precision,
    recall,
)


def test_precision():
    """
    The equation for precision is;
        if len(pred_toks) == 0:
            return 0
        return 1.0 * num_same / len(pred_toks)

    where;
        num_same (int): number of similar tokens in grount truth and predicted string
        gold_toks (list[str]): list of gold tokens
    """

    # Test case "Correct prediction"
    assert precision("18 metra", "18 metra") == 1.0

    # Test case "Incorrect prediction"
    assert precision("18 metra", "Munchen") == 0.0


def test_precision_partial():
    # Test case "Partial correct prediction, first token"
    assert precision("18 metra", "18") == 1.0

    # Test case "Partial correct prediction, last token"
    assert precision("18 metra", "metra") == 1.0

    # Test case "Partial correct prediction, overflow"
    assert precision("18 metra", "18 metra 18 metra") == 0.5

    # Test case "Partial correct prediction with punctuation"
    assert precision("18 metra", "18 metra.") == 1.0

    # Test case "Partial correct prediction case insensitive"
    assert precision("18 metra", "18 METRA") == 1.0

    # Test case "Partial correct prediction joined"
    assert precision("18 metra", "18metra") == 0.0


def test_precision_empty():
    # Test case "Empty prediction"
    assert precision("18 metra", "") == 0.0

    # Test case "Empty ground truth"
    assert precision("", "18 metra") == 0.0

    # Test case "Empty prediction and ground truth"
    assert precision("", "") == 0.0


def test_recall_partial():
    # Test case "Partial correct prediction, first token"
    assert recall("18 metra", "18") == 0.5

    # Test case "Partial correct prediction, last token"
    assert recall("18 metra", "metra") == 0.5

    # Test case "Partial correct prediction, overflow"
    assert recall("18 metra", "18 metra 18 metra") == 1.0

    # Test case "Partial correct prediction with punctuation"
    assert recall("18 metra", "18 metra.") == 1.0

    # Test case "Partial correct prediction case insensitive"
    assert recall("18 metra", "18 METRA") == 1.0

    # Test case "Partial correct prediction joined"
    assert recall("18 metra", "18metra") == 0.0


def test_recall_empty():
    # Test case "Empty prediction"
    assert recall("18 metra", "") == 0.0

    # Test case "Empty ground truth"
    assert recall("", "18 metra") == 0.0

    # Test case "Empty prediction and ground truth"
    assert recall("", "") == 0.0


def test_f1():
    """
    The equation for f1 is;
        p = precision(gold_toks, pred_toks)
        r = recall(gold_toks, pred_toks)
        if p + r == 0:
            return 0
        return 2.0 * p * r / (p + r)

    where;
        gold_toks (list[str]): list of gold tokens
        pred_toks (list[str]): list of predicted tokens
    """

    # Test case "Correct prediction"
    assert f1("18 metra", "18 metra") == 0.999999995

    # Test case "Incorrect prediction"
    assert f1("18 metra", "Munchen") == 0.0


def test_f1_partial():
    # Test case "Partial correct prediction, first token"
    assert f1("18 metra", "18") == 0.6666666622222223

    # Test case "Partial correct prediction, last token"
    assert f1("18 metra", "metra") == 0.6666666622222223

    # Test case "Partial correct prediction, overflow"
    assert f1("18 metra", "18 metra 18 metra") == 0.6666666622222223

    # Test case "Partial correct prediction with punctuation"
    assert f1("18 metra", "18 metra.") == 0.999999995

    # Test case "Partial correct prediction case insensitive"
    assert f1("18 metra", "18 METRA") == 0.999999995

    # Test case "Partial correct prediction joined"
    assert f1("18 metra", "18metra") == 0.0


def test_f1_empty():
    # Test case "Empty prediction"
    assert f1("18 metra", "") == 0.0

    # Test case "Empty ground truth"
    assert f1("", "18 metra") == 0.0

    # Test case "Empty prediction and ground truth"
    assert f1("", "") == 1.0


# END: ed8c6549bwf9
