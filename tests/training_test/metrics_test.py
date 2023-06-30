# BEGIN: abpxx6d04wxr
# This file is used to test the metrics.py file in the training folder.


from training.metrics import (
    f1,
    precision,
    recall,
)


def test_precision():
    # Test case 1
    assert precision("hello", "hello") == 1.0

    # Test case 2
    assert precision("hello", "world") == 0.0


def test_recall():
    # Test case 1
    assert recall("hello", "hello") == 1.0

    # Test case 2
    assert recall("hello", "world") == 0.0


def test_f1():
    # Test case 1
    assert f1("hello", "hello") == 0.999999995

    # Test case 2
    assert f1("hello", "world") == 0.0


# END: ed8c6549bwf9
