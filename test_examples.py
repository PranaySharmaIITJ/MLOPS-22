import math
import numpy as np

def test_sqrt():
    num = 25
    assert math.sqrt(num) == 5, "Sqrt failed"


def test_square():
    num = 4
    assert num**2 == 16, "Square failed"


def test_equality():
    assert 10 == 10, "not equal"

