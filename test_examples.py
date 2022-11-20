import math
import numpy as np

def test_sqrt():
    num = 36
    assert math.sqrt(num) == 6, "Sqrt failed"


def test_square():
    num = 5
    assert num**2 == 25, "Square failed"


def test_equality():
    assert 20 == 20, "not equal"