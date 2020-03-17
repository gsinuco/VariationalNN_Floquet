# -*- coding: utf-8 -*-

import pytest
from variationalnn_floquet.skeleton import fib

__author__ = "German Sinuco"
__copyright__ = "German Sinuco"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
