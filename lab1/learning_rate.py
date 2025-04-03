from math import exp
from typing import Callable


def constant(c: float) -> Callable[[int], float]:
    return lambda _: c


def time_based(nu: float, k: float) -> Callable[[int], float]:
    return lambda i: nu / (1 + k * i)


def exponential(nu: float, k: float) -> Callable[[int], float]:
    return lambda i: nu * exp(-i * k)
