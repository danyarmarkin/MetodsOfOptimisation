from math import sqrt, sin
from typing import Callable


def add(a: tuple, b: tuple) -> tuple:
    assert len(a) == len(b)
    return tuple(a[i] + b[i] for i in range(len(a)))


def neg(a: tuple) -> tuple:
    return tuple(map(lambda x: -x, a))


def sub(a: tuple, b: tuple) -> tuple:
    return add(a, neg(b))


def mul(a: tuple, b: float) -> tuple:
    return tuple(map(lambda x: x * b, a))


def norm(v: tuple) -> float:
    return sqrt(sum(map(lambda x: x * x, v)))


def approximate(f, x: tuple, delta: tuple) -> float:
    return (f(add(x, delta)) - f(x)) / norm(delta)


def symmetric(f, x: tuple, delta: tuple) -> float:
    return (f(add(x, delta)) - f(sub(x, delta))) / (2 * norm(delta))


def grad(f, x: tuple, delta=1e-10, der=symmetric) -> tuple:
    r = []
    for i in range(len(x)):
        d = list(0 for _ in range(len(x)))
        d[i] = delta
        r.append(der(f, x, tuple(d)))
    try:
        f.grad()
    except AttributeError:
        pass
    return tuple(r)


def normalize(x: tuple) -> tuple:
    n = norm(x)
    return tuple(i / n for i in x)


def tf(f_) -> Callable[[tuple], float]:
    return lambda t: f_(*t)
