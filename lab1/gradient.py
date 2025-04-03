from calculus import *


def golden_section_search(func: Callable[[float], float], a: float, b: float, tol=1e-5):
    phi = (1 + sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1, f2 = func(x1), func(x2)

    while abs(b - a) > tol:
        if f1 < f2:
            b, f2 = x2, f1
            x2 = x1
            x1 = a + resphi * (b - a)
            f1 = func(x1)
        else:
            a, f1 = x1, f2
            x1 = x2
            x2 = b - resphi * (b - a)
            f2 = func(x2)

    return (a + b) / 2


def ternary_search(func: Callable[[float], float], a: float, b: float, tol=1e-5):
    while abs(b - a) > tol:
        left_third = a + (b - a) / 3
        right_third = b - (b - a) / 3
        if func(left_third) < func(right_third):
            b = right_third
        else:
            a = left_third
    return (a + b) / 2


def gradient_descent(func, n, step, **kwargs):
    lim = kwargs.get("lim", 100000)
    eps = kwargs.get("eps", 0.00001)
    x = kwargs.get("x", None)
    if not x:
        x = tuple(0 for _ in range(n))

    yield x

    for i in range(lim):
        g = grad(func, x)
        if norm(g) < eps:
            break
        g = normalize(g)
        x = sub(x, mul(g, step(i, x, g)))
        yield x


def gradient_descent_with_lr(func, n: int, **kwargs):
    lr = kwargs.get("lr")
    return gradient_descent(func, n, lambda i, x, g: lr(i), **kwargs)


def gradient_descent_with_us(func, n: int, method, **kwargs):
    hyper_a = kwargs.get("hyper_a", 0)
    hyper_b = kwargs.get("hyper_b", 1)
    step = lambda i, x, g: method(
        lambda alpha: func(sub(x, mul(g, alpha))), hyper_a, hyper_b)
    return gradient_descent(func, n, step, **kwargs)


def gradient_descent_with_gss(func, n: int, **kwargs):
    return gradient_descent_with_us(func, n, golden_section_search, **kwargs)


def gradient_descent_with_ts(func, n: int, **kwargs):
    return gradient_descent_with_us(func, n, ternary_search, **kwargs)
