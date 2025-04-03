from calculus import *


def gradient_descent(func, n: int, lr: Callable[[int], float], i_lim=100000, eps=0.000001, x=None) -> tuple:
    if not x:
        x = tuple(0 for _ in range(n))

    yield x

    for i in range(i_lim):
        g = grad(func, x)
        if norm(g) < eps:
            print(" < eps")
            break
        g = normalize(g)
        x = sub(x, mul(g, lr(i)))
        yield x


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


def gradient_descent_with_gss(func, n: int, lr: Callable[[int], float], i_lim=100000, eps=1e-6, x=None):
    if not x:
        x = tuple(0 for _ in range(n))

    yield x

    for i in range(i_lim):
        g = grad(func, x)
        if norm(g) < eps:
            break
        g = normalize(g)

        def f_alpha(alpha):
            return func(sub(x, mul(g, alpha)))

        alpha_opt = golden_section_search(f_alpha, 0, 5)
        x = sub(x, mul(g, alpha_opt))
        yield x
