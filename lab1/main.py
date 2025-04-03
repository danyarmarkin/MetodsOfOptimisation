import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from gradient import gradient_descent_with_lr, gradient_descent_with_gss, gradient_descent_with_ts
import learning_rate
from func import *

FIRST = lambda x: x[0]
SECOND = lambda x: x[1]

WITH_LR = lambda lr: lambda func: gradient_descent_with_lr(func, 2, lr=lr, eps=0.01, x=(-0, -2))
CONSTANT = WITH_LR(learning_rate.constant(0.01))
EXPONENTIAL = WITH_LR(learning_rate.exponential(0.2, 0.01))
TIME_BASED = WITH_LR(learning_rate.time_based(0.1, 0.01))

TS_GRAD = lambda func: gradient_descent_with_ts(func, 2, eps=0.0001, x=(-0, -2))
GSS_GRAD = lambda func: gradient_descent_with_gss(func, 2, eps=0.0001, x=(-0, -2))

SCIPY_CG = lambda func: minimize(func, np.array([-0, -2]), method='CG')

cases = [
    (EXPONENTIAL, "exponential", "green"),
    (TIME_BASED, "time-based", "orange"),
    (CONSTANT, "constant", "pink"),
    (TS_GRAD, "Ternary Search", "red"),
    (GSS_GRAD, "GS Search", "brown"),
]

funcs = [
    (lambda x, y: x ** 2 - 2 * x * y + 4 * y ** 2 - 19 * y + 10 * x - 5, "func-0"),
    (lambda x, y: x ** 4 - 4 * x * x + y * y + 3 * x - 4 * y, "func-1"),
    (lambda x, y: (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2, "func-2"),
]

for func, f_name in funcs:
    for m, m_name, color in cases:
        print(f"Running case {m_name}({f_name})...")

        min_x = 1e10
        min_y = 1e10
        max_x = -1e10
        max_y = -1e10

        f = Func(func)
        coords = list(m(f))
        x_points = list(map(FIRST, coords))
        y_points = list(map(SECOND, coords))

        max_x = max(max_x, max(x_points))
        min_x = min(min_x, min(x_points))
        max_y = max(max_y, max(y_points))
        min_y = min(min_y, min(y_points))
        print("points count:", len(coords))
        print("f calls count:", f.calls)
        print()
        x = np.linspace(min_x - 1, max_x + 1, 100)
        y = np.linspace(min_y - 1, max_y + 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = f((X, Y))
        plt.figure(figsize=(10, 6))
        plt.plot(x_points, y_points, color=color, label=f"{m_name}({f_name})")
        plt.plot([], [], ' ', label=f"Points count: {len(coords)}")
        plt.plot([], [], ' ', label=f"Grad count: {f.grads}")
        plt.plot([], [], ' ', label=f"Calls count: {f.calls}")
        plt.plot([], [], ' ', label=f"x_0 = (0, -2)")
        plt.plot([], [], ' ', label=f"x_min = ({x_points[-1]:.4}, {y_points[-1]:.4})")

        plt.contour(X, Y, Z, levels=15, colors='blue', alpha=0.5)

        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"{m_name}({f_name})")
        plt.legend()

        # plt.show()
        # plt.clf()
        plt.savefig(f"imgs/{m_name}({f_name}).png")


for func, f_name in funcs:
    print(f"Running case minimize({f_name})...")
    f = Func(func)
    x = SCIPY_CG(f)
    print(x)
    print("func calls: ", f.calls)
    print()