import matplotlib.pyplot as plt
import numpy as np

from gradient import gradient_descent_with_lr, gradient_descent_with_gss, gradient_descent_with_ts
from calculus import *
import learning_rate


def f(x, y):
    return x ** 4 - 4 * x * x + y * y + 3 * x - 4 * y
    # return (1 - x)**2 + 100 * (y - x**2)**2
    # return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


FIRST = lambda x: x[0]
SECOND = lambda x: x[1]

cases = [
    (learning_rate.exponential(0.2, 0.01), "exponential", "green"),
    # (learning_rate.time_based(0.001, 0.001), "time-based", "brown"),
    # (learning_rate.constant(0.01), "constant", "red")
]


min_x = 1e10
min_y = 1e10
max_x = -1e10
max_y = -1e10

xx = []
yy = []
for case in cases:
    print("start", case[1])
    coords = list(gradient_descent_with_ts(tf(f), 2, lr=case[0], eps=0.0001, x=(-0, -2)))
    x_points = list(map(FIRST, coords))
    y_points = list(map(SECOND, coords))

    xx.append(x_points)
    yy.append(y_points)

    max_x = max(max_x, max(x_points))
    min_x = min(min_x, min(x_points))
    max_y = max(max_y, max(y_points))
    min_y = min(min_y, min(y_points))
    print("points count:", len(coords))


# Создание сетки для линий уровня
x = np.linspace(min_x - 1, max_x + 1, 100)
y = np.linspace(min_y - 1, max_y + 1, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Создание графика
plt.figure(figsize=(10, 6))

for x_points, y_points, case in zip(xx, yy, cases):
    plt.plot(x_points, y_points, color=case[2], label=case[1])

# Рисуем линии уровня (красный цвет)
plt.contour(X, Y, Z, levels=15, colors='blue', alpha=0.5)

# Настройки графика
plt.grid(True)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('График точек и линий уровня')
plt.legend()

# Показать график
plt.show()