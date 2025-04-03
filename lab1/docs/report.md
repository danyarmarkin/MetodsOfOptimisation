# Отчет по лабораторной работе #1

## Функции для исследования

### `func-0`
```math
f(x, y) = x ** 2 - 2 * x * y + 4 * y ** 2 - 19 * y + 10 * x - 5
```

### `func-1`
```math
f(x, y) = x ** 4 - 4 * x * x + y * y + 3 * x - 4 * y
```

### `func-2`
```math
f(x, y) = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
```

## Реализованные методы

### Gradient descent with learning rate

Метод содержит критерий остановки: 
изменение значения становится меньше `eps`, который задается как гиперпараметр.

В моей реализации `eps=1e-2` для стратегий выборов и 
`eps=1e-4` для одномерных поисков

Так же сделан предохранитель для количества итераций - по умолчанию 
`lim=1e5`

Для одномерных поисков есть критерий остановки - `tol` - аналог `eps`. По дефолту равен `1e-5`.

#### Constant learning rate
    
Шаг представляет собой константу. Значение шага выбрано `1e-2`
![](../imgs/constant(func-0).png)
![](../imgs/constant(func-1).png)
![](../imgs/constant(func-2).png)

#### Exponential learning rate
    
Шаг задается формулой `nu / (1 + k * i)`, где `(nu, k)` - гиперпараметры,
а `i` - номер итерации. 

В приведенных примерах `nu, k = 0.2, 0.01`
![](../imgs/exponential(func-0).png)
![](../imgs/exponential(func-1).png)
![](../imgs/exponential(func-2).png)

#### Time-based learning rate
Шаг задается формулой `nu * exp(-i * k)`, где `(nu, k)` - гиперпараметры,
а `i` - номер итерации.

В приведенных примерах `nu, k = 0.1, 0.01`
![](../imgs/time-based(func-0).png)
![](../imgs/time-based(func-1).png)
![](../imgs/time-based(func-2).png)

### Gradient descent with unary search function

#### Ternary search

Выбор шага определяется при помощи тернарного поиска.

![](../imgs/Ternary%20Search(func-0).png)
![](../imgs/Ternary%20Search(func-1).png)
![](../imgs/Ternary%20Search(func-2).png)

#### Golden section search

Выбор шага определяется при помощи GSS

![](../imgs/GS%20Search(func-0).png)
![](../imgs/GS%20Search(func-1).png)
![](../imgs/GS%20Search(func-2).png)



## `scipy.optimize`

### `minimize`

Библиотечный метод градиентного спуска
```python
from scipy.optimize import minimize

SCIPY_CG = lambda func: minimize(func, np.array([-0, -2]), method='CG')
```

```text
Running case minimize(func-0)...
 message: Optimization terminated successfully.
 success: True
  status: 0
     fun: -36.74999999999745
       x: [-3.500e+00  1.500e+00]
     nit: 2
     jac: [-2.384e-06  6.199e-06]
    nfev: 15
    njev: 5
func calls:  15
```

```text
Running case minimize(func-1)...
 message: Optimization terminated successfully.
 success: True
  status: 0
     fun: -12.494017550905998
       x: [-1.574e+00  2.000e+00]
     nit: 15
     jac: [-4.053e-06 -1.907e-06]
    nfev: 81
    njev: 27
func calls:  81
```

```text
Running case minimize(func-2)...
 message: Optimization terminated successfully.
 success: True
  status: 0
     fun: 1.773886799115342e-15
       x: [ 3.584e+00 -1.848e+00]
     nit: 10
     jac: [ 4.656e-07  4.715e-07]
    nfev: 75
    njev: 25
func calls:  75
```
