import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white

def fast_white_test(x, y):
    """
    Быстрый тест Уайта для двух массивов x и y
    """
    # Добавляем константу к x
    X = sm.add_constant(x)

    # Оцениваем модель
    model = sm.OLS(y, X).fit()

    # Тест Уайта
    lm_stat, p_value, f_stat, f_p_value = het_white(model.resid, X)

    return {
        'lm_statistic': lm_stat,
        'p_value': p_value,
        'reject_h0': p_value < 0.05,
        'heteroscedasticity': p_value < 0.05
    }

# Пример использования
if __name__ == "__main__":
    # Ваши данные
    x = np.array([40, 30, 30, 25, 50, 60, 65, 10, 15, 20, 55, 40, 35, 30])
    y = np.array([1000, 1500, 1200, 1800, 800, 1000, 500, 3000, 2500, 2000, 800, 1500, 2000, 2000])

    # Быстрый тест
    result = fast_white_test(x, y)
    print("Результаты теста Уайта:")
    print(f"LM-статистика: {result['lm_statistic']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Гетероскедастичность: {'ДА' if result['heteroscedasticity'] else 'НЕТ'}")