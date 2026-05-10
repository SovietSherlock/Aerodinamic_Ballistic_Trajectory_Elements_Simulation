"""
Модуль численного решения обыкновенных дифференциальных уравнений
методом Рунге-Кутты 4-го порядка

@Автор: <Барышев Савва>
Назначение: Расчет траектории ЛА на пассивном баллистическом участке
Версия: 1.0

Module for numerical technique for solving Ordinary Differential Equations (ODEs)
using 4th Order Runge-Kutta method

@author: <Savva Baryshev>
Purpose: Calculation of the aircraft trajectory in the passive ballistic section
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def Runge_Kutta4(function, init_condition_x, init_condition_y, end_condition_x, time_step):
    # функция, реализующая метод численного интегрирования системы обыкновенных дифференциальных уравнений, метод Рунге-Кутты 4-го порядка
    # function - функция математической модели, принимаемая на вход алгоритмом Runge_Kutta4
    # innit_condition_x - начальные условия независимой переменной
    # innit_condition_y - начальные условия зависимой переменной
    # end_condition_x - конечное условие для независимой переменной
    # time_step - шаг интегрирования по независимой переменной (в нашем случае, шаг по времени)

    x_vals = [init_condition_x]
    y_vals = [init_condition_y]
    x = init_condition_x
    y = init_condition_y


    while x<end_condition_x - 1e-12:
        k1 = function(x, y)
        k2 = function(x + time_step/2, y + k1*time_step/2)
        k3 = function(x + time_step/2, y + k2*time_step/2)
        k4 = function(x + time_step, y + k3*time_step)

        y = y + (time_step/6)*(k1 + 2*k2 + 2*k3 + k4)
        x = x + time_step

        x_vals.append(x)
        y_vals.append(y)

    return x_vals, y_vals


def f(x, y):
    return y

x_vals, y_vals = Runge_Kutta4(f, 0, 1, 2, 0.1)


print(f"x = {x_vals[-1]:.2f}, y = {y_vals[-1]:.10f}")
print(f"Точное значение e^2 = {exp(2):.10f}")
print(f"Ошибка = {abs(y_vals[-1] - exp(2)):.2e}")

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='Рунге-Кутта 4 (числ. решение)')
plt.plot(x_vals, [exp(x) for x in x_vals], 'r--', linewidth=1, label='Точное решение: e^x')
plt.xlabel('x (независимая переменная)')
plt.ylabel('y (зависимая переменная)')
plt.title('Решение ОДУ методом Рунге-Кутты 4-го порядка')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()