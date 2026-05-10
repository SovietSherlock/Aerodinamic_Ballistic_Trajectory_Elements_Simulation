import numpy as np
import matplotlib.pyplot as plt
from math import *

def Runge_Kutta4(function, innit_condition_x, innit_condition_y, end_condition_x, time_step):
    # функция, реализующая метод численного интегрирования системы обыкновенных дифференциальных уравнений, метод Рунге-Кутты 4-ого порядка
    # function - функция математической модели, принимаемая на вход алгоритмом Runge_Kutta4
    # innit_condition_x - начальные условия независимой переменной
    # innit_condition_y - начальные условия зависимой переменной
    # end_condition_x - конечное условие для независимой переменной
    # time_step - шаг интегрирования по независимой переменной (в нашем случае, шаг по времени)

    x = []
    y = []
    x = x_0
    y = y_0


    while x<end_condition_x - 1e-12:
        k1 = function(x, y)
        k2 = function(x + time_step/2, y + k1*time_step/2)
        k3 = function(x + time_step/2, y + k2*time_step/2)
        k4 = function(x + time_step, y + k3*time_step)

        y = y + (time_step/6)*(k1 + 2*k2 + 2*k3 + k4)
        x = x + time_step

        x.append(x)
        y.append(y)

    return x, y


x_vals, y_vals = Runge_Kutta4(f, 0, 1, 2, 0.1)
plt.plot(x_vals, y_vals)
