"""
Модуль расчета элементов траектории летательного аппарата на пассивном (баллистическом) участке

@Автор: <Барышев Савва>
Назначение: Расчет траектории ЛА на пассивном баллистическом участке
Версия: 1.0

Module for calculating the elements of an aircraft trajectory in the passive (ballistic) section

@author: <Savva Baryshev>
Purpose: Calculation of the aircraft trajectory in the passive ballistic section
Version: 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from Runge_Kutta4 import *
import math
from atmosphere import *
from math import sin, cos, radians, degrees
import pandas as pd




class Aircraft_Initial_Parameters:
    # Класс исходных параметров ЛА

    def __init__(self):
        # Кинематические параметры начального состояния движения ЛА:
        self.v_01 = 245 # начальная скорость летательного аппарата в первом случае, м/с
        self.v_02 = 952 # начальная скорость летательного аппарата во втором случае, м/с
        self.Theta_c0_1 = math.radians(20) # Начальный угол наклона траектории 20 ̊
        self.Theta_c0_2 = math.radians(30) # Начальный угол наклона траектории 30 ̊
        self.Theta_c0_3 = math.radians(40) # Начальный угол наклона траектории 40 ̊
        self.Theta_c0_4 = math.radians(50) # Начальный угол наклона траектории 50 ̊

        # Инерционные параметры ЛА:
        self.g_0 = 9.80665 # ускорение силы притяжения на поверхности Земли, м/(с^2)
        self.m_0 = 800 # начальная масса ЛА, кг
        self.J_z = 120 # момент инерции ЛА относительно связанной оси z, кг*м^2

        # Геометрические параметры ЛА:
        self.S_m = 0.2 # характерная площадь ЛА, м^2
        self.delta_l = 0.4 # расстояние от центра давления до центра масс ЛА, м

        # Зависимости аэродинамических коэффициентов от числа Маха:
        self.M =    np.array([0.01, 0.55, 0.80, 0.90, 1.00, 1.06, 1.10, 1.20, 1.30, 1.40, 2.00, 2.60, 3.40, 6.00, 10.0]) # массив значений числа Маха
        self.C_Xa = np.array([0.30, 0.30, 0.55, 0.70, 0.84, 0.86, 0.87, 0.83, 0.80, 0.79, 0.65, 0.55, 0.50, 0.45, 0.40]) # массив значений аэродинамического коэффициента C_Xa
        self.C_Ya = np.array([0.25, 0.25, 0.25, 0.20, 0.30, 0.31, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) # массив значений аэродинамического коэффициента C_Ya

        # Шаг интегрирования:
        self.delta_t = 0.1 # сек

        # Аэродинамические коэффициенты:
        self.C_Ya_interp = None
        self.C_Xa_interp = None

    def interp_C_XY_a(self, M):
        # функция интерполяции аэродинамических коэффициентов C_Xa и C_Ya:
        M_limit = np.clip(M, self.M[0], self.M[-1])
        self.C_Xa_interp = np.interp(M_limit, self.M, self.C_Xa)
        self.C_Ya_interp = np.interp(M_limit, self.M, self.C_Ya)
        return self.C_Xa_interp, self.C_Ya_interp

class Math_Model(Aircraft_Initial_Parameters):
    # Класс реализации математической модели расчета элементов траектории летательного аппарата на пассивном (баллистическом) участке
    #t[0] = V (скорость ЛА, м/с)
    #t[1] = Theta_c (угол наклона траеткории, рад)
    #t[2] = x (координаты центра масс ЛА в стартовой системе координат по оси X, м )
    #t[3] = y (координаты центра масс ЛА в стартовой системе координат по оси Y, м)
    #t[4] = omega_z (угловая скорость относительно связанной оси z ЛА, 1/с)
    #t[5] = theta (угол тангажа, рад)

    def __init__(self):
        super().__init__()
        self.dt_dtau = None
        self.atm = Atmosphere_GOST_4401_81 ()
        self.dtau = self.delta_t

    def alpha(self, t):
        # Метод вычисления угла атаки, ̊:
        return t[5] - t[1]

    def X_a(self, t):
        # Метод вычисления проекции аэродинамической силы на ось X скоростной системы координат:
        return self.C_Xa_interp*self.S_m*(self.atm.rho(t[3])/2)*(t[0]**2)

    def Y_a(self, t):
        # Метод вычисления проекции аэродинамической силы на ось Y скоростной системы координат:
        return self.C_Ya_interp*self.S_m*(self.atm.rho(t[3])/2)*(t[0]**2)*self.alpha(t)

    def M_z_alpha(self, t):
        # Метод вычисления градиента статического аэродинамического момента относительно связанной оси z ЛА:
        return -(self.C_Xa_interp + self.C_Ya_interp)*self.S_m*(self.atm.rho(t[3])/2)*(t[0]**2)*self.delta_l

    def a(self,t):
        # Метод вычисления скорости звука:
        return 20.046796*(self.atm.T(t[3])**(1/2))

    def Mach_number(self, t):
        # Метод вычисления числа Маха:
        return t[0]/self.a(t)

    def ODE_system(self, tau, t):
        # Метод вычисления системы ОДУ:
        self.dt_dtau = np.zeros(6) # создание numpy массива из нулей под заготовки dt_dtau
        self.dt_dtau[0] = -self.X_a(t)/self.m_0 - self.atm.g(t[3])*sin(t[1])
        self.dt_dtau[1] = self.Y_a(t)/(self.m_0*t[0]) - self.atm.g(t[3])*sin(t[1])/t[0]
        self.dt_dtau[2] = t[0]*cos(t[1])
        self.dt_dtau[3] = t[0]*sin(t[1])
        self.dt_dtau[4] = (self.M_z_alpha(t)/self.J_z)*self.alpha(t)
        self.dt_dtau[5] = t[4]
        return np.array(self.dt_dtau)

    def init_conditions_1(self):
        # функция входных параметров для Theta_c0_1:
        return np.array([0, self.Theta_c0_1, 0, 0, 0, 0])

    def init_conditions_2(self):
        # функция входных параметров для Theta_c0_2:
        return np.array([0, self.Theta_c0_2, 0, 0, 0, 0])

    def init_conditions_3(self):
        # функция входных параметров для Theta_c0_3:
        return np.array([0, self.Theta_c0_3, 0, 0, 0, 0])

    def init_conditions_4(self):
        # функция входных параметров для Theta_c0_4:
        return np.array([0, self.Theta_c0_4, 0, 0, 0, 0])


    def init_ODE_system(self, tau, t):
        # функция подготовки системы ОДУ на вход метода Runge-Kutta4
        M = self.Mach_number(t)  # вычисление числа Маха на шаге интегрирования
        self.interp_C_XY_a(M)  # вычисление соответствующих коэффициентов вычисленному числу Маха
        return self.ODE_system(tau, t)

    def record(self, t):
        # функция выходных данных для алгоритма Runge_Kutta4:
        return np.array([tau, self.m_0, t[0], self.a(t), self.Mach_number(t), self.C_Xa_interp, self.X_a(t),
                         self.alpha(t), t[1], self.dt_dtau[0], self.C_Ya_interp, self.Y_a(t), self.dt_dtau[1],
                         math.degrees(t[1]), math.degrees(t[5]), t[3], self.dt_dtau[3], t[2], self.dt_dtau[2],
                         self.M_z_alpha(t), t[4], self.dt_dtau[4], self.atm.rho(t[3]), self.atm.p(t[3])])

    def stop_conditions(self, t):
         # функция условия окончания интегрирования:
         if t[3] <= 0:
             return -1
         return 1

    def time_step(self, t):
        self.dtau = self.delta_t
        return self.dtau

class Simulation(Math_Model):
    # Класс вычислений переменных параметров по математической модели:

    def __init__(self, max_steps=100000):
        super().__init__()

        result_Theta_c0_1 = Runge_Kutta4(self.init_ODE_system, self.init_conditions_1, self.stop_conditions, self.record, self.delta_t, 0, max_steps)
        result_Theta_c0_2 = Runge_Kutta4(self.init_ODE_system, self.init_conditions_2, self.stop_conditions, self.record, self.delta_t, 0, max_steps)
        result_Theta_c0_3 = Runge_Kutta4(self.init_ODE_system, self.init_conditions_3, self.stop_conditions, self.record, self.delta_t, 0, max_steps)
        result_Theta_c0_4 = Runge_Kutta4(self.init_ODE_system, self.init_conditions_4, self.stop_conditions, self.record, self.delta_t, 0, max_steps)

        columns = ['tau', 'm_0', 'V', 'a', 'Much_Number', 'C_Xa', 'X_a', 'alpha', 'Theta_c_rad', 'dV/dtau', 'C_Ya', 'Y_a', 'dTheta_c/dtau', 'Theta_c_deg',
                   'theta', 'y', 'dy/dtau', 'x', 'dx/dttau', 'M_z_alpha', 'omega_z', 'omega_z/dtau', 'rho', 'p']

        self.df_1 = pd.DataFrame(result_Theta_c0_1, columns = columns)
        self.df_2 = pd.DataFrame(result_Theta_c0_2, columns = columns)
        self.df_3 = pd.DataFrame(result_Theta_c0_3, columns = columns)
        self.df_4 = pd.DataFrame(result_Theta_c0_4, columns = columns)

        pd.set_option('display.precision', 5)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

    def print_table(self, case=1, time_step=1.0):
        # Вывод таблицы результатов с шагом по времени

        # Выбор номера DataFrame
        if case == 1:
            df = self.df_1
            angle = 20
        elif case == 2:
            df = self.df_2
            angle = 30
        elif case == 3:
            df = self.df_3
            angle = 40
        elif case == 4:
            df = self.df_4
            angle = 50

        # Фильтрация данных (только положительная высота и скорость):
        df_filtered = df[(df['y'] > 0) & (df['V'] > 0)].copy()

        # Округление времени до целой секунды:
        df_filtered['time_rounded'] = np.round(df_filtered['tau'] / time_step) * time_step

        # Группировка по округленному времени и выбор строки в каждой группе
        df_table = df_filtered.groupby('time_rounded').first().reset_index()

        # Удаление вспомогательного столбца
        df_table = df_table.drop(columns=['time_rounded'])

        # Выбор нужных столбцов для отображения
        display_columns = ['tau', 'm_0', 'V', 'a', 'Much_Number', 'C_Xa', 'X_a', 'alpha', 'Theta_c_rad', 'dV/dtau', 'C_Ya', 'Y_a', 'dTheta_c/dtau', 'Theta_c_deg',
                   'theta', 'y', 'dy/dtau', 'x', 'dx/dtau', 'M_z_alpha', 'omega_z', 'omega_z/dtau', 'rho', 'p']

        # Создание таблицы вывода
        table = df_table[display_columns].copy()

        # Название столбца - параметр и его размерность
        table = table.columns = [
            't, с',
            'm_0, кг',
            'V, м/с',
            'a, м/с'
            'M',
            'C_Xa',
            'X_a, H',
            'alpha, град',
            'Θ_с, рад',
            'dV/dt, м/с^2',
            'C_Ya_alpha',
            'Y_a, H',
            'dΘ_с/dt, с^-1'
            'Θ_с, град',
            'θ, град',
            'y, м',
            'dy/dt, м/с',
            'x, м',
            'dx/dt, м/с',
            'M_z_alpha, кг*м^2/с^2',
            'ω_z, c^-1',
            'ω_z/dt, c^-2',
            'ρ, кг/м³',
            'p, Па'
        ]

        # Вывод заголовка
        print("\n" + "=" * 120)
        print(f"РЕЗУЛЬТАТЫ РАСЧЕТА ТРАЕКТОРИИ (начальный угол Θ_с0 = {angle}°)")
        print("=" * 120)
        print(f"\nШаг вывода: {time_step} секунда(ы)")
        print(f"Всего точек в таблице: {len(table)}")
        print("\n")

        # Вывод таблицы
        print(table.to_string(index=False))

        return table

    def print_all_tables(self, time_step=1.0):
        # Вывод таблиц для всех четырех углов:
        for case in range(1, 5):
            self.print_table(case, time_step)
            print("\n" + "-" * 120 + "\n")

    def save_table_to_csv(self, case=1, filename=None, time_step=1.0):
        # Сохранение таблицы в CSV файл:
        table = self.print_table(case, time_step)

        if filename is None:
            angle = [20, 30, 40, 50][case - 1]
            filename = f"Результаты расчета для угла Θ_с0= {angle}°.csv"

        table.to_csv(filename, index=False)
        print(f"\nТаблица сохранена в файл: {filename}")

    def get_table_at_time(self, case=1, target_time=10.0):
        #Получение параметров в конкретный момент времени:
        if case == 1:
            df = self.df_1
        elif case == 2:
            df = self.df_2
        elif case == 3:
            df = self.df_3
        else:
            df = self.df_4

        # Поиск ближайшей строки к целевому времени
        idx = (df['tau'] - target_time).abs().idxmin()
        result = df.loc[idx]

        print(f"\nПараметры в момент t = {target_time} с:")
        print("-" * 50)
        print(f"  Скорость V:              {result['V']:.2f} м/с")
        print(f"  Высота y:                {result['y']:.2f} м")
        print(f"  Дальность x:             {result['x']:.2f} м")
        print(f"  Угол траектории Θ:       {result['Theta_c_deg']:.3f}°")
        print(f"  Число Маха M:            {result['M']:.4f}")
        print(f"  Угол атаки α:            {result['alpha']:.4f} рад")
        print(f"  Плотность ρ:             {result['rho']:.5f} кг/м³")
        print(f"  Давление p:              {result['p']:.1f} Па")

        return result
