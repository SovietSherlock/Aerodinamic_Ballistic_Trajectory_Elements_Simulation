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
        self.dt_dtau[1] = self.Y_a(t)/(self.m_0*t[0]) - self.atm.g(t[3])*cos(t[1])/t[0]
        self.dt_dtau[2] = t[0]*cos(t[1])
        self.dt_dtau[3] = t[0]*sin(t[1])
        self.dt_dtau[4] = (self.M_z_alpha(t)/self.J_z)*self.alpha(t)
        self.dt_dtau[5] = t[4]
        return np.array(self.dt_dtau)

    def init_conditions_1(self):
        # функция входных параметров для Theta_c0_1:
        return np.array([self.v_01, self.Theta_c0_1, 0, 0.001, 0, 0])

    def init_conditions_2(self):
        # функция входных параметров для Theta_c0_2:
        return np.array([self.v_01, self.Theta_c0_2, 0, 0.001, 0, 0])

    def init_conditions_3(self):
        # функция входных параметров для Theta_c0_3:
        return np.array([self.v_01, self.Theta_c0_3, 0, 0.001, 0, 0])

    def init_conditions_4(self):
        # функция входных параметров для Theta_c0_4:
        return np.array([self.v_01, self.Theta_c0_4, 0, 0.001, 0, 0])


    def init_ODE_system(self, tau, t):
        # функция подготовки системы ОДУ на вход метода Runge-Kutta4
        M = self.Mach_number(t)  # вычисление числа Маха на шаге интегрирования
        self.interp_C_XY_a(M)  # вычисление соответствующих коэффициентов вычисленному числу Маха
        return self.ODE_system(tau, t)

    def record(self, tau, t):
        # функция выходных данных для алгоритма Runge_Kutta4:
        M = self.Mach_number(t)
        self.interp_C_XY_a(M)
        if self.dt_dtau is None:
            dV_dt = 0.0
            dTheta_dt = 0.0
            dx_dt = 0.0
            dy_dt = 0.0
            domega_dt = 0.0
        else:
            dV_dt = self.dt_dtau[0]
            dTheta_dt = self.dt_dtau[1]
            dx_dt = self.dt_dtau[2]
            dy_dt = self.dt_dtau[3]
            domega_dt = self.dt_dtau[4]
        return np.array([tau, self.m_0, t[0], self.a(t), self.Mach_number(t), self.C_Xa_interp, self.X_a(t),
                         self.alpha(t), t[1], dV_dt, self.C_Ya_interp, self.Y_a(t), dTheta_dt,
                         math.degrees(t[1]), math.degrees(t[5]), t[3], dy_dt, t[2], dx_dt,
                         self.M_z_alpha(t), t[4], domega_dt, self.atm.rho(t[3]), self.atm.p(t[3])])

    def stop_conditions(self, tau, t):
         # функция условия окончания интегрирования:
         if t[3] < -1:
             return -1
         return 1

    def time_step(self, t):
        self.dtau = self.delta_t
        return self.dtau

class Simulation(Math_Model):
    # Класс вычислений переменных параметров по математической модели:

    def __init__(self, max_steps=10):
        super().__init__()

        result_Theta_c0_1 = Runge_Kutta4(self.init_ODE_system, self.init_conditions_1(), self.stop_conditions, self.record, self.delta_t, 0, max_steps)
        result_Theta_c0_2 = Runge_Kutta4(self.init_ODE_system, self.init_conditions_2(), self.stop_conditions, self.record, self.delta_t, 0, max_steps)
        result_Theta_c0_3 = Runge_Kutta4(self.init_ODE_system, self.init_conditions_3(), self.stop_conditions, self.record, self.delta_t, 0, max_steps)
        result_Theta_c0_4 = Runge_Kutta4(self.init_ODE_system, self.init_conditions_4(), self.stop_conditions, self.record, self.delta_t, 0, max_steps)

        columns = ['tau', 'm_0', 'V', 'a', 'Much_Number', 'C_Xa', 'X_a', 'alpha', 'Theta_c_rad', 'dV_dtau', 'C_Ya', 'Y_a', 'dTheta_c_dtau', 'Theta_c_deg',
                   'theta', 'y', 'dy_dtau', 'x', 'dx_dtau', 'M_z_alpha', 'omega_z', 'omega_z_dtau', 'rho', 'p']

        self.df_1 = pd.DataFrame(result_Theta_c0_1[:, 7:31], columns=columns)
        self.df_2 = pd.DataFrame(result_Theta_c0_2[:, 7:31], columns=columns)
        self.df_3 = pd.DataFrame(result_Theta_c0_3[:, 7:31], columns=columns)
        self.df_4 = pd.DataFrame(result_Theta_c0_4[:, 7:31], columns=columns)

        pd.set_option('display.precision', 5)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        # Список для сбора всех данных для CSV
        all_data = []

        # Данные для вывода: (DataFrame, угол)
        cases = [(self.df_1, 20), (self.df_2, 30), (self.df_3, 40), (self.df_4, 50)]

        for df, angle in cases:
            df_filtered = df.query('V>=0 and y>=-100').reset_index(drop=True)

            # Заголовок таблицы
            print("\n" + "=" * 130)
            print(f"РЕЗУЛЬТАТЫ РАСЧЕТА ТРАЕКТОРИИ (начальный угол Θ₀ = {angle}°)")
            print("=" * 130)
            print(f"{'t,с':>6} | {'V,м/с':>10} | {'Θ,град':>10} | {'x,м':>12} | {'y,м':>10} | {'M':>8} | {'α,рад':>10} | {'C_Xa':>8} | {'C_Ya':>8} | {'ρ,кг/м³':>12} | {'p,Па':>10}")
            print("-" * 130)

            for k in range(0, len(df_filtered), 10):
                if len(df_filtered) > k:
                    row = df_filtered.iloc[k]

                    tau = f"{row['tau']:.2f}"
                    V_val = f"{row['V']:.2f}"
                    Theta_c_deg_val = f"{row['Theta_c_deg']:.3f}"
                    x_val = f"{row['x']:.2f}"
                    y_val = f"{row['y']:.2f}"
                    M_val = f"{row['Much_Number']:.4f}"
                    alpha_val = f"{row['alpha']:.5f}"
                    C_Xa_val = f"{row['C_Xa']:.4f}"
                    C_Ya_val = f"{row['C_Ya']:.4f}"
                    rho_val = f"{row['rho']:.5f}"
                    p_val = f"{row['p']:.1f}"

                    # Вывод в консоль
                    print(f"{tau:>6} | {V_val:>10} | {Theta_c_deg_val:>10} | {x_val:>12} | {y_val:>10} | {M_val:>8} | {alpha_val:>10} | {C_Xa_val:>8} | {C_Ya_val:>8} | {rho_val:>12} | {p_val:>10}")

                    # Сохраняем данные для CSV
                    all_data.append({
                        'Угол, град': angle,
                        't, с': float(tau),
                        'V, м/с': float(V_val),
                        'Θ, град': float(Theta_c_deg_val),
                        'x, м': float(x_val),
                        'y, м': float(y_val),
                        'M': float(M_val),
                        'α, рад': float(alpha_val),
                        'C_Xa': float(C_Xa_val),
                        'C_Ya': float(C_Ya_val),
                        'ρ, кг/м³': float(rho_val),
                        'p, Па': float(p_val)
                    })

        # Сохранение в CSV файл (с добавлением точек падения)
        all_data_with_impact = all_data.copy()  # копируем существующие данные

        # Собираем точки падения и добавляем их в основной список
        for df, angle in cases:
            df_all = df.query('V>=0').reset_index(drop=True)

            for i in range(1, len(df_all)):
                if df_all.iloc[i - 1]['y'] > 0 and df_all.iloc[i]['y'] <= 0:
                    y1 = df_all.iloc[i - 1]['y']
                    y2 = df_all.iloc[i]['y']
                    t1 = df_all.iloc[i - 1]['tau']
                    t2 = df_all.iloc[i]['tau']

                    if t2 != t1:
                        t_impact = t1 + (0 - y1) * (t2 - t1) / (y2 - y1)
                        frac = (t_impact - t1) / (t2 - t1)
                    else:
                        t_impact = t1
                        frac = 0

                    V_impact = df_all.iloc[i - 1]['V'] + frac * (df_all.iloc[i]['V'] - df_all.iloc[i - 1]['V'])
                    Theta_impact = df_all.iloc[i - 1]['Theta_c_deg'] + frac * (
                                df_all.iloc[i]['Theta_c_deg'] - df_all.iloc[i - 1]['Theta_c_deg'])
                    x_impact = df_all.iloc[i - 1]['x'] + frac * (df_all.iloc[i]['x'] - df_all.iloc[i - 1]['x'])
                    M_impact = df_all.iloc[i - 1]['Much_Number'] + frac * (
                                df_all.iloc[i]['Much_Number'] - df_all.iloc[i - 1]['Much_Number'])
                    alpha_impact = df_all.iloc[i - 1]['alpha'] + frac * (
                                df_all.iloc[i]['alpha'] - df_all.iloc[i - 1]['alpha'])
                    Cx_impact = df_all.iloc[i - 1]['C_Xa'] + frac * (
                                df_all.iloc[i]['C_Xa'] - df_all.iloc[i - 1]['C_Xa'])
                    Cy_impact = df_all.iloc[i - 1]['C_Ya'] + frac * (
                                df_all.iloc[i]['C_Ya'] - df_all.iloc[i - 1]['C_Ya'])
                    rho_impact = df_all.iloc[i - 1]['rho'] + frac * (df_all.iloc[i]['rho'] - df_all.iloc[i - 1]['rho'])
                    p_impact = df_all.iloc[i - 1]['p'] + frac * (df_all.iloc[i]['p'] - df_all.iloc[i - 1]['p'])

                    # Добавляем точку падения в список
                    all_data_with_impact.append({
                        'Угол, град': angle,
                        't, с': t_impact,
                        'V, м/с': V_impact,
                        'Θ, град': Theta_impact,
                        'x, м': x_impact,
                        'y, м': 0.0,
                        'M': M_impact,
                        'α, рад': alpha_impact,
                        'C_Xa': Cx_impact,
                        'C_Ya': Cy_impact,
                        'ρ, кг/м³': rho_impact,
                        'p, Па': p_impact
                    })
                    break

        # Сохраняем основной CSV с добавленными точками падения
        if all_data_with_impact:
            df_output = pd.DataFrame(all_data_with_impact)
            # Сортируем по углу и времени
            df_output = df_output.sort_values(['Угол, град', 't, с']).reset_index(drop=True)
            df_output.to_csv('траектория_полета.csv', index=False, encoding='utf-8-sig')
            print(f"\n✅ Данные сохранены в файл: траектория_полета.csv")
            print(f"   Всего сохранено строк: {len(df_output)} (включая точки падения)")
        else:
            print("\n⚠️ Нет данных для сохранения")



print("\n" + "=" * 80)
print("НАЧАЛО РАСЧЕТА БАЛЛИСТИЧЕСКОЙ ТРАЕКТОРИИ")
print("=" * 80)

# Создание экземпляра класса Simulation (внутри __init__ уже происходят расчеты)
sim = Simulation(max_steps=15000)

print("\n" + "=" * 80)
print("РАСЧЕТ ЗАВЕРШЕН")
print("=" * 80)

# Вывод дополнительной информации
print("\nДоступные DataFrame:")
print(f"  df_1 (угол 20°): {len(sim.df_1)} записей")
print(f"  df_2 (угол 30°): {len(sim.df_2)} записей")
print(f"  df_3 (угол 40°): {len(sim.df_3)} записей")
print(f"  df_4 (угол 50°): {len(sim.df_4)} записей")

# Пример вывода первых строк df_1
print("\n" + "=" * 80)
print("ПЕРВЫЕ 5 СТРОК df_1 (угол 20°):")
print("=" * 80)
print(sim.df_1.head())