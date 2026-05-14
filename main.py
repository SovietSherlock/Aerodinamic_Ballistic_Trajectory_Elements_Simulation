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
        self.velocities = np.array([self.v_01, self.v_02])
        self.angles = np.array([self.Theta_c0_1, self.Theta_c0_2, self.Theta_c0_3, self.Theta_c0_4])

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

    def init_conditions(self):
        # генератор входных параметров для алгоритма Runge_Kutta4:
        for v in self.velocities:
            for Theta in self.angles:
                yield np.array([v, Theta, 0, 0.001, Theta, 0])



    #def init_conditions_1(self):
        # функция входных параметров для Theta_c0_1:
        #return np.array([self.v_01, self.Theta_c0_1, 0, 0.001, 0, 0])

    #def init_conditions_2(self):
        # функция входных параметров для Theta_c0_2:
        #return np.array([self.v_01, self.Theta_c0_2, 0, 0.001, 0, 0])

    #def init_conditions_3(self):
        # функция входных параметров для Theta_c0_3:
        #return np.array([self.v_01, self.Theta_c0_3, 0, 0.001, 0, 0])

    #def init_conditions_4(self):
        # функция входных параметров для Theta_c0_4:
        #return np.array([self.v_01, self.Theta_c0_4, 0, 0.001, 0, 0])


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

        # Словарь для хранения всех результатов
        self.results = {}
        self.dataframes = {}

        # Список для хранения всех комбинаций (DataFrame, скорость, угол)
        self.cases_list = []

        # Перебираем все комбинации через генератор
        for init_cond in self.init_conditions():
            # Определяем скорость и угол из начальных условий
            V0 = init_cond[0]
            Theta0 = init_cond[1]

            # Переводим угол в градусы для названия
            Theta_deg = round(math.degrees(Theta0))
            V_int = int(V0)

            # Выполняем расчет
            result = Runge_Kutta4(
                self.init_ODE_system,
                init_cond,
                self.stop_conditions,
                self.record,
                self.delta_t,
                0,
                max_steps
            )

            # Сохраняем результат
            key = f"V{V_int}_Theta{Theta_deg}"
            self.results[key] = result

            # Создаем DataFrame
            columns = ['tau', 'm_0', 'V', 'a', 'Much_Number', 'C_Xa', 'X_a', 'alpha',
                       'Theta_c_rad', 'dV_dtau', 'C_Ya', 'Y_a', 'dTheta_c_dtau', 'Theta_c_deg',
                       'theta', 'y', 'dy_dtau', 'x', 'dx_dtau', 'M_z_alpha', 'omega_z',
                       'domega_z_dtau', 'rho', 'p']

            df = pd.DataFrame(result[:, 7:31], columns=columns)
            self.dataframes[key] = df

            # Добавляем в список всех случаев
            self.cases_list.append((df, V_int, Theta_deg))

            # Для обратной совместимости создаем атрибуты
            if V_int == 245:
                if Theta_deg == 20:
                    self.df_1 = df
                elif Theta_deg == 30:
                    self.df_2 = df
                elif Theta_deg == 40:
                    self.df_3 = df
                elif Theta_deg == 50:
                    self.df_4 = df
            elif V_int == 952:
                if Theta_deg == 20:
                    self.df_5 = df
                elif Theta_deg == 30:
                    self.df_6 = df
                elif Theta_deg == 40:
                    self.df_7 = df
                elif Theta_deg == 50:
                    self.df_8 = df

        pd.set_option('display.precision', 5)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        # Список для сбора всех данных для CSV
        all_data = []

        # ВЫВОД ДЛЯ ВСЕХ СЛУЧАЕВ (все скорости и углы)
        for df, velocity, angle in self.cases_list:
            df_filtered = df.query('V>=0 and y>=-100').reset_index(drop=True)

            # Заголовок таблицы
            print("\n" + "=" * 310)
            print(f"РЕЗУЛЬТАТЫ РАСЧЕТА ТРАЕКТОРИИ (V0 = {velocity} м/с, начальный угол Θ_c0 = {angle}°)")
            print("=" * 310)
            print(
                f"{'t,с':>6} | {'m,кг':>8} | {'V,м/с':>9} | {'a,м/с':>8} | {'M':>6} | {'C_Xa':>6} | {'X_a,Н':>9} | {'α,рад':>9} | {'Θ_c,рад':>9} | {'dV/dt':>9} | {'C_Ya':>6} | {'Y_a,Н':>8} | {'dΘ_c/dt':>9} | {'Θ_c,град':>9} | {'θ,град':>8} | {'y,м':>8} | {'dy/dt':>8} | {'x,м':>10} | {'dx/dt':>8} | {'M_z^α':>10} | {'ω_z':>8} | {'dω_z/dt':>9} | {'ρ,кг/м³':>11} | {'p,Па':>9}")
            print("-" * 310)

            # Собираем все строки для вывода (регулярные + точка падения)
            all_rows = []

            # Добавляем регулярные строки с шагом 10
            for k in range(0, len(df_filtered), 10):
                if len(df_filtered) > k:
                    row = df_filtered.iloc[k]
                    all_rows.append({
                        'type': 'regular',
                        'tau': row['tau'],
                        'm_0': row['m_0'],
                        'V': row['V'],
                        'a': row['a'],
                        'Much_Number': row['Much_Number'],
                        'C_Xa': row['C_Xa'],
                        'X_a': row['X_a'],
                        'alpha': row['alpha'],
                        'Theta_c_rad': row['Theta_c_rad'],
                        'dV_dtau': row['dV_dtau'],
                        'C_Ya': row['C_Ya'],
                        'Y_a': row['Y_a'],
                        'dTheta_c_dtau': row['dTheta_c_dtau'],
                        'Theta_c_deg': row['Theta_c_deg'],
                        'theta': row['theta'],
                        'y': row['y'],
                        'dy_dtau': row['dy_dtau'],
                        'x': row['x'],
                        'dx_dtau': row['dx_dtau'],
                        'M_z_alpha': row['M_z_alpha'],
                        'omega_z': row['omega_z'],
                        'domega_z_dtau': row['domega_z_dtau'],
                        'rho': row['rho'],
                        'p': row['p']
                    })

            # Поиск и добавление точки падения
            for i in range(1, len(df_filtered)):
                if df_filtered.iloc[i - 1]['y'] > 0 and df_filtered.iloc[i]['y'] <= 0:
                    y1 = df_filtered.iloc[i - 1]['y']
                    y2 = df_filtered.iloc[i]['y']
                    t1 = df_filtered.iloc[i - 1]['tau']
                    t2 = df_filtered.iloc[i]['tau']

                    if t2 != t1:
                        t_impact = t1 + (0 - y1) * (t2 - t1) / (y2 - y1)
                        frac = (t_impact - t1) / (t2 - t1)
                    else:
                        t_impact = t1
                        frac = 0

                    # Интерполяция всех параметров
                    m_impact = df_filtered.iloc[i - 1]['m_0'] + frac * (
                                df_filtered.iloc[i]['m_0'] - df_filtered.iloc[i - 1]['m_0'])
                    V_impact = df_filtered.iloc[i - 1]['V'] + frac * (
                                df_filtered.iloc[i]['V'] - df_filtered.iloc[i - 1]['V'])
                    a_impact = df_filtered.iloc[i - 1]['a'] + frac * (
                                df_filtered.iloc[i]['a'] - df_filtered.iloc[i - 1]['a'])
                    M_impact = df_filtered.iloc[i - 1]['Much_Number'] + frac * (
                                df_filtered.iloc[i]['Much_Number'] - df_filtered.iloc[i - 1]['Much_Number'])
                    C_Xa_impact = df_filtered.iloc[i - 1]['C_Xa'] + frac * (
                                df_filtered.iloc[i]['C_Xa'] - df_filtered.iloc[i - 1]['C_Xa'])
                    X_a_impact = df_filtered.iloc[i - 1]['X_a'] + frac * (
                                df_filtered.iloc[i]['X_a'] - df_filtered.iloc[i - 1]['X_a'])
                    alpha_impact = df_filtered.iloc[i - 1]['alpha'] + frac * (
                                df_filtered.iloc[i]['alpha'] - df_filtered.iloc[i - 1]['alpha'])
                    Theta_c_rad_impact = df_filtered.iloc[i - 1]['Theta_c_rad'] + frac * (
                                df_filtered.iloc[i]['Theta_c_rad'] - df_filtered.iloc[i - 1]['Theta_c_rad'])
                    dV_dtau_impact = df_filtered.iloc[i - 1]['dV_dtau'] + frac * (
                                df_filtered.iloc[i]['dV_dtau'] - df_filtered.iloc[i - 1]['dV_dtau'])
                    C_Ya_impact = df_filtered.iloc[i - 1]['C_Ya'] + frac * (
                                df_filtered.iloc[i]['C_Ya'] - df_filtered.iloc[i - 1]['C_Ya'])
                    Y_a_impact = df_filtered.iloc[i - 1]['Y_a'] + frac * (
                                df_filtered.iloc[i]['Y_a'] - df_filtered.iloc[i - 1]['Y_a'])
                    dTheta_c_dtau_impact = df_filtered.iloc[i - 1]['dTheta_c_dtau'] + frac * (
                                df_filtered.iloc[i]['dTheta_c_dtau'] - df_filtered.iloc[i - 1]['dTheta_c_dtau'])
                    Theta_c_deg_impact = df_filtered.iloc[i - 1]['Theta_c_deg'] + frac * (
                                df_filtered.iloc[i]['Theta_c_deg'] - df_filtered.iloc[i - 1]['Theta_c_deg'])
                    theta_impact = df_filtered.iloc[i - 1]['theta'] + frac * (
                                df_filtered.iloc[i]['theta'] - df_filtered.iloc[i - 1]['theta'])
                    dy_dtau_impact = df_filtered.iloc[i - 1]['dy_dtau'] + frac * (
                                df_filtered.iloc[i]['dy_dtau'] - df_filtered.iloc[i - 1]['dy_dtau'])
                    x_impact = df_filtered.iloc[i - 1]['x'] + frac * (
                                df_filtered.iloc[i]['x'] - df_filtered.iloc[i - 1]['x'])
                    dx_dtau_impact = df_filtered.iloc[i - 1]['dx_dtau'] + frac * (
                                df_filtered.iloc[i]['dx_dtau'] - df_filtered.iloc[i - 1]['dx_dtau'])
                    M_z_alpha_impact = df_filtered.iloc[i - 1]['M_z_alpha'] + frac * (
                                df_filtered.iloc[i]['M_z_alpha'] - df_filtered.iloc[i - 1]['M_z_alpha'])
                    omega_z_impact = df_filtered.iloc[i - 1]['omega_z'] + frac * (
                                df_filtered.iloc[i]['omega_z'] - df_filtered.iloc[i - 1]['omega_z'])
                    domega_z_dtau_impact = df_filtered.iloc[i - 1]['domega_z_dtau'] + frac * (
                                df_filtered.iloc[i]['domega_z_dtau'] - df_filtered.iloc[i - 1]['domega_z_dtau'])
                    rho_impact = df_filtered.iloc[i - 1]['rho'] + frac * (
                                df_filtered.iloc[i]['rho'] - df_filtered.iloc[i - 1]['rho'])
                    p_impact = df_filtered.iloc[i - 1]['p'] + frac * (
                                df_filtered.iloc[i]['p'] - df_filtered.iloc[i - 1]['p'])

                    impact_row = {
                        'type': 'impact',
                        'tau': t_impact,
                        'm_0': m_impact,
                        'V': V_impact,
                        'a': a_impact,
                        'Much_Number': M_impact,
                        'C_Xa': C_Xa_impact,
                        'X_a': X_a_impact,
                        'alpha': alpha_impact,
                        'Theta_c_rad': Theta_c_rad_impact,
                        'dV_dtau': dV_dtau_impact,
                        'C_Ya': C_Ya_impact,
                        'Y_a': Y_a_impact,
                        'dTheta_c_dtau': dTheta_c_dtau_impact,
                        'Theta_c_deg': Theta_c_deg_impact,
                        'theta': theta_impact,
                        'y': 0.0,
                        'dy_dtau': dy_dtau_impact,
                        'x': x_impact,
                        'dx_dtau': dx_dtau_impact,
                        'M_z_alpha': M_z_alpha_impact,
                        'omega_z': omega_z_impact,
                        'domega_z_dtau': domega_z_dtau_impact,
                        'rho': rho_impact,
                        'p': p_impact
                    }
                    all_rows.append(impact_row)
                    break

            # Сортируем строки по времени
            all_rows.sort(key=lambda x: x['tau'])

            # Вывод всех строк в одной таблице
            for row in all_rows:
                suffix = " *ПАДЕНИЕ*" if row['type'] == 'impact' else ""
                print(
                    f"{row['tau']:6.2f} | {row['m_0']:8.2f} | {row['V']:9.2f} | {row['a']:8.2f} | {row['Much_Number']:6.4f} | {row['C_Xa']:6.4f} | {row['X_a']:9.2f} | {row['alpha']:9.5f} | {row['Theta_c_rad']:9.4f} | {row['dV_dtau']:9.2f} | {row['C_Ya']:6.4f} | {row['Y_a']:8.2f} | {row['dTheta_c_dtau']:9.4f} | {row['Theta_c_deg']:9.3f} | {row['theta']:8.2f} | {row['y']:8.2f} | {row['dy_dtau']:8.2f} | {row['x']:10.2f} | {row['dx_dtau']:8.2f} | {row['M_z_alpha']:10.2f} | {row['omega_z']:8.4f} | {row['domega_z_dtau']:9.4f} | {row['rho']:11.5f} | {row['p']:9.1f}{suffix}")

                # Сохраняем ВСЕ строки в all_data с округлением
                all_data.append({
                    'Скорость, м/с': velocity,
                    'Угол, град': angle,
                    't, с': round(row['tau'], 2),
                    'm, кг': round(row['m_0'], 2),
                    'V, м/с': round(row['V'], 2),
                    'a, м/с': round(row['a'], 2),
                    'M': round(row['Much_Number'], 4),
                    'C_Xa': round(row['C_Xa'], 4),
                    'X_a, Н': round(row['X_a'], 2),
                    'α, рад': round(row['alpha'], 5),
                    'Θ_c, рад': round(row['Theta_c_rad'], 4),
                    'dV/dt, м/с²': round(row['dV_dtau'], 2),
                    'C_Ya': round(row['C_Ya'], 4),
                    'Y_a, Н': round(row['Y_a'], 2),
                    'dΘ_c/dt, с⁻¹': round(row['dTheta_c_dtau'], 4),
                    'Θ_c, град': round(row['Theta_c_deg'], 3),
                    'θ, град': round(row['theta'], 2),
                    'y, м': round(row['y'], 2),
                    'dy/dt, м/с': round(row['dy_dtau'], 2),
                    'x, м': round(row['x'], 2),
                    'dx/dt, м/с': round(row['dx_dtau'], 2),
                    'M_z^α, Н·м/рад': round(row['M_z_alpha'], 2),
                    'ω_z, с⁻¹': round(row['omega_z'], 4),
                    'dω_z/dt, с⁻²': round(row['domega_z_dtau'], 4),
                    'ρ, кг/м³': round(row['rho'], 5),
                    'p, Па': round(row['p'], 1)
                })

            print("=" * 310)

        # Сохраняем CSV
        if all_data:
            df_output = pd.DataFrame(all_data)
            df_output = df_output.sort_values(['Скорость, м/с', 'Угол, град', 't, с']).reset_index(drop=True)
            df_output.to_csv('траектория_полета.csv', index=False, encoding='utf-8-sig')
            print(f"\n✅ Данные сохранены в файл: траектория_полета.csv")
            print(f"   Всего сохранено строк: {len(df_output)} (включая точки падения)")
        else:
            print("\n⚠️ Нет данных для сохранения")


class Plotter:
    # Класс вывода графиков зависимостей параметров ЛА от координаты и от времени

    def __init__(self, sim_instance):
        self.sim = sim_instance
        self.velocities = [245, 952]
        self.angles = [20, 30, 40, 50]

        # Цвета для разных углов
        self.angle_colors = {
            20: 'steelblue',
            30: 'seagreen',
            40: 'mediumpurple',
            50: 'coral'
        }

        # Настройка шрифтов
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['lines.linewidth'] = 1

        # Использование Times New Roman в математических формулах:
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

        # Словарь для хранения данных по скоростям и углам
        self.data = {}
        for V in self.velocities:
            for angle in self.angles:
                key = f"V{V}_Theta{angle}"
                if key in self.sim.dataframes:
                    self.data[(V, angle)] = self.sim.dataframes[key]

    def _get_linestyle(self, V):
        #Возвращает стиль линии в зависимости от скорости
        return '-' if V == 245 else '--'

    def _get_color(self, angle):
        #Возвращает цвет в зависимости от угла
        return self.angle_colors.get(angle, 'black')

    def plot_V_t(self, save_path=None):
        #График зависимости скорости от времени V(t)
        fig, ax = plt.subplots(figsize=(20, 10))

        for V in self.velocities:
            linestyle = self._get_linestyle(V)
            for angle in self.angles:
                color = self._get_color(angle)
                if (V, angle) in self.data:
                    df = self.data[(V, angle)]
                    ax.plot(df['tau'], df['V'],
                            color=color, linestyle=linestyle, linewidth=1,
                            label=f'$V_0={V}$ м/с, $\\Theta_0={angle}^\\circ$')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$V$, м/с')
        ax.set_title('Зависимость скорости от времени $V(t)$')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_Theta_c_t(self, save_path=None):
        #График зависимости угла траектории от времени Θ_c(t)
        fig, ax = plt.subplots(figsize=(20, 10))

        for V in self.velocities:
            linestyle = self._get_linestyle(V)
            for angle in self.angles:
                color = self._get_color(angle)
                if (V, angle) in self.data:
                    df = self.data[(V, angle)]
                    ax.plot(df['tau'], df['Theta_c_deg'],
                            color=color, linestyle=linestyle, linewidth=1,
                            label=f'$V_0={V}$ м/с, $\\Theta_0={angle}^\\circ$')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$\\Theta_c$, град')
        ax.set_title('Зависимость угла траектории от времени $\\Theta_c(t)$')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_y_t(self, save_path=None):
        #График зависимости высоты от времени y(t)
        fig, ax = plt.subplots(figsize=(20, 10))

        for V in self.velocities:
            linestyle = self._get_linestyle(V)
            for angle in self.angles:
                color = self._get_color(angle)
                if (V, angle) in self.data:
                    df = self.data[(V, angle)]
                    ax.plot(df['tau'], df['y'],
                            color=color, linestyle=linestyle, linewidth=1,
                            label=f'$V_0={V}$ м/с, $\\Theta_0={angle}^\\circ$')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$y$, м')
        ax.set_title('Зависимость высоты от времени $y(t)$')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_x_t(self, save_path=None):
        #График зависимости дальности от времени x(t)
        fig, ax = plt.subplots(figsize=(20, 10))

        for V in self.velocities:
            linestyle = self._get_linestyle(V)
            for angle in self.angles:
                color = self._get_color(angle)
                if (V, angle) in self.data:
                    df = self.data[(V, angle)]
                    ax.plot(df['tau'], df['x'],
                            color=color, linestyle=linestyle, linewidth=1,
                            label=f'$V_0={V}$ м/с, $\\Theta_0={angle}^\\circ$')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$x$, м')
        ax.set_title('Зависимость дальности от времени $x(t)$')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_theta_t(self, save_path=None):
        #График зависимости угла тангажа от времени θ(t)
        fig, ax = plt.subplots(figsize=(20, 10))

        for V in self.velocities:
            linestyle = self._get_linestyle(V)
            for angle in self.angles:
                color = self._get_color(angle)
                if (V, angle) in self.data:
                    df = self.data[(V, angle)]
                    ax.plot(df['tau'], df['theta'],
                            color=color, linestyle=linestyle, linewidth=1,
                            label=f'$V_0={V}$ м/с, $\\Theta_0={angle}^\\circ$')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$\\theta$, град')
        ax.set_title('Зависимость угла тангажа от времени $\\theta(t)$')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_alpha_t(self, save_path=None):
        #График зависимости угла атаки от времени α(t)
        fig, ax = plt.subplots(figsize=(20, 10))

        for V in self.velocities:
            linestyle = self._get_linestyle(V)
            for angle in self.angles:
                color = self._get_color(angle)
                if (V, angle) in self.data:
                    df = self.data[(V, angle)]
                    ax.plot(df['tau'], df['alpha'],
                            color=color, linestyle=linestyle, linewidth=1,
                            label=f'$V_0={V}$ м/с, $\\Theta_0={angle}^\\circ$')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$\\alpha$, рад')
        ax.set_title('Зависимость угла атаки от времени $\\alpha(t)$')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_V_x(self, save_path=None):
        #График зависимости скорости от дальности V(x)
        fig, ax = plt.subplots(figsize=(20, 10))

        for V in self.velocities:
            linestyle = self._get_linestyle(V)
            for angle in self.angles:
                color = self._get_color(angle)
                if (V, angle) in self.data:
                    df = self.data[(V, angle)]
                    df_filtered = df[df['x'] >= 0]
                    ax.plot(df_filtered['x'], df_filtered['V'],
                            color=color, linestyle=linestyle, linewidth=1,
                            label=f'$V_0={V}$ м/с, $\\Theta_0={angle}^\\circ$')

        ax.set_xlabel('$x$, м')
        ax.set_ylabel('$V$, м/с')
        ax.set_title('Зависимость скорости от дальности $V(x)$')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_Theta_c_x(self, save_path=None):
        #График зависимости угла траектории от дальности Θ_c(x)
        fig, ax = plt.subplots(figsize=(20, 10))

        for V in self.velocities:
            linestyle = self._get_linestyle(V)
            for angle in self.angles:
                color = self._get_color(angle)
                if (V, angle) in self.data:
                    df = self.data[(V, angle)]
                    df_filtered = df[df['x'] >= 0]
                    ax.plot(df_filtered['x'], df_filtered['Theta_c_deg'],
                            color=color, linestyle=linestyle, linewidth=1,
                            label=f'$V_0={V}$ м/с, $\\Theta_0={angle}^\\circ$')

        ax.set_xlabel('$x$, м')
        ax.set_ylabel('$\\Theta_c$, град')
        ax.set_title('Зависимость угла траектории от дальности $\\Theta_c(x)$')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_y_x(self, save_path=None):
        #График траектории полета y(x)
        fig, ax = plt.subplots(figsize=(20, 10))

        for V in self.velocities:
            linestyle = self._get_linestyle(V)
            for angle in self.angles:
                color = self._get_color(angle)
                if (V, angle) in self.data:
                    df = self.data[(V, angle)]
                    df_filtered = df[df['x'] >= 0]
                    ax.plot(df_filtered['x'], df_filtered['y'],
                            color=color, linestyle=linestyle, linewidth=1,
                            label=f'$V_0={V}$ м/с, $\\Theta_0={angle}^\\circ$')

        ax.set_xlabel('$x$, м')
        ax.set_ylabel('$y$, м')
        ax.set_title('Траектория полета $y(x)$')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_omega_z_t(self, save_path=None):
        #График зависимости угловой скорости от времени ω_z(t)
        fig, ax = plt.subplots(figsize=(20, 10))

        for V in self.velocities:
            linestyle = self._get_linestyle(V)
            for angle in self.angles:
                color = self._get_color(angle)
                if (V, angle) in self.data:
                    df = self.data[(V, angle)]
                    ax.plot(df['tau'], df['omega_z'],
                            color=color, linestyle=linestyle, linewidth=1,
                            label=f'$V_0={V}$ м/с, $\\Theta_0={angle}^\\circ$')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$\\omega_z$, с$^{-1}$')
        ax.set_title('Зависимость угловой скорости от времени $\\omega_z(t)$')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_all(self, save_dir="graphics"):
        #Построение всех графиков с сохранением в папку graphics
        import os
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("ПОСТРОЕНИЕ И СОХРАНЕНИЕ ГРАФИКОВ")
        print("=" * 80)

        graphs = [
            ('V(t)', self.plot_V_t),
            ('Theta_c(t)', self.plot_Theta_c_t),
            ('y(t)', self.plot_y_t),
            ('x(t)', self.plot_x_t),
            ('theta(t)', self.plot_theta_t),
            ('alpha(t)', self.plot_alpha_t),
            ('V(x)', self.plot_V_x),
            ('Theta_c(x)', self.plot_Theta_c_x),
            ('y(x)', self.plot_y_x),
            ('omega_z(t)', self.plot_omega_z_t)
        ]

        for name, func in graphs:
            print(f"  Строим и сохраняем: {name}")
            save_path = os.path.join(save_dir, f"{name.replace('(', '_').replace(')', '')}.png")
            func(save_path=save_path)
            plt.close()  # Закрываем фигуру после сохранения

        print(f"\n✅ Все графики сохранены в папку: {save_dir}")
        print("=" * 80)


class inverse_problem(Simulation):
    # Класс вычисления оптимальных значений начального угла наклона траектории, обеспечивающих максимальную дальность полета ЛА

    def __init__(self):
        super().__init__()

# ==================== Объектная часть ====================

print("\n" + "=" * 80)
print("НАЧАЛО РАСЧЕТА БАЛЛИСТИЧЕСКОЙ ТРАЕКТОРИИ")
print("=" * 80)

# Создание экземпляра класса Simulation
sim = Simulation(max_steps=15000)

print("\n" + "=" * 80)
print("РАСЧЕТ ЗАВЕРШЕН")
print("=" * 80)

# Построение графиков
plotter = Plotter(sim)
plotter.plot_all(save_dir="results/graphs")

# Вывод дополнительной информации
print("\nДоступные DataFrame:")
print(f"\nВсего рассчитано траекторий: {len(sim.dataframes)}")
for key, df in sim.dataframes.items():
    print(f"  {key}: {len(df)} записей")

print("\nТаблицы для V=245 м/с:")
if hasattr(sim, 'df_1'):
    print(f"  df_1 (V=245, угол 20°): {len(sim.df_1)} записей")
if hasattr(sim, 'df_2'):
    print(f"  df_2 (V=245, угол 30°): {len(sim.df_2)} записей")
if hasattr(sim, 'df_3'):
    print(f"  df_3 (V=245, угол 40°): {len(sim.df_3)} записей")
if hasattr(sim, 'df_4'):
    print(f"  df_4 (V=245, угол 50°): {len(sim.df_4)} записей")

# Построение и сохранение графиков
plotter = Plotter(sim)
plotter.plot_all(save_dir="graphics")  # Сохраняем в папку graphics