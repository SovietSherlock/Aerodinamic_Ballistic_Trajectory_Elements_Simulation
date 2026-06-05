import numpy as np
import matplotlib.pyplot as plt
from Runge_Kutta4 import *
import math
from atmosphere import *
from math import sin, cos, radians, degrees
import pandas as pd
import os


class Aircraft_Initial_Parameters_Optimal:
    # Класс исходных параметров ЛА

    def __init__(self):
        # Кинематические параметры начального состояния движения ЛА:
        self.v_01 = 245  # начальная скорость летательного аппарата в первом случае, м/с
        self.v_02 = 952  # начальная скорость летательного аппарата во втором случае, м/с
        self.Theta_c0_1 = math.radians(44.25)  # Оптимальный начальный угол наклона траектории для 245 м/с 44,20°
        self.Theta_c0_2 = math.radians(47.40)  # Оптимальный начальный угол наклона траектории для 952 м/с 47,40°

        # Инерционные параметры ЛА:
        self.g_0 = 9.80665  # ускорение силы притяжения на поверхности Земли, м/(с^2)
        self.m_0 = 800  # начальная масса ЛА, кг
        self.J_z = 120  # момент инерции ЛА относительно связанной оси z, кг*м^2

        # Геометрические параметры ЛА:
        self.S_m = 0.2  # характерная площадь ЛА, м^2
        self.delta_l = 0.4  # расстояние от центра давления до центра масс ЛА, м

        # Зависимости аэродинамических коэффициентов от числа Маха:
        self.M = np.array([0.01, 0.55, 0.80, 0.90, 1.00, 1.06, 1.10, 1.20, 1.30, 1.40, 2.00, 2.60, 3.40, 6.00, 10.0])
        self.C_Xa = np.array([0.30, 0.30, 0.55, 0.70, 0.84, 0.86, 0.87, 0.83, 0.80, 0.79, 0.65, 0.55, 0.50, 0.45, 0.40])
        self.C_Ya = np.array([0.25, 0.25, 0.25, 0.20, 0.30, 0.31, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

        # Шаг интегрирования:
        self.delta_t = 0.1  # сек

        # Аэродинамические коэффициенты:
        self.C_Ya_interp = None
        self.C_Xa_interp = None

    def interp_C_XY_a(self, M):
        # функция интерполяции аэродинамических коэффициентов C_Xa и C_Ya:
        M_limit = np.clip(M, self.M[0], self.M[-1])
        self.C_Xa_interp = np.interp(M_limit, self.M, self.C_Xa)
        self.C_Ya_interp = np.interp(M_limit, self.M, self.C_Ya)
        return self.C_Xa_interp, self.C_Ya_interp


class Math_Model_Optimal(Aircraft_Initial_Parameters_Optimal):
    # Класс реализации математической модели

    def __init__(self):
        super().__init__()
        self.dt_dtau = None
        self.atm = Atmosphere_GOST_4401_81()
        self.dtau = self.delta_t

    def alpha(self, t):
        return t[5] - t[1]

    def X_a(self, t):
        return self.C_Xa_interp * self.S_m * (self.atm.rho(t[3]) / 2) * (t[0] ** 2)

    def Y_a(self, t):
        return self.C_Ya_interp * self.S_m * (self.atm.rho(t[3]) / 2) * (t[0] ** 2) * self.alpha(t)

    def M_z_alpha(self, t):
        return -(self.C_Xa_interp + self.C_Ya_interp) * self.S_m * (self.atm.rho(t[3]) / 2) * (t[0] ** 2) * self.delta_l

    def a(self, t):
        return 20.046796 * (self.atm.T(t[3]) ** (1 / 2))

    def Mach_number(self, t):
        return t[0] / self.a(t)

    def ODE_system(self, tau, t):
        self.dt_dtau = np.zeros(6)
        self.dt_dtau[0] = -self.X_a(t) / self.m_0 - self.atm.g(t[3]) * sin(t[1])
        self.dt_dtau[1] = self.Y_a(t) / (self.m_0 * t[0]) - self.atm.g(t[3]) * cos(t[1]) / t[0]
        self.dt_dtau[2] = t[0] * cos(t[1])
        self.dt_dtau[3] = t[0] * sin(t[1])
        self.dt_dtau[4] = (self.M_z_alpha(t) / self.J_z) * self.alpha(t)
        self.dt_dtau[5] = t[4]
        return np.array(self.dt_dtau)

    def init_ODE_system(self, tau, t):
        M = self.Mach_number(t)
        self.interp_C_XY_a(M)
        return self.ODE_system(tau, t)

    def record(self, tau, t):
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

    def stop_conditions(self, t):
        if t[3] < 0:
            return True
        return False

    def time_step(self, t):
        self.dtau = self.delta_t
        return self.dtau


class Simulation_Optimal:
    # Класс для расчета оптимальных траекторий

    def __init__(self, max_steps=15000):
        self.model = Math_Model_Optimal()
        self.max_steps = max_steps
        self.results = {}
        self.dataframes = {}

        # Оптимальные параметры
        self.optimal_cases = [
            {'V': 245, 'angle_deg': 44.20, 'angle_rad': math.radians(44.20), 'color': 'crimson',
             'name': 'V₀ = 245 м/с, Θ₀ = 44.20°'},
            {'V': 952, 'angle_deg': 47.40, 'angle_rad': math.radians(47.40), 'color': 'royalblue',
             'name': 'V₀ = 952 м/с, Θ₀ = 47.40°'}
        ]

        # Выполняем расчеты
        for case in self.optimal_cases:
            init_cond = np.array([case['V'], case['angle_rad'], 0, 0.001, 0, case['angle_rad']])

            result = Runge_Kutta4(
                self.model.init_ODE_system,
                init_cond,
                self.model.stop_conditions,
                self.model.record,
                self.model.delta_t,
                0,
                max_steps
            )

            columns = ['tau', 'm_0', 'V', 'a', 'Much_Number', 'C_Xa', 'X_a', 'alpha',
                       'Theta_c_rad', 'dV_dtau', 'C_Ya', 'Y_a', 'dTheta_c_dtau', 'Theta_c_deg',
                       'theta', 'y', 'dy_dtau', 'x', 'dx_dtau', 'M_z_alpha', 'omega_z',
                       'domega_z_dtau', 'rho', 'p']

            df = pd.DataFrame(result[:, 7:31], columns=columns)
            case['df'] = df
            self.dataframes[f"V{case['V']}_Theta{case['angle_deg']}"] = df

        # Вывод таблиц в отформатированном виде
        self._print_tables_formatted()
        # Сохранение CSV
        self._save_to_csv()

    def _print_tables_formatted(self):
        """Вывод таблиц с правильным форматированием и выравниванием (с запятой как разделителем)"""

        # Формируем заголовки с нужной шириной
        headers = [
            ('t,с', 8), ('m,кг', 8), ('V,м/с', 9), ('a,м/с', 8), ('M', 8),
            ('C_Xa', 8), ('X_a,Н', 10), ('α,рад', 10), ('Θ_c,рад', 10), ('dV/dt', 10),
            ('C_Ya', 8), ('Y_a,Н', 9), ('dΘ_c/dt', 10), ('Θ_c,град', 10), ('θ,град', 8),
            ('y,м', 9), ('dy/dt', 8), ('x,м', 10), ('dx/dt', 8), ('M_z^α', 11),
            ('ω_z', 9), ('dω_z/dt', 10), ('ρ,кг/м³', 10), ('p,Па', 9)
        ]

        for case in self.optimal_cases:
            df = case['df']
            V = case['V']
            angle = case['angle_deg']

            # Фильтруем данные
            df_filtered = df.query('V>=0 and y>=-100').reset_index(drop=True)

            # Заголовок
            print("\n" + "=" * 230)
            print(f"ОПТИМАЛЬНАЯ ТРАЕКТОРИЯ (V₀ = {V} м/с, Θ₀ = {angle:.2f}°)")
            print("=" * 230)

            # Формируем строку заголовка
            header_line = ""
            for h, w in headers:
                header_line += f"{h:>{w}} │ "
            header_line = header_line.rstrip(" │ ")
            print(header_line)
            print("-" * len(header_line))

            # Функция для замены точки на запятую
            def format_with_comma(value, format_str):
                """Форматирует число и заменяет точку на запятую"""
                return format_str.format(value).replace('.', ',')

            # Собираем строки для вывода
            all_rows = []

            # Добавляем строки с шагом (не более 40 строк на траекторию)
            step = 10

            for k in range(0, len(df_filtered), step):
                if len(df_filtered) > k:
                    row = df_filtered.iloc[k]
                    all_rows.append({
                        'type': 'regular',
                        't': row['tau'],
                        'm': row['m_0'],
                        'V': row['V'],
                        'a': row['a'],
                        'M': row['Much_Number'],
                        'C_Xa': row['C_Xa'],
                        'X_a': row['X_a'],
                        'alpha': row['alpha'],
                        'Theta_c_rad': row['Theta_c_rad'],
                        'dV_dt': row['dV_dtau'],
                        'C_Ya': row['C_Ya'],
                        'Y_a': row['Y_a'],
                        'dTheta_c_dt': row['dTheta_c_dtau'],
                        'Theta_c_deg': row['Theta_c_deg'],
                        'theta': row['theta'],
                        'y': row['y'],
                        'dy_dt': row['dy_dtau'],
                        'x': row['x'],
                        'dx_dt': row['dx_dtau'],
                        'M_z_alpha': row['M_z_alpha'],
                        'omega_z': row['omega_z'],
                        'domega_z_dt': row['domega_z_dtau'],
                        'rho': row['rho'],
                        'p': row['p']
                    })

            # Поиск точки падения
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

                    def interp(key):
                        return df_filtered.iloc[i - 1][key] + frac * (
                                    df_filtered.iloc[i][key] - df_filtered.iloc[i - 1][key])

                    all_rows.append({
                        'type': 'impact',
                        't': t_impact,
                        'm': interp('m_0'),
                        'V': interp('V'),
                        'a': interp('a'),
                        'M': interp('Much_Number'),
                        'C_Xa': interp('C_Xa'),
                        'X_a': interp('X_a'),
                        'alpha': interp('alpha'),
                        'Theta_c_rad': interp('Theta_c_rad'),
                        'dV_dt': interp('dV_dtau'),
                        'C_Ya': interp('C_Ya'),
                        'Y_a': interp('Y_a'),
                        'dTheta_c_dt': interp('dTheta_c_dtau'),
                        'Theta_c_deg': interp('Theta_c_deg'),
                        'theta': interp('theta'),
                        'y': 0.0,
                        'dy_dt': interp('dy_dtau'),
                        'x': interp('x'),
                        'dx_dt': interp('dx_dtau'),
                        'M_z_alpha': interp('M_z_alpha'),
                        'omega_z': interp('omega_z'),
                        'domega_z_dt': interp('domega_z_dtau'),
                        'rho': interp('rho'),
                        'p': interp('p')
                    })
                    break

            # Сортируем по времени
            all_rows.sort(key=lambda x: x['t'])

            # Вывод строк с форматированием (с запятой вместо точки)
            for row in all_rows:
                suffix = "  *** ПАДЕНИЕ ***" if row['type'] == 'impact' else ""

                # Форматируем каждое значение с нужной шириной и заменяем точку на запятую
                line = (f"{format_with_comma(row['t'], '{:>8.2f}')} │ "
                        f"{format_with_comma(row['m'], '{:>8.2f}')} │ "
                        f"{format_with_comma(row['V'], '{:>9.2f}')} │ "
                        f"{format_with_comma(row['a'], '{:>8.2f}')} │ "
                        f"{format_with_comma(row['M'], '{:>8.4f}')} │ "
                        f"{format_with_comma(row['C_Xa'], '{:>8.4f}')} │ "
                        f"{format_with_comma(row['X_a'], '{:>10.2f}')} │ "
                        f"{format_with_comma(row['alpha'], '{:>10.5f}')} │ "
                        f"{format_with_comma(row['Theta_c_rad'], '{:>10.4f}')} │ "
                        f"{format_with_comma(row['dV_dt'], '{:>10.2f}')} │ "
                        f"{format_with_comma(row['C_Ya'], '{:>8.4f}')} │ "
                        f"{format_with_comma(row['Y_a'], '{:>9.2f}')} │ "
                        f"{format_with_comma(row['dTheta_c_dt'], '{:>10.4f}')} │ "
                        f"{format_with_comma(row['Theta_c_deg'], '{:>10.3f}')} │ "
                        f"{format_with_comma(row['theta'], '{:>8.2f}')} │ "
                        f"{format_with_comma(row['y'], '{:>9.2f}')} │ "
                        f"{format_with_comma(row['dy_dt'], '{:>8.2f}')} │ "
                        f"{format_with_comma(row['x'], '{:>10.2f}')} │ "
                        f"{format_with_comma(row['dx_dt'], '{:>8.2f}')} │ "
                        f"{format_with_comma(row['M_z_alpha'], '{:>11.2f}')} │ "
                        f"{format_with_comma(row['omega_z'], '{:>9.4f}')} │ "
                        f"{format_with_comma(row['domega_z_dt'], '{:>10.4f}')} │ "
                        f"{format_with_comma(row['rho'], '{:>10.5f}')} │ "
                        f"{format_with_comma(row['p'], '{:>9.1f}')}{suffix}")

                # Для точки падения добавляем разделитель
                if row['type'] == 'impact':
                    print("-" * len(header_line))
                    print(line)
                    print("=" * len(header_line))
                else:
                    print(line)

            print("=" * 230)

            # Вывод сводной информации (с запятой)
            max_height = df_filtered['y'].max()
            max_range = df_filtered['x'].max()
            flight_time = df_filtered['tau'].max()
            impact_row = [r for r in all_rows if r['type'] == 'impact']
            impact_angle = impact_row[0]['Theta_c_deg'] if impact_row else 0
            impact_velocity = impact_row[0]['V'] if impact_row else 0

            print(f"\n📊 Сводка для V₀ = {V} м/с, Θ₀ = {angle:.2f}°:")
            print(f"   • Максимальная высота:          {format_with_comma(max_height, '{:>10.2f}')} м")
            print(f"   • Дальность полета:             {format_with_comma(max_range, '{:>10.2f}')} м")
            print(f"   • Время полета:                 {format_with_comma(flight_time, '{:>10.2f}')} с")
            print(f"   • Скорость в момент падения:    {format_with_comma(impact_velocity, '{:>10.2f}')} м/с")
            print(f"   • Угол в момент падения:        {format_with_comma(impact_angle, '{:>10.3f}')} град")
            print("-" * 80)

    def _save_to_csv(self):
        """Сохранение оптимальных траекторий в CSV файл"""

        # Создаем папку results/optimization, если её нет
        os.makedirs("results/optimization", exist_ok=True)

        # Заголовки в порядке как в исходном файле
        headers = [
            'Скорость, м/с', 'Угол, град', 't, с', 'm, кг', 'V, м/с', 'a, м/с', 'M', 'C_Xa',
            'X_a, Н', 'α, рад', 'Θ_c, рад', 'dV/dt, м/с²', 'C_Ya', 'Y_a, Н', 'dΘ_c/dt, с⁻¹',
            'Θ_c, град', 'θ, град', 'y, м', 'dy/dt, м/с', 'x, м', 'dx/dt, м/с', 'M_z^α, Н·м/рад',
            'ω_z, с⁻¹', 'dω_z/dt, с⁻²', 'ρ, кг/м³', 'p, Па'
        ]

        all_data = []

        # Функция для форматирования числа с запятой
        def format_number(value, decimals=2):
            if isinstance(value, (int, float)):
                # Заменяем точку на запятую
                return f"{value:.{decimals}f}".replace('.', ',')
            return str(value)

        for case in self.optimal_cases:
            df = case['df']
            V = case['V']
            angle = case['angle_deg']

            df_filtered = df.query('V>=0 and y>=-100').reset_index(drop=True)

            # Добавляем все строки с шагом для уменьшения размера файла
            step = 10

            for k in range(0, len(df_filtered), step):
                if len(df_filtered) > k:
                    row = df_filtered.iloc[k]
                    all_data.append({
                        'Скорость, м/с': str(V),
                        'Угол, град': format_number(angle, 2),
                        't, с': format_number(row['tau'], 2),
                        'm, кг': format_number(row['m_0'], 2),
                        'V, м/с': format_number(row['V'], 2),
                        'a, м/с': format_number(row['a'], 2),
                        'M': format_number(row['Much_Number'], 4),
                        'C_Xa': format_number(row['C_Xa'], 4),
                        'X_a, Н': format_number(row['X_a'], 2),
                        'α, рад': format_number(row['alpha'], 5),
                        'Θ_c, рад': format_number(row['Theta_c_rad'], 4),
                        'dV/dt, м/с²': format_number(row['dV_dtau'], 2),
                        'C_Ya': format_number(row['C_Ya'], 4),
                        'Y_a, Н': format_number(row['Y_a'], 2),
                        'dΘ_c/dt, с⁻¹': format_number(row['dTheta_c_dtau'], 4),
                        'Θ_c, град': format_number(row['Theta_c_deg'], 3),
                        'θ, град': format_number(row['theta'], 2),
                        'y, м': format_number(row['y'], 2),
                        'dy/dt, м/с': format_number(row['dy_dtau'], 2),
                        'x, м': format_number(row['x'], 2),
                        'dx/dt, м/с': format_number(row['dx_dtau'], 2),
                        'M_z^α, Н·м/рад': format_number(row['M_z_alpha'], 2),
                        'ω_z, с⁻¹': format_number(row['omega_z'], 4),
                        'dω_z/dt, с⁻²': format_number(row['domega_z_dtau'], 4),
                        'ρ, кг/м³': format_number(row['rho'], 5),
                        'p, Па': format_number(row['p'], 1)
                    })

            # Добавляем точку падения
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

                    def interp(key):
                        return df_filtered.iloc[i - 1][key] + frac * (
                                    df_filtered.iloc[i][key] - df_filtered.iloc[i - 1][key])

                    all_data.append({
                        'Скорость, м/с': str(V),
                        'Угол, град': format_number(angle, 2),
                        't, с': format_number(t_impact, 2),
                        'm, кг': format_number(interp('m_0'), 2),
                        'V, м/с': format_number(interp('V'), 2),
                        'a, м/с': format_number(interp('a'), 2),
                        'M': format_number(interp('Much_Number'), 4),
                        'C_Xa': format_number(interp('C_Xa'), 4),
                        'X_a, Н': format_number(interp('X_a'), 2),
                        'α, рад': format_number(interp('alpha'), 5),
                        'Θ_c, рад': format_number(interp('Theta_c_rad'), 4),
                        'dV/dt, м/с²': format_number(interp('dV_dtau'), 2),
                        'C_Ya': format_number(interp('C_Ya'), 4),
                        'Y_a, Н': format_number(interp('Y_a'), 2),
                        'dΘ_c/dt, с⁻¹': format_number(interp('dTheta_c_dtau'), 4),
                        'Θ_c, град': format_number(interp('Theta_c_deg'), 3),
                        'θ, град': format_number(interp('theta'), 2),
                        'y, м': "0,00",
                        'dy/dt, м/с': format_number(interp('dy_dtau'), 2),
                        'x, м': format_number(interp('x'), 2),
                        'dx/dt, м/с': format_number(interp('dx_dtau'), 2),
                        'M_z^α, Н·м/рад': format_number(interp('M_z_alpha'), 2),
                        'ω_z, с⁻¹': format_number(interp('omega_z'), 4),
                        'dω_z/dt, с⁻²': format_number(interp('domega_z_dtau'), 4),
                        'ρ, кг/м³': format_number(interp('rho'), 5),
                        'p, Па': format_number(interp('p'), 1)
                    })
                    break

        # Сохраняем в CSV
        if all_data:
            df_output = pd.DataFrame(all_data)

            # Сортируем по скорости и времени
            df_output['t_sort'] = df_output['t, с'].str.replace(',', '.').astype(float)
            df_output['V_sort'] = df_output['Скорость, м/с'].astype(float)
            df_output = df_output.sort_values(['V_sort', 't_sort']).reset_index(drop=True)
            df_output = df_output.drop(columns=['t_sort', 'V_sort'])

            # Сохраняем с точкой с запятой в качестве разделителя полей
            csv_path = "results/optimization/оптимальная_траектория.csv"
            df_output.to_csv(csv_path, index=False, sep=',', encoding='utf-8-sig')

            print(f"\n✅ Оптимальные траектории сохранены в файл: {csv_path}")
            print(f"   Всего сохранено строк: {len(df_output)} (включая точки падения)")
            print(f"   Разделитель полей: точка с запятой (;)")
            print(f"   Десятичный разделитель: запятая (,)")


class Plotter_Optimal:
    # Класс для построения графиков оптимальных траекторий

    def __init__(self, sim_instance):
        self.sim = sim_instance

        # Настройка шрифтов
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 16
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['legend.fontsize'] = 16
        plt.rcParams['xtick.labelsize'] = 16
        plt.rcParams['ytick.labelsize'] = 16
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['lines.linewidth'] = 2

        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    def plot_trajectories(self, save_path=None):
        """График траекторий полета y(x)"""
        fig, ax = plt.subplots(figsize=(20, 10))

        for case in self.sim.optimal_cases:
            df = case['df']
            df_filtered = df[df['x'] >= 0]
            ax.plot(df_filtered['x'], df_filtered['y'],
                    color=case['color'], linewidth=2,
                    label=f'V₀ = {case["V"]} м/с, Θ₀ = {case["angle_deg"]:.2f}°')

            # Отмечаем точку падения
            df_ground = df[df['y'] <= 0]
            if len(df_ground) > 0:
                ax.scatter(df_ground.iloc[0]['x'], 0,
                           color=case['color'], s=150, zorder=5, marker='s')

        ax.set_xlabel('$x$, м')
        ax.set_ylabel('$y$, м')
        ax.set_title('Оптимальные траектории полета ЛА, $y(x)$')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')
        ax.axis('equal')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_V_t(self, save_path=None):
        """График зависимости скорости от времени V(t)"""
        fig, ax = plt.subplots(figsize=(20, 10))

        for case in self.sim.optimal_cases:
            df = case['df']
            ax.plot(df['tau'], df['V'],
                    color=case['color'], linewidth=2,
                    label=f'V₀ = {case["V"]} м/с, Θ₀ = {case["angle_deg"]:.2f}°')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$V$, м/с')
        ax.set_title('Зависимость скорости ЛА от времени полета, $V(t)$ (оптимальные углы)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_Theta_c_t(self, save_path=None):
        """График зависимости угла траектории от времени Θ_c(t)"""
        fig, ax = plt.subplots(figsize=(20, 10))

        for case in self.sim.optimal_cases:
            df = case['df']
            ax.plot(df['tau'], df['Theta_c_deg'],
                    color=case['color'], linewidth=2,
                    label=f'V₀ = {case["V"]} м/с, Θ₀ = {case["angle_deg"]:.2f}°')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$\\Theta_c$, град')
        ax.set_title('Зависимость угла траектории от времени, $\\Theta_c(t)$ (оптимальные углы)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_y_t(self, save_path=None):
        """График зависимости высоты от времени y(t)"""
        fig, ax = plt.subplots(figsize=(20, 10))

        for case in self.sim.optimal_cases:
            df = case['df']
            ax.plot(df['tau'], df['y'],
                    color=case['color'], linewidth=2,
                    label=f'V₀ = {case["V"]} м/с, Θ₀ = {case["angle_deg"]:.2f}°')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$y$, м')
        ax.set_title('Зависимость высоты полета ЛА от времени, $y(t)$ (оптимальные углы)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_x_t(self, save_path=None):
        """График зависимости дальности от времени x(t)"""
        fig, ax = plt.subplots(figsize=(20, 10))

        for case in self.sim.optimal_cases:
            df = case['df']
            ax.plot(df['tau'], df['x'],
                    color=case['color'], linewidth=2,
                    label=f'V₀ = {case["V"]} м/с, Θ₀ = {case["angle_deg"]:.2f}°')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$x$, м')
        ax.set_title('Зависимость дальности полета ЛА от времени, $x(t)$ (оптимальные углы)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_theta_t(self, save_path=None):
        """График зависимости угла тангажа от времени θ(t)"""
        fig, ax = plt.subplots(figsize=(20, 10))

        for case in self.sim.optimal_cases:
            df = case['df']
            ax.plot(df['tau'], df['theta'],
                    color=case['color'], linewidth=2,
                    label=f'V₀ = {case["V"]} м/с, Θ₀ = {case["angle_deg"]:.2f}°')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$\\theta$, град')
        ax.set_title('Зависимость угла тангажа от времени, $\\theta(t)$ (оптимальные углы)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_alpha_t(self, save_path=None):
        """График зависимости угла атаки от времени α(t)"""
        fig, ax = plt.subplots(figsize=(20, 10))

        for case in self.sim.optimal_cases:
            df = case['df']
            ax.plot(df['tau'], df['alpha'],
                    color=case['color'], linewidth=2,
                    label=f'V₀ = {case["V"]} м/с, Θ₀ = {case["angle_deg"]:.2f}°')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$\\alpha$, рад')
        ax.set_title('Зависимость угла атаки от времени, $\\alpha(t)$ (оптимальные углы)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_V_x(self, save_path=None):
        """График зависимости скорости от дальности V(x)"""
        fig, ax = plt.subplots(figsize=(20, 10))

        for case in self.sim.optimal_cases:
            df = case['df']
            df_filtered = df[df['x'] >= 0]
            ax.plot(df_filtered['x'], df_filtered['V'],
                    color=case['color'], linewidth=2,
                    label=f'V₀ = {case["V"]} м/с, Θ₀ = {case["angle_deg"]:.2f}°')

        ax.set_xlabel('$x$, м')
        ax.set_ylabel('$V$, м/с')
        ax.set_title('Зависимость скорости ЛА от дальности, $V(x)$ (оптимальные углы)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_Theta_c_x(self, save_path=None):
        """График зависимости угла траектории от дальности Θ_c(x)"""
        fig, ax = plt.subplots(figsize=(20, 10))

        for case in self.sim.optimal_cases:
            df = case['df']
            df_filtered = df[df['x'] >= 0]
            ax.plot(df_filtered['x'], df_filtered['Theta_c_deg'],
                    color=case['color'], linewidth=2,
                    label=f'V₀ = {case["V"]} м/с, Θ₀ = {case["angle_deg"]:.2f}°')

        ax.set_xlabel('$x$, м')
        ax.set_ylabel('$\\Theta_c$, град')
        ax.set_title('Зависимость угла траектории от дальности, $\\Theta_c(x)$ (оптимальные углы)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_omega_z_t(self, save_path=None):
        """График зависимости угловой скорости от времени ω_z(t)"""
        fig, ax = plt.subplots(figsize=(20, 10))

        for case in self.sim.optimal_cases:
            df = case['df']
            ax.plot(df['tau'], df['omega_z'],
                    color=case['color'], linewidth=2,
                    label=f'V₀ = {case["V"]} м/с, Θ₀ = {case["angle_deg"]:.2f}°')

        ax.set_xlabel('$t$, с')
        ax.set_ylabel('$\\omega_z$, с$^{-1}$')
        ax.set_title('Зависимость угловой скорости от времени, $\\omega_z(t)$ (оптимальные углы)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_all(self, save_dir="results/optimization"):
        """Построение всех графиков"""
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ ОПТИМАЛЬНЫХ ТРАЕКТОРИЙ")
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
            ('y(x)', self.plot_trajectories),
            ('omega_z(t)', self.plot_omega_z_t)
        ]

        for name, func in graphs:
            print(f"  Строим и сохраняем: {name}")
            save_path = os.path.join(save_dir, f"{name.replace('(', '_').replace(')', '')}.png")
            func(save_path=save_path)
            plt.close()

        print(f"\n✅ Все графики сохранены в папку: {save_dir}")
        print("=" * 80)


# ==================== ОСНОВНАЯ ЧАСТЬ ====================

print("\n" + "=" * 80)
print("РАСЧЕТ ОПТИМАЛЬНЫХ ТРАЕКТОРИЙ")
print("=" * 80)
print("\nОптимальные углы бросания:")
print("  • Для V₀ = 245 м/с: Θ₀ = 44.20°")
print("  • Для V₀ = 952 м/с: Θ₀ = 47.40°")
print("\n" + "=" * 80)

# Расчет оптимальных траекторий
sim_optimal = Simulation_Optimal(max_steps=15000)

# Построение графиков
plotter = Plotter_Optimal(sim_optimal)
plotter.plot_all(save_dir="results/optimization")

print("\n" + "=" * 80)
print("РАСЧЕТ ОПТИМАЛЬНЫХ ТРАЕКТОРИЙ ЗАВЕРШЕН")
print("=" * 80)