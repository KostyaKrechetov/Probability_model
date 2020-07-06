# -*- coding: utf-8 -*-
import pymc3 as pm
import numpy as np
import pandas as pd
from theano import shared
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import re

az.style.use('arviz-darkgrid')

np.random.seed(1)

# In[] Загрузка данных
file_name = 'Несколько моделей Скважина 4.xlsx'
data = pd.read_excel(file_name)
well_num = str(re.sub('[\D]', '', file_name))

# In[] Формирование обучающей выборки
# (для скважины 4 n = 27, для скважин 1 и 2 - n равен 28)!!!!!!!!!!!!
k = 0
n = 27
time = pd.to_datetime(data.iloc[k:n, 0])
time_zero = time.values[-1]
time_deltas = shared(np.array([abs((elem - time_zero).days) for elem in time]))

Q_real=shared(data.iloc[k:n,1].values)
Q_model1=shared(data.iloc[k:n,2].values)
Q_model2=shared(data.iloc[k:n,3].values)
Q_model3=shared(data.iloc[k:n,4].values)
Q_model4=shared(data.iloc[k:n,5].values)
Q_model5=shared(data.iloc[k:n,6].values)

# In[] Настройка модели
with pm.Model() as model_g:
    w0 = pm.Normal('w0', mu=0, sd=1)
    w1 = pm.Normal('w1', mu=0, sd=1)
    w2 = pm.Normal('w2', mu=0, sd=1)
    w3 = pm.Normal('w3', mu=0, sd=1)
    w4 = pm.Normal('w4', mu=0, sd=1)
    w5 = pm.Normal('w5', mu=0, sd=1)
    w6 = pm.HalfNormal('w6', sd=80)
    w7 = pm.HalfNormal('w7', sd=30)
    # sd = pm.HalfNormal('sd', sd=100)

    mu = pm.Deterministic('mu',
                          w0 + w1 * Q_model1 + w2 * Q_model2 + w3 * Q_model3 +
                          w4 * Q_model4 + w5 * Q_model5)
    sd = pm.Deterministic('sd', w6 + w7 * time_deltas)
    y_pred = pm.Normal('y_pred', mu=mu, sd=sd, observed=Q_real)

    trace_g = pm.sample(2000, tune=1000, target_accept=.93, cores=-1)

# Отображение статистики по параметрами. Хорошо, если распределения гладкие, унимодальные
# az.plot_trace(trace_g, var_names=['w0', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'sd'])

# In[] Проверка модели на периоде обучения
# Генерация сэмплов прогнозов из построенной модели. Нужно для того, чтобы было много сэмплов для построения доверительного интервала
ppc = pm.sample_posterior_predictive(trace_g, samples=1000, model=model_g)
# In[]
# Отрисовка прогноза наиболее вероятного(среднего) прогноза
w0_m = trace_g['w0'].mean()
w1_m = trace_g['w1'].mean()
w2_m = trace_g['w2'].mean()
w3_m = trace_g['w3'].mean()
w4_m = trace_g['w4'].mean()
w5_m = trace_g['w5'].mean()
w6_m = trace_g['w6'].mean()
w7_m = trace_g['w7'].mean()


plt.plot(time, w0_m + w1_m * Q_model1.get_value() + w2_m * Q_model2.get_value() + w3_m * Q_model3.get_value()
         + w4_m * Q_model4.get_value() + w5_m * Q_model5.get_value(),
         c='k', label='Q-наиболее вероятное')

plt.plot(time, Q_real.get_value(), 'C0.')
plt.plot(time, Q_model1.get_value(), 'g--', label='Q_модель1')
plt.plot(time, Q_model2.get_value(), 'b--', label='Q_модель2')
plt.plot(time, Q_model3.get_value(), 'y--', label='Q_модель3')
plt.plot(time, Q_model4.get_value(), 'm--', label='Q_модель4')
plt.plot(time, Q_model5.get_value(), 'c--', label='Q_модель5')

# Отрисовка доверительных интервалов уровнем доверия 50% и 95%
az.plot_hpd(time, ppc['y_pred'], credible_interval=0.5, color='tan', smooth=False)
az.plot_hpd(time, ppc['y_pred'], credible_interval=0.95, color='gray', smooth=False)
plt.rc('legend', fontsize=8)
plt.rc('axes', labelsize=8)  # fontsize of the tick labels
plt.title('Адаптация модели')
plt.legend()
plt.savefig(str(well_num) + ' - обучение.png')
plt.tight_layout()
plt.show()


# In[]
# Оценка сдвига последней точки факта от прогноза. Это сдвиг будет использован для сдвига будущего прогноза
delta = Q_model1.get_value()[n - 1] - (
            w0_m + w1_m * Q_model1.get_value()[n - 1] + w2_m * Q_model2.get_value()[n - 1] + w3_m *
            Q_model3.get_value()[n - 1]
            + w4_m * Q_model4.get_value()[n - 1] + w5_m * Q_model5.get_value()[n - 1])

# In[] Построение модели на периоде прогноза
time = pd.to_datetime(data.iloc[n:, 0])
# Переопределение входных параметров модели
Q_model1.set_value(data.iloc[n:, 2])
Q_model2.set_value(data.iloc[n:, 3])
Q_model3.set_value(data.iloc[n:, 4])
Q_model4.set_value(data.iloc[n:, 5])
Q_model5.set_value(data.iloc[n:, 6])
time_deltas.set_value(np.array([abs((elem - time_zero).days) for elem in time]))

# In[]
# Генерация сэмплов прогнозов из построенной модели. Нужно для того, чтобы было много сэмплов для построения доверительного интервала
ppc2 = pm.sample_posterior_predictive(trace_g, samples=1000, model=model_g)


plt.plot(time, delta + w0_m + w1_m * Q_model1.get_value() + w2_m * Q_model2.get_value() +
         w3_m * Q_model3.get_value() + w4_m * Q_model4.get_value() + w5_m * Q_model5.get_value(),
         c='k', label='Q-наиболее вероятное')

plt.plot(time, data.iloc[n:, 1].values, 'C0.')
plt.plot(time, Q_model1.get_value(), 'g--', label='Q_модель1')
plt.plot(time, Q_model2.get_value(), 'b--', label='Q_модель2')
plt.plot(time, Q_model3.get_value(), 'y--', label='Q_модель3')
plt.plot(time, Q_model4.get_value(), 'm--', label='Q_модель4')
plt.plot(time, Q_model5.get_value(), 'c--', label='Q_модель5')

# Отрисовка доверительных интервалов уровнем доверия 50% и 95%
az.plot_hpd(time, ppc2['y_pred'] + delta, credible_interval=0.5, color='tan', smooth=False)
az.plot_hpd(time, ppc2['y_pred'] + delta, credible_interval=0.95, color='gray', smooth=False)
plt.rc('legend', fontsize=8)
plt.rc('axes', labelsize=8)  # fontsize of the tick labels

plt.title('Прогноза модели')
plt.legend()
plt.tight_layout()
plt.savefig(str(well_num) + ' - прогноз.png')
plt.show()


