# Подключаем все необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats as st
# Время обучения модели
import timeit
# Тренды и сезонность
from statsmodels.tsa.seasonal import seasonal_decompose
# Проверка на стационарность
from statsmodels.tsa.stattools import adfuller, kpss
# Проверка на дисперсию с помощью теста Андерсона-Дарлинга
from scipy.stats import anderson

from lightgbm import LGBMRegressor
# Вызов библиотеки для отключения предупреждения
import warnings

# Разбиение на обучающую, валидационную и тестовую выборку
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
# Применим кроссвалидацию для повышения качеств обучения
# Для константной модели
from sklearn.dummy import DummyRegressor
from catboost import CatBoostRegressor

# Масштабируемость модели
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

# и для машинного обучения разными способами (по условию мы выбираем линейную регрессию):
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    # Точность модели
    accuracy_score,
    # Матрицы ошибок (для борьбы с дисбалансом)
    confusion_matrix,
    # Полнота
    recall_score,
    # Точность
    precision_score,
    # F1-мера
    f1_score,
    # Метрика AUC-ROC
    roc_auc_score,
    roc_curve,
    # MSE
    mean_squared_error,
    mean_absolute_error,
    fbeta_score,
    make_scorer
)

# Контроль выборки
from sklearn.utils import shuffle



df = pd.read_csv('taxi.csv', index_col=[0], parse_dates=[0])

df.info()

# Отсортируем данные на всякий
df.sort_index(inplace=True)

# Ресемплирируем данные по часами
df = df.resample('1H').sum()

# Создадим класс, который автоматически будет добиваться минимального RMSE
class search_small_rmse:
    # По умолчанию создается, если ничего не передаем аргументы
    def __init__(self, data, column, name_model, parameters, cv_size, name_scoring, name_features, name_target, test_size, random_state):
        self.data = data
        self.rmse_v = 1_000_000
        self.max_lag = 1
        self.rolling_mean_size = 1
        self.name_model = name_model
        self.parameters = parameters
        self.cv_size = cv_size
        self.name_scoring = name_scoring
        self.name_features = name_features
        self.name_target = name_target
        self.test_size = test_size
        self.random_state = random_state
        self.name_column = column

    # Создадим признаки, чтобы правильно обучить модель
    def make_features(self, max_lag, rolling_mean_size):
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month
        self.data['day'] = self.data.index.day
        self.data['dayofweek'] = self.data.index.dayofweek

        for lag in range(1, max_lag + 1):
            self.data[f'lag_{lag}'] = self.data[self.name_column].shift(lag)

        self.data['rolling_mean'] = self.data[self.name_column].shift().rolling(rolling_mean_size).mean()

    # Разбиение на обучающие и тестовые выборки
    def make_train_test_split(self):
        # Разделим обучающую и тестовую выборку
        self.train, self.test = train_test_split(self.data, test_size=self.test_size, random_state=self.random_state, shuffle=False)
        self.train = self.train.dropna()

        self.features_train = self.train.drop(self.name_features, axis=1)
        self.target_train = self.train[self.name_target]

        self.features_test = self.test.drop(self.name_features, axis=1)
        self.target_test = self.test[self.name_target]

    # Здесь будет бесконечная итерация, которая добивается введенного желаемого RMSE
    def striving_rmse(self, desired_rmse, max_value_lag, max_value_rol_mean_size):
        self.desired_rmse = desired_rmse
        self.max_value_lag = max_value_lag
        self.max_value_rol_mean_size = max_value_rol_mean_size

        for max_lag in range(1, max_value_lag):
            for rolling_mean_size in range(1, max_value_rol_mean_size):
                # Создаем новые признаки
                self.make_features(max_lag, rolling_mean_size)

                # Разбиваем на обучающие и тестовые выборки
                self.make_train_test_split()

                # Проверим
                print(self.features_train.shape)
                print(self.features_test.shape)

                # Инициализируем модель
                self.model = GridSearchCV(self.name_model, param_grid=self.parameters, cv=self.cv_size, scoring=self.name_scoring)

                self.learning_model()
                self.predicting_model()

                print(f'{max_lag}:{rolling_mean_size}; RMSE TRAIN:', self.rmse_t)
                print(f'{max_lag}:{rolling_mean_size}; RMSE TEST:', self.rmse_v)

                if (self.rmse_v < self.desired_rmse):
                    print('Successfully')
                    print('BEST RMSE:', self.rmse_v)
                    break

        print('NO successfully')
        print('RMSE:', self.rmse_v)

    # Обучим модель на обучающей выборке
    def learning_model(self):
        self.model.fit(self.features_train, self.target_train)
        self.rmse_t = -self.model.best_score_
        self.refit_time_t = -self.model.refit_time_

    # Получим предсказания на тестовой выборки
    def predicting_model(self):
        start_time = timeit.default_timer()
        self.predictions = self.model.predict(self.features_test)
        self.elapsed = round(timeit.default_timer() - start_time, 3)
        self.rmse_v = mean_squared_error(self.target_test, self.predictions, squared=False)


# На всякий случай сделаем копию, чтобы было удобно организовать несколько подборов
df_copy = df.copy()

# Объявляем класс
df_class = search_small_rmse(df_copy, 'num_orders', LGBMRegressor(), {'num_leaves': [5, 10],
              'learning_rate': [0.1, 0.3],
              'max_depth': [3, 5],
              'n_estimators': [10, 25]}, 5, 'neg_root_mean_squared_error', 'num_orders', 'num_orders', 0.1, 12345)

# Создаем новые признаки
df_class.striving_rmse(48, 48, 96)


# Разбиваем на обучающие и тестовые выборки
# df_class.make_train_test_split(name_features='num_orders', name_target='num_orders', test_size=0.1, random_state=12345)

# Обучаем и получаем rmse
# df_class.striving_rmse(name_model=LinearRegression(), parameters={}, cv_size=5, name_scoring='neg_root_mean_squared_error', desired_rmse=48, max_value_lag=24, max_value_rol_mean_size=48)
