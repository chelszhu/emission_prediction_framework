import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection, metrics, preprocessing, pipeline, ensemble
import matplotlib.pyplot as plt


def train_test_split(X, y, test_size):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=26)
    print('test size: {}'.format(test_size))
    print('Training Data Shape:', X_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Data Shape:', X_test.shape)
    print('Testing Labels Shape:', y_test.shape)
    return X_train, X_test, y_train, y_test


def trainPipe(X, y, n_estimators=100):
    # Instantiate model with 1000 decision trees
    rf = ensemble.RandomForestRegressor(n_estimators=n_estimators, n_jobs=7, verbose=1)
    pipe = pipeline.Pipeline(steps=[('std_scl', preprocessing.StandardScaler()),
                                    ('model', rf)])

    # Train the model on training data
    pipe.fit(X, y)
    pipe_rf = pipe['model']

    return pipe, pipe_rf


def evaluatePipe(pipe, X, y):
    predictions = pipe.predict(X)
    name = list(y.columns)[0]

    mape = metrics.mean_absolute_percentage_error(y, predictions)
    print('Mean absolute percentage error:', round(mape, 3))
    msle = metrics.mean_squared_log_error(y, predictions)
    print('Mean squared log error:', round(msle, 3))
    rmse = metrics.mean_squared_error(y, predictions, squared=False)
    print('Root mean squared error:',
          round(rmse, 3))
    r2 = metrics.r2_score(y, predictions)
    print('R2 score:', round(r2, 3))

    y_copy = y.copy(deep=True)
    y_copy['pred_{}'.format(name)] = predictions

    return y_copy

