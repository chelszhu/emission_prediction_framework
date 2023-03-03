#################### STEP 4: Train Models. ####################

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import json
import pickle
from time import gmtime, strftime
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from engine import modeling
from configs.static_vars import DataPath, RootPath
from configs.static_vars import RoadID, Months, MonthDict, ModelName, TestSize, NumEstimators, Predict, SaveModel
from engine.modeling import train_test_split
from utils.visualisation import plot_feature_importance, plot_truth_pred
from utils.data_preprocessing import make_dir

version = strftime("%m%d%H%M", gmtime())
print('MODELING ' + ModelName)

########## Read and prepare dataset ##########
lst = []
for month in Months:
    data_name = (MonthDict[month])
    y, m = data_name['year'], data_name['month']
    df = pd.read_csv(os.path.join(DataPath, 'train/train_{}{}.csv').format(y, m))
    lst.append(df)
    del df

model_df = pd.concat(lst)
del lst

# calculate y
if ModelName == 'volume':
    model_df["vo"] = model_df["小型汽车号牌"] + 2 * model_df["大型汽车号牌"]
    predict_variable = 'vo'
if ModelName == 'cartype':
    model_df["ratio"] = model_df["小型汽车号牌"] / (model_df["小型汽车号牌"] + 2 * model_df["大型汽车号牌"])
    model_df = model_df.dropna()
    predict_variable = 'ratio'

# exclude unnecessary variables; make sure to exclude y, 'vo' in this case
exclude_col_list = [RoadID, '小型汽车号牌', '大型汽车号牌', 'year', 'month', 'day', predict_variable]
train_variables = [x for x in list(model_df) if x not in exclude_col_list]

model_df_sample = model_df[model_df['cong'] > 1].sample(100000)

# split train and test
X_train, X_test, y_train, y_test = train_test_split(model_df_sample[train_variables], model_df_sample[[predict_variable]], test_size=TestSize)
# train
pipe, model = modeling.trainPipe(X_train, y_train, n_estimators=NumEstimators)
# evaluate
predict_df = modeling.evaluatePipe(pipe, X_test, y_test)

# calculate permutation importance
result = permutation_importance(
    model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(
    result.importances[sorted_idx].T, vert=False, labels=X_test.columns[sorted_idx]
)
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()
plt.savefig(os.path.join(make_dir(RootPath, 'img'), 'PermutationImportance_{}_v{}.png'.format('volume', version)), bbox_inches="tight", dpi=500)
########## Make plots and save ##########
# img_path = make_dir(RootPath, 'img')
# plot_feature_importance(model, train_variables, img_path, ModelName, version)
# plot_truth_pred(predict_df, predict_variable, img_path, ModelName, version)

########## Predict requested data if training results are satisfactory ##########
if Predict:
    for month in Months:
        print()
        print('predicting:', month)
        data_name = (MonthDict[month])
        y, m = data_name['year'], data_name['month']
        predict_input = pd.read_csv(os.path.join(DataPath, 'predict/predict_input_{}{}.csv'.format(y, m)), nrows=9999)

        predict_input['pred_{}'.format(predict_variable)] = pipe.predict(predict_input[train_variables])
        keep_cols = [RoadID, 'year', 'month', 'day', 'weekday', 'hour', 'rdClassEN', 'pred_{}'.format(predict_variable)]
        predict_input[keep_cols].to_csv(os.path.join(DataPath, 'predict/{}_{}_predictions.csv'.format(month, ModelName)), index=False)
        print(month + ' ' + ModelName + ' predictions saved!')

        # rd_features = gpd.read_file(RoadDataPath)

########## Save model and training variables; this may take a while ##########
if SaveModel:
    model_output_path = make_dir(RootPath, 'model')
    with open(os.path.join(model_output_path, '{}_training_variables.json'.format(ModelName)), 'w') as fp:
        json.dump({'variables': train_variables}, fp)

    pickle.dump(pipe, open(os.path.join(model_output_path, "{}_pipe.p".format(ModelName)), 'wb'))