import numpy as np
import pandas as pd
import geopandas as gpd
import os
import random

from configs.static_vars import RoadID, PathName, RoadNumToCN, Months
from configs.static_vars import RootPath, DataPath, RoadDataPath
from utils.data_preprocessing import make_dir, BS_ratio
from utils.visualisation import stacked_plots, plot_sample_rd

rd_features = gpd.read_file(RoadDataPath)
rd_features = rd_features[[RoadID, PathName]]

for month in Months:
    print(month)
    volume_pred = pd.read_csv(os.path.join(DataPath, 'predict/{}_volume_predictions.csv'.format(month)),
                              encoding='utf-8')
    cartype_pred = pd.read_csv(os.path.join(DataPath, 'predict/{}_cartype_predictions.csv'.format(month)),
                               encoding='utf-8')
    final_df = pd.merge(volume_pred, cartype_pred, on=[RoadID, 'year', 'month', 'day', 'weekday', 'hour', 'rdClassEN'])
    final_df.rename(columns={'pred_ratio': '小型车占比', 'pred_vo': '当量', 'rdClassEN': '道路等级'}, inplace=True)
    final_df['道路等级'] = final_df['道路等级'].replace(RoadNumToCN)
    final_df['大型车占比'] = 1 - final_df['小型车占比']
    final_df['小型车流量'] = final_df['当量'] * final_df['小型车占比']
    final_df['大型车流量'] = final_df['当量'] * final_df['大型车占比'] / 2
    final_df = pd.merge(final_df, rd_features, on=RoadID)
    final_df[PathName] = final_df[PathName].fillna(value='NoName')

    BS_ratio(final_df, '大型车流量', '小型车流量')

    # make plot
    img_path = make_dir(RootPath, 'img')
    for k in ['bar', 'line']:
        stacked_plots(final_df, k, 'hour', '大型车流量', '小型车流量', img_path)
        stacked_plots(final_df, k, '道路等级', '大型车占比', '小型车占比', img_path)
        stacked_plots(final_df, k, 'weekday', '大型车占比', '小型车占比', img_path)

    # plot 24h volume by cartype of sampled roads
    rd_sample = random.sample(list(np.unique(final_df[[RoadID]])), 10)
    df_dict = {}
    for count, r in enumerate(rd_sample):
        df = final_df[final_df[RoadID] == r]
        df_dict[count] = {'name': np.unique(df[PathName])[0], 'df': df}
    plot_sample_rd(2, 5, 'hour', df_dict, '大型车流量', '小型车流量', img_path)

    # save final result
    save_final_result = input("Hit 'y' if you want to save the predictions.\nHit 'Enter' to exit.\n")
    if save_final_result == 'y':
        final_df_path = make_dir(DataPath, 'final_df')
        final_df.to_csv(os.path.join(final_df_path, '{}_final_predictions.csv'.format(month)), index=False,
                        encoding='utf-8')
        print(month + ' final predictions saved!')
    else:
        break
