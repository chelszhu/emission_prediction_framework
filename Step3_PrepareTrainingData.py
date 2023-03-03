import pandas as pd
import geopandas as gpd
import numpy as np
import os
from sklearn import preprocessing, decomposition

from configs.static_vars import RoadID, PathName, MonthDict, RoadStrToNum, PCARoad, Months
from configs.static_vars import DataPath, RoadDataPath
from utils.data_preprocessing import get_season, get_week_number, get_weekday, generate_predict_df, make_dir, BS_ratio


train_output_path = make_dir(DataPath, 'train')
predict_output_path = make_dir(DataPath, 'predict')


########## Read and preprocess road features ##########
rd_df = gpd.read_file(RoadDataPath, encoding='utf-8').fillna(0)
rd_ids = list(rd_df[RoadID])
rd_df['rdClassEN'] = rd_df['rdClass'].replace(RoadStrToNum).fillna(0)

# exclude some definitely not relevant columns
exclude_col_list = ["index", "intersection", "LaneNum", "uniqueID", "tag", "PathName", "rdClass", "CarNum2", "quxian",
                    "Shape_Leng", "geometry"]
select_col_list = [x for x in list(rd_df) if x not in exclude_col_list]
rd_df = rd_df[select_col_list]

# do pca on rd_df
# TODO: rdclassEN
# if PCARoad:
#     scaler = preprocessing.MinMaxScaler()
#     scaler.fit(rd_df.drop(columns={RoadID, 'rdClassEN'}))
#     pca = decomposition.PCA(n_components=2)
#     components = pca.fit_transform(scaler.transform(rd_df.drop(columns={RoadID, 'rdClassEN'})))
#     n_components = np.arange(components.shape[1])
#     columns = [str(c) for c in n_components]
#     rd_new = pd.DataFrame(components, index=rd_df[RoadID].values, columns=columns).add_prefix('road').reset_index().rename(columns={'index': RoadID})
#     rd_df = pd.merge(rd_new, rd_df[[RoadID, 'rdClassEN']], on=RoadID)

if PCARoad:
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(rd_df.drop(columns={RoadID}))
    pca = decomposition.PCA(n_components=2)
    components = pca.fit_transform(scaler.transform(rd_df.drop(columns={RoadID})))
    n_components = np.arange(components.shape[1])
    columns = [str(c) for c in n_components]
    rd_new = pd.DataFrame(components, index=rd_df[RoadID].values, columns=columns).add_prefix('road').reset_index().rename(columns={'index': RoadID})
    rd_df = pd.merge(rd_new, rd_df[[RoadID]], on=RoadID)


for month in Months:
    print(month + ':')
    data_name = (MonthDict[month])
    y, m = data_name['year'], data_name['month']

    ########## Read volume and congestion data ##########
    vol = pd.read_csv(os.path.join(DataPath, 'processed_volume/{}{}.csv'.format(y, m)))
    cong = pd.read_csv(os.path.join(DataPath, 'congestion/congestion_{}_{}.csv'.format(y, m)))
    vol['season'] = vol['month'].apply(lambda x: get_season(x))
    vol['week'] = pd.to_datetime(vol[['year', 'month', 'day']]).apply(lambda x: get_week_number(x))

    ########## Merge volume, congestion, road features on ROADID ##########
    rd_cong = pd.merge(rd_df, cong, on=RoadID)
    model_df = pd.merge(rd_cong, vol, on=[RoadID, 'year', 'month', 'day', 'weekday', 'hour'])

    # filter rows with no volume but congestion not smallest
    model_df['vo'] = model_df["小型汽车号牌"] + 2 * model_df["大型汽车号牌"]
    problematic = model_df[(model_df['vo'] == 0) & (model_df['cong'] > 1)]
    col_keys = [RoadID, 'day', 'month']
    keys = list(problematic[col_keys].columns.values)
    i1 = model_df.set_index(keys).index
    i2 = problematic[col_keys].set_index(keys).index
    model_df = model_df[~i1.isin(i2)]

    training_exclude_list = ['vo', 'index', '卡口名称', '卡口方向', '分析时间', '交通流量',
                             'direction', 'ratio1', 'ratio2', '其它号牌', 'KKMC1', 'rd_id', 'vo', 'norm_cong',
                             'factor', 'PntID', 'intersection', 'Unnamed: 0', 'geometry', 'rdClass', PathName]
    col_list = list(model_df)
    training_list = [x for x in col_list if x not in training_exclude_list]

    BS_ratio(model_df, '大型汽车号牌', '小型汽车号牌')

    ########## Save model dataframe ##########
    model_df[training_list].to_csv(os.path.join(train_output_path, 'train_{}{}.csv'.format(y, m)), index=False,
                                   encoding='utf-8')

    ########## Prepare predict dataset ##########
    predict_df = generate_predict_df(rd_ids, RoadID, np.unique(model_df['day']), int(y), int(m))
    predict_df = pd.merge(predict_df, rd_cong, on=[RoadID, 'year', 'month', 'day', 'weekday', 'hour'])
    predict_df.to_csv(os.path.join(predict_output_path, 'predict_input_{}{}.csv'.format(y, m)), index=False, encoding='utf-8')

    del model_df
    del predict_df
    del rd_cong
    del cong

