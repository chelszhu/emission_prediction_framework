#################### STEP 1: Geocode camera locations. ####################
import numpy as np
import pandas as pd
import os

from configs.static_vars import DataPath
from configs.static_vars import PROVINCE, CITY
from configs.static_vars import BaiduAK
from utils.geocoding import get_coordinates_from_baidu

raw_vol_data_path = os.path.join(DataPath, 'raw_volume')

########## Get a list of unique intersections #####v
locs_set = set()
for filename in os.listdir(raw_vol_data_path):
    if filename.endswith('.xls') or filename.endswith('.xlsx'):
        print('Reading ' + filename)
        file_path = os.path.join(raw_vol_data_path, filename)
        df_dict = pd.read_excel(file_path, sheet_name=None)
        df = pd.concat(list(df_dict.values()))

        locs = np.unique(df['卡口名称'])
        locs_set.update(locs)

unique_locs = list(locs_set)

########## Geocoding ##########
address_gdf = get_coordinates_from_baidu(unique_locs, BaiduAK, province=PROVINCE, city=CITY)

########## Save file ##########
# address_df.to_csv(os.path.join(root_path, 'intersection_address.csv'), index=False, encoding='gb18030')
address_gdf.to_file(os.path.join(DataPath, 'intersection_address.geojson'), driver='GeoJSON', encoding='utf-8')

#################### NEXT STEP: Check if coordinates are reliable & do manual correction. ####################

