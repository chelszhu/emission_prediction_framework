#################### STEP 2: Match geocoded points with cleaned volume to road spatial data. ####################

import numpy as np
import geopandas as gpd
import pandas as pd
import os

from configs.static_vars import DataPath, RoadDataPath
from configs.static_vars import RoadID, EPSG
from utils.data_preprocessing import assign_pnt_to_road


pnts = gpd.read_file(os.path.join(DataPath, 'geocoded_pnt.geojson'), encoding='utf-8')
roads = gpd.read_file(RoadDataPath, encoding='utf-8')


########## 点位落位到道路 ##########
pnts_on_rd = assign_pnt_to_road(pnts, roads, EPSG, RoadID)
pnts_on_rd.to_file(os.path.join(DataPath, 'pnts_on_road.geojson', driver='GeoJSON'))

########## 流量落位到点位 ##########
# processed final product should include fields: rd_id, year, month, day, weekday, hour,
# volume_small_car, volume_big_car[, season, week_number]


