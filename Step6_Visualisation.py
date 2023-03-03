import numpy as np
import pandas as pd
import geopandas as gpd
import os

from configs.static_vars import EPSG, DataPath, \
    RoadDataPath, RoadID, PathName, \
    RegionDataPath, RegionID, \
    NoxDataPath, Nox1Col, Nox2Col, NoxTotalCol
from utils.visualisation import make_grid, GridBasedMap, RegionBasedMap, RoadBasedMap

##### Make directory to store outputs #####
output_path = os.path.join(DataPath, 'vis')
if not os.path.exists(output_path):
    os.mkdir(output_path)


def save_geojson(file, name):
    file.to_file(os.path.join(output_path, '{}.geojson'.format(name)), driver='GeoJSON', encoding='utf-8')
    print('Saved!')
    print('----------------------------------------')


##### Read data and convert to local projected crs #####
nox = pd.read_csv(NoxDataPath, encoding='gb18030')
region = gpd.read_file(RegionDataPath)
roads = gpd.read_file(RoadDataPath, encoding='utf-8')
roads = roads.to_crs(EPSG)

##### Make grid based on road layer and cut roads with grids #####
grid = make_grid(roads, 1000, 1000, EPSG)
grid.to_file(os.path.join(output_path, 'grid.geojson'), driver='GeoJSON')

##### Create an instance of GridBasedMap #####
# gridBasedMap = GridBasedMap(grid, roads, nox, Nox1Col, Nox2Col, NoxTotalCol, RoadID, 'grid_id', EPSG)
#
# quxianBasedMap = RegionBasedMap(region, roads, nox, Nox1Col, Nox2Col, NoxTotalCol, RoadID, RegionID, EPSG)
# year_grid = quxianBasedMap.year_total()
# save_geojson(year_grid, '区县年化排放量')

by_pathname = RoadBasedMap(roads.dissolve(by=PathName), PathName, nox, RoadID, Nox1Col, Nox2Col, NoxTotalCol)
year_total_by_pathname = by_pathname.year_total()
save_geojson(year_total_by_pathname, '道路名年化排放量')

# ##### 年化排放 #####
# print('计算年化排放：')
# year_grid = gridBasedMap.year_total()
# save_geojson(year_grid, '网格年化排放量')
#
# ##### 典型工作日/双休日日均排放 #####
# print('计算日均排放：')
# day_grid = gridBasedMap.day_average()
# save_geojson(day_grid, '网格日均排放量')
#
# ##### 典型工作日/双休日高峰小时排放 #####
# print('计算高峰小时排放：')
# peak_grid = gridBasedMap.peakhour_average()
# save_geojson(peak_grid, '网格高峰小时排放量')
#
# ##### 典型工作日/双休日24h平均排放 #####
# print('计算时均排放：')
# for weekday in [True, False]:
#     hour_average_grid = gridBasedMap.hour_average_dynamic(weekday=weekday, year=2021, month=7, day=1)
#     d = '工作日' if weekday else '双休日'
#     save_geojson(hour_average_grid, '基于道路典型日排放_{}'.format(d))
