import numpy as np
import pandas as pd
import geopandas as gpd
import math
import os
import shapely
from shapely.geometry import Point, LineString, MultiPoint


def make_dir(root, name):
    path = os.path.join(root, name)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

# def create_rectangle(pnt, height, width, direction):
#     geometry = pnt.geometry.item()
#     if direction == 'up':
#         yoff = shapely.affinity.translate(geom=geometry, yoff=height)
#         xleft = shapely.affinity.translate(geom=geometry, xoff=-width / 2)
#         xright = shapely.affinity.translate(geom=geometry, xoff=width / 2)
#         return MultiPoint([yoff, xleft, xright]).envelope
#     elif direction == 'down':
#         yoff = shapely.affinity.translate(geom=geometry, yoff=-height)
#         xleft = shapely.affinity.translate(geom=geometry, xoff=-width / 2)
#         xright = shapely.affinity.translate(geom=geometry, xoff=width / 2)
#         return MultiPoint([yoff, xleft, xright]).envelope
#     elif direction == 'right':
#         xoff = shapely.affinity.translate(geom=geometry, xoff=width)
#         yup = shapely.affinity.translate(geom=geometry, yoff=height / 2)
#         ydown = shapely.affinity.translate(geom=geometry, yoff=-height / 2)
#         return MultiPoint([xoff, yup, ydown]).envelope
#     else:
#         xoff = shapely.affinity.translate(geom=geometry, xoff=-width)
#         yup = shapely.affinity.translate(geom=geometry, yoff=height / 2)
#         ydown = shapely.affinity.translate(geom=geometry, yoff=-height / 2)
#         return MultiPoint([xoff, yup, ydown]).envelope


def create_rectangles(pnt, height, width, direction):
    geometry = pnt.geometry.item()

    if direction == 'verticle':
        yup = shapely.affinity.translate(geom=geometry, yoff=height)
        ydown = shapely.affinity.translate(geom=geometry, yoff=-height)
        xleft = shapely.affinity.translate(geom=geometry, xoff=-width / 2)
        xright = shapely.affinity.translate(geom=geometry, xoff=width / 2)
        return MultiPoint([yup, ydown, xleft, xright]).envelope
    elif direction == 'horizontal':
        xleft = shapely.affinity.translate(geom=geometry, xoff=-width)
        xright = shapely.affinity.translate(geom=geometry, xoff=width)
        yup = shapely.affinity.translate(geom=geometry, yoff=height / 2)
        ydown = shapely.affinity.translate(geom=geometry, yoff=-height / 2)
        return MultiPoint([xleft, xright, yup, ydown]).envelope


def assign_pnt_to_road(pnts, roads, projected_crs, RoadID):
    pnts = pnts.to_crs(projected_crs)
    roads = roads.to_crs(projected_crs)

    pnts = pnts.replace({'direction': {'南北': 'NS', '东西': 'EW'}})

    for i, row in pnts.iterrows():

        pnt = gpd.GeoDataFrame(row).T
        buffer = gpd.GeoDataFrame(geometry=pnt.buffer(100), crs=projected_crs)
        rd_in_buffer = gpd.sjoin(roads, buffer, how='inner', op='intersects')

        if pnt['direction'].item() == 'NS':
            direc_poly = create_rectangles(pnt, 50, 20, 'verticle')
        else:
            direc_poly = create_rectangles(pnt, 20, 50, 'horizontal')

        direc_poly = gpd.GeoDataFrame({'geometry': [direc_poly]}, crs=projected_crs)
        clipped_rd = gpd.clip(rd_in_buffer, direc_poly)
        if clipped_rd.empty:
            continue
        rdidx = clipped_rd.length.idxmax()
        pnts.at[i, RoadID] = rd_in_buffer.loc[rdidx, RoadID]
        pnts.at[i, 'geometry'] = clipped_rd.loc[[rdidx]].representative_point().item()

    pnts = pnts.to_crs(4326)

    return pnts


def get_season(month):
    """Return season number according to the lunar calendar.
    Keyword arguments:
    	month -- month number, int"""
    season_dict = {'spring': 0,
                   'summer': 1,
                   'autumn': 2,
                   'winter': 3}
    if month in [3, 4, 5]:
        season = 'spring'
    elif month in [6, 7, 8]:
        season = 'summer'
    elif month in [9, 10, 11]:
        season = 'autumn'
    elif month in [12, 1, 2]:
        season = 'winter'
    else:
        return "month number not in range, check your data"
    return season_dict[season]


def get_weekday(date):
    """
	Returns the weekday
	date: datetime object
	"""
    return date.dayofweek


def get_week_number(date):
    """
	Returns the week number w.r.t. the month
	date: datetime object
	"""
    n = date.isocalendar()[1] - date.replace(day=1).isocalendar()[1]
    return n


def generate_predict_df(rdids, rdid_col, days, year, month):
    all_rd = pd.DataFrame({rdid_col: rdids})
    all_rd['merge'] = 1
    all_day = pd.DataFrame({'day': days})
    all_day['merge'] = 1
    all_hr = pd.DataFrame({'hour': np.arange(24)})
    all_hr['merge'] = 1
    predict_df = pd.merge(all_rd, all_day, on='merge')
    predict_df = pd.merge(predict_df, all_hr, on='merge').drop(columns={'merge'})
    predict_df['year'] = year
    predict_df['month'] = month
    dt = pd.to_datetime(predict_df[['year', 'month', 'day']])
    predict_df['season'] = get_season(month)
    predict_df['week'] = dt.apply(lambda x: get_week_number(x))
    predict_df['weekday'] = dt.apply(lambda x: get_weekday(x))

    return predict_df


def BS_ratio(df, b_col, s_col):

    b = df[b_col].sum()
    s = df[s_col].sum()
    print(b_col, b)
    print(s_col, s)
    print('大型汽车流量占比', b/(s+b))

    return
