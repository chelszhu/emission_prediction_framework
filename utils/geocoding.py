import pymongo
import requests
import warnings
warnings.filterwarnings('ignore')
import datetime
today_str = str(datetime.datetime.now()).split(".")[0].replace(":", "_")
from collections import Counter
from copy import deepcopy
import pandas as pd
import geopandas as gpd
import json
from functools import wraps
import time
from shapely.geometry import Point
from . import coordinate_transform as ct



def retry(howmany, return_v):
    def tryIt(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < howmany:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(e)
                    print("retrying")
                    attempts += 1
                    time.sleep(1)
            return return_v
        return wrapper
    return tryIt


@retry(5, "nodata")
def get_result(url):
    s = requests.session()
    s.keep_alive = False
    requests.adapters.DEFAULT_RETRIES = 5
    response = requests.get(url)
    return response


def get_coordinates_from_baidu(locs, baidu_ak, **kwargs):

    address_df = pd.DataFrame({'address': locs})

    address_df["lat_baidu"] = ""
    address_df["lng_baidu"] = ""
    address_df["precise"] = ""
    address_df["confidence"] = ""
    address_df["comprehension"] = ""
    address_df["level"] = ""
    address_df["状态"] = ""
    address_df['full_address'] = address_df['address']

    if 'city' in kwargs:
        address_df['full_address'] = kwargs.get('city') + address_df['full_address']

    if 'province' in kwargs:
        address_df['full_address'] = kwargs.get('province') + address_df['full_address']



    for i, row in address_df.iterrows():
        
        address = row['full_address']
            
        geocoding_url = "http://api.map.baidu.com/geocoding/v3/?address={}&output=json&ak={}".format(address, baidu_ak)
        response = get_result(geocoding_url)
        
        # The HTTP 200 OK success status response code indicates that the request has succeeded.     
        if response.status_code == 200:
      
            data = response.json()

            try:
                coordinates = data["result"]['location']
                lng_baidu = coordinates['lng']
                lat_baidu = coordinates['lat']
                
                precise = data["result"]['precise']
                confidence = data["result"]['confidence']
                comprehension = data["result"]['comprehension']
                level = data["result"]['level']

                lat_wgs, lng_wgs = ct.bd2wgs(lat_baidu, lng_baidu)
                
                # update the result
                address_df.at[i, "lat_baidu"] = lat_baidu
                address_df.at[i, "lng_baidu"] = lng_baidu
                address_df.at[i, "lat_wgs"] = lat_wgs
                address_df.at[i, "lng_wgs"] = lng_wgs
                address_df.at[i, "precise"] = precise
                address_df.at[i, "confidence"] = confidence
                address_df.at[i, "comprehension"] = comprehension
                address_df.at[i, "level"] = level
                address_df.at[i, "状态"] = "地址查询成功"
            
            except:
                address_df.at[i, "状态"] = "地址查询无结果"

        else:
            address_df.at[i, "状态"] = "地址查询无结果"

    print("Address Queries Completed!")

    address_gdf = gpd.GeoDataFrame(address_df, geometry=gpd.points_from_xy(address_df.lng_wgs, address_df.lat_wgs), crs=4326)

    return address_gdf




