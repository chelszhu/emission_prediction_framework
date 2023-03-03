import os
from datetime import datetime

# city
PROVINCE = '山东省'
CITY = '济南市'
EPSG = 2334

RootPath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DataPath = os.path.join(RootPath, 'data')

## road
RoadDataName = 'featured_road4.shp'
RoadDataPath = os.path.join(DataPath, os.path.join('featured_road', RoadDataName))
RoadID = 'OBJECTID'
PathName = 'PathName'
RoadStrToNum = {'rd00': 6, 'rd01': 5, 'rd02': 4, 'rd03': 3, 'rd04': 2, 'rd05': 1}
RoadNumToCN = {6: '高速公路', 5: '国道', 4: '都市高速环线', 3: '省道', 2: '县道', 1: '乡镇道路'}
## region
RegionDataName = 'qquxian.shp'
RegionDataPath = os.path.join(DataPath, os.path.join('jinan_quxian_jiedao', RegionDataName))
RegionID = '县'
## emission
NoxDataName = 'nox_data.csv'
NoxDataPath = os.path.join(DataPath, os.path.join('nox', NoxDataName))
Nox1Col = 'sp1'
Nox2Col = 'sp2'
NoxTotalCol = 'sp_total'


# time dimension
MonthDict = {'september': {'year': '2020', 'month': '09'},
              'november': {'year': '2020', 'month': '11'},
              'march': {'year': '2021', 'month': '03'},
              'july': {'year': '2021', 'month': '07'}}

# Step1: Geocoding
BaiduAK = "VgtmjVws37LTLCDmkNa6Q5rmlzp3m3GK"

# Step2: Spatial Matching

# Step3: Prepare Training Data
## Do Principal Component Analysis on road features or not.
## PCA on continuous variables only.
PCARoad = True
Months = ['september']
##

# Step4: Modeling
## Train volume model or cartype model
ModelName = 'volume'
## test data portion
TestSize = 0.2
## number of estimators for random forest
NumEstimators = 300
## Whether to predict or not; predict if training results are satisfactory
Predict = False
## Whether to save model or not
SaveModel = False

# Step5: Results

# Step6: Visualisation

# quxian = ['古里镇',
#  '沙家浜镇',
#  '董浜镇',
#  '常熟经济技术开发区',
#  '梅李镇',
#  '辛庄镇',
#  '支塘镇',
#  '虞山镇',
#  '江苏省常熟高新技术产业开发区',
#  '尚湖镇',
#  '江苏常熟服装城',
#  '常熟虞山尚湖旅游度假区',
#  '虞山林场',
#  '碧溪街道',
#  '海虞镇']

## query time frame ##

# start = datetime(2021, 3, 1)
# end = datetime(2021, 5, 1)

