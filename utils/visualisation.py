import pandas as pd
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns


def make_grid(rd_gdf, length, width, projected_crs):
    """Make grids bounded by the extent of rd_gdf with length and width."""

    # Make sure GeoDataFrame is in projected coordinate system
    rd_gdf = rd_gdf.to_crs(projected_crs)

    xmin, ymin, xmax, ymax = rd_gdf.total_bounds

    cols = list(np.arange(xmin, xmax + width, width))
    rows = list(np.arange(ymin, ymax + length, length))

    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x, y), (x + width, y), (x + width, y + length), (x, y + length)]))

    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=projected_crs)
    grid['grid_id'] = grid.index + 1
    # grid = grid.to_crs(4326)

    return grid


def distribute_nox(base_layer_gdf, rd_gdf, nox_df, nox1_col, nox2_col, nox_col, rd_id, layer_id, projected_crs):
    """Cut rd_gdf with base_layer_gdf and distribute emission data based on segment length"""
    base_layer_gdf = base_layer_gdf.to_crs(projected_crs)
    rd_gdf = rd_gdf.to_crs(projected_crs)

    gridded_rd = gpd.overlay(rd_gdf[[rd_id, 'geometry']], base_layer_gdf)

    # Distribute emission based on length ratio
    rd_gdf['total_length'] = rd_gdf.length
    gridded_rd['split_length'] = gridded_rd.length
    gridded_rd = gridded_rd.merge(rd_gdf[[rd_id, 'total_length']], on=rd_id)
    gridded_rd['ratio'] = gridded_rd['split_length'] / gridded_rd['total_length']

    gridded_nox = pd.merge(gridded_rd[[rd_id, layer_id, 'ratio']], nox_df, on=rd_id)
    for col in [nox1_col, nox2_col, nox_col]:
        gridded_nox[col] = gridded_nox[col] * gridded_nox['ratio']

    return gridded_nox


class LayerBasedMap:
    """Generate layer-based GeoDataFrame for visualisation.Typical base layers include roads, grids, or regions."""

    def __init__(self, base_layer_gdf: gpd.GeoDataFrame, gridded_nox_df: pd.DataFrame, nox1_col: str, nox2_col: str,
                 nox_col: str, smallest_unit: list, group_id: str):
        """
        Keyword arguments:
            base_layer_gdf
            gridded_nox_df: nox dataframe that is split by the smallest unit
            nox1_col: the column name of small car emission
            nox2_col: the column name of big car emission
            nox_col: the column name of total emission
            smallest_unit: a list of column names that refers to the smallest unit of gridded_nox_df; for example, ['rd_id', 'grid_id'] for GridBasedMap
            group_id: the unit to group-by; for example, 'rd_id' for RoadBasedMap, 'grid_id' for GridBasedMap
        """

        self.base_layer_gdf = base_layer_gdf
        self.gridded_nox_df = gridded_nox_df
        self.nox1_col = nox1_col
        self.nox2_col = nox2_col
        self.nox_col = nox_col
        self.smallest_unit = smallest_unit
        self.group_id = group_id

    def year_total(self):

        """Returns: yearly total emission"""

        nox_year = self.gridded_nox_df.groupby(self.smallest_unit + ['year', 'month', 'day'],
                                               as_index=False).agg(
            {self.nox1_col: 'sum', self.nox2_col: 'sum', self.nox_col: 'sum'}). \
            groupby(self.smallest_unit, as_index=False).agg(
            {self.nox1_col: 'mean', self.nox2_col: 'mean', self.nox_col: 'mean'}). \
            groupby([self.group_id], as_index=False).agg(
            {self.nox1_col: 'sum', self.nox2_col: 'sum', self.nox_col: 'sum'})

        for col in [self.nox1_col, self.nox2_col, self.nox_col]:
            nox_year[col] = nox_year[col] * 365 / 1000000

        nox_year.rename(columns={self.nox1_col: '小车型年化排放（吨）', self.nox2_col: '大车型年化排放（吨）', self.nox_col: '总年化排放（吨）'},
                        inplace=True)
        nox_year = self.base_layer_gdf.merge(nox_year, on=self.group_id)
        nox_year = nox_year.to_crs(4326)

        return nox_year

    def day_average(self):

        """Calculate daily average emission."""

        day_dict = {'工作日': [0, 1, 2, 3, 4], '双休日': [5, 6]}

        lst = []

        for wkday in ['工作日', '双休日']:
            splitted_nox = self.gridded_nox_df[self.gridded_nox_df['weekday'].isin(day_dict[wkday])]
            nox_day = splitted_nox.groupby(self.smallest_unit + ['year', 'month', 'day'], as_index=False).agg(
                {self.nox1_col: 'sum', self.nox2_col: 'sum', self.nox_col: 'sum'}). \
                groupby(self.smallest_unit, as_index=False).agg(
                {self.nox1_col: 'mean', self.nox2_col: 'mean', self.nox_col: 'mean'}). \
                groupby([self.group_id]).agg({self.nox1_col: 'sum', self.nox2_col: 'sum', self.nox_col: 'sum'})

            for col in [self.nox1_col, self.nox2_col, self.nox_col]:
                nox_day[col] = nox_day[col] * 365 / 1000

            nox_day['day'] = wkday
            lst.append(nox_day)

        nox_day = pd.concat(lst)
        nox_day.rename(columns={self.nox1_col: '小车型日均排放（千克）', self.nox2_col: '大车型日均排放（千克）', self.nox_col: '总日均排放（千克）'},
                       inplace=True)
        nox_day = self.base_layer_gdf.merge(nox_day, on=self.group_id)
        nox_day = nox_day.to_crs(4326)

        return nox_day

    def peakhour_average(self):

        """Calculate average peak hours' emission. Morning peak hours include 7am-9am. Evening peak hours include
        5pm-7pm. """

        hour_dict = {'早高峰': [7, 8, 9], '晚高峰': [17, 18, 19]}
        day_dict = {'工作日': [0, 1, 2, 3, 4], '双休日': [5, 6]}

        lst = []

        for d in ['工作日', '双休日']:
            for h in ['早高峰', '晚高峰']:
                splitted_nox = self.gridded_nox_df[(self.gridded_nox_df['hour'].isin(hour_dict[h])) & (
                    self.gridded_nox_df['weekday'].isin(day_dict[d]))]
                nox_peakhour = splitted_nox.groupby(self.smallest_unit + ['hour'], as_index=False).agg(
                    {self.nox1_col: 'mean', self.nox2_col: 'mean', self.nox_col: 'mean'}). \
                    groupby(self.smallest_unit, as_index=False).agg(
                    {self.nox1_col: 'mean', self.nox2_col: 'mean', self.nox_col: 'mean'}). \
                    groupby([self.group_id], as_index=False).agg(
                    {self.nox1_col: 'sum', self.nox2_col: 'sum', self.nox_col: 'sum'})

                nox_peakhour['day'] = d
                nox_peakhour['peak'] = h
                lst.append(nox_peakhour)

        nox_peakhour = pd.concat(lst)
        nox_peakhour.rename(
            columns={self.nox1_col: '高峰小车型平均排放量（克）', self.nox2_col: '高峰大车型平均排放量（克）', self.nox_col: '高峰总平均排放量（克）'},
            inplace=True)
        nox_peakhour = self.base_layer_gdf.merge(nox_peakhour, on=self.group_id)
        nox_peakhour = nox_peakhour.to_crs(4326)

        return nox_peakhour

    def hour_average_dynamic(self, weekday, year, month, day):

        """Calculate 24hour dynamic emission; have datetime column that enables gif-making. Since the dataframe is
        hour average, the inputs year, month, and day refers to a typical date that should correspond to weekday """

        day_dict = {True: [0, 1, 2, 3, 4], False: [5, 6]}
        splitted_nox = self.gridded_nox_df[self.gridded_nox_df['weekday'].isin(day_dict[weekday])]
        nox_hour = splitted_nox.groupby(self.smallest_unit + ['hour'], as_index=False).agg(
            {self.nox1_col: 'mean', self.nox2_col: 'mean', self.nox_col: 'mean'}). \
            groupby([self.group_id, 'hour'], as_index=False).agg(
            {self.nox1_col: 'sum', self.nox2_col: 'sum', self.nox_col: 'sum'})

        # for col in [nox1_col, nox2_col, nox_col]:
        # 	nox_year[col] = nox_year[col]*365/1000000

        nox_hour.rename(columns={self.nox1_col: '小车型平均排放（克）', self.nox2_col: '大车型平均排放（克）', self.nox_col: '总排放（克）'},
                        inplace=True)
        nox_hour = self.base_layer_gdf.merge(nox_hour, on=self.group_id)
        nox_hour = nox_hour.to_crs(4326)

        nox_hour['year'] = year
        nox_hour['month'] = month
        nox_hour['day'] = day
        nox_hour['datetime'] = pd.to_datetime(nox_hour[['year', 'month', 'day', 'hour']])

        return nox_hour

    def ratio_small_dynamic(self, weekday, year, month, day):

        """Calculate 24hour dynamic small vehicles' contribution to total emission."""

        hour_grid = self.hour_average_dynamic(weekday, year, month, day)
        hour_grid['小车贡献率'] = hour_grid['小车型平均排放（克）'] / (hour_grid['小车型平均排放（克）'] + hour_grid['大车型平均排放（克）'])

        return hour_grid


class GridBasedMap(LayerBasedMap):

    def __init__(self, grid_gdf, grid_id, rd_gdf, rd_id, nox_df, nox1_col, nox2_col, nox_col, projected_crs):
        gridded_nox_df = distribute_nox(grid_gdf, rd_gdf, nox_df, nox1_col, nox2_col, nox_col, rd_id, grid_id,
                                        projected_crs)
        self.projected_crs = projected_crs
        super().__init__(grid_gdf, gridded_nox_df, nox1_col, nox2_col, nox_col, [rd_id, grid_id], grid_id)


class RegionBasedMap(LayerBasedMap):

    def __init__(self, region_gdf, region_id, rd_gdf, rd_id, nox_df, nox1_col, nox2_col, nox_col, projected_crs):
        gridded_nox_df = distribute_nox(region_gdf, rd_gdf, nox_df, nox1_col, nox2_col, nox_col, rd_id, region_id,
                                        projected_crs)
        self.projected_crs = projected_crs
        super().__init__(region_gdf, gridded_nox_df, nox1_col, nox2_col, nox_col, [rd_id, region_id], region_id)

    # 每平方公里排放
    def year_total(self):
        nox_year = super().year_total()
        nox_year = nox_year.to_crs(self.projected_crs)
        for col in ['小车型年化排放（吨）', '大车型年化排放（吨）', '总年化排放（吨）']:
            nox_year[col] = nox_year[col] / nox_year.area * 1000000
        nox_year = nox_year.to_crs(4326)
        return nox_year

    def day_average(self):
        nox_day = super().day_average()
        nox_day = nox_day.to_crs(self.projected_crs)
        for col in ['小车型日均排放（千克）', '大车型日均排放（千克）', '总日均排放（千克）']:
            nox_day[col] = nox_day[col] / nox_day.area * 1000000
        nox_day = nox_day.to_crs(4326)
        return nox_day

    def peakhour_average(self):
        nox_peak = super().peakhour_average()
        nox_peak = nox_peak.to_crs(self.projected_crs)
        for col in ['高峰小车型平均排放量（克）', '高峰大车型平均排放量（克）', '高峰总平均排放量（克）']:
            nox_peak[col] = nox_peak[col] / nox_peak.area * 1000000
        nox_peak = nox_peak.to_crs(4326)
        return nox_peak

    def hour_average_dynamic(self, weekday, year, month, day):
        nox_hour = super().hour_average_dynamic(weekday, year, month, day)
        nox_hour = nox_hour.to_crs(self.projected_crs)
        for col in ['小车型平均排放（克）', '大车型平均排放（克）', '总排放（克）']:
            nox_hour[col] = nox_hour[col] / nox_hour.area * 1000000
        nox_hour = nox_hour.to_crs(4326)
        return nox_hour


class RoadBasedMap(LayerBasedMap):

    def __init__(self, base_layer_gdf, layer_id, nox_df, rd_id, nox1_col, nox2_col, nox_col):
        """
        Args:
            base_layer_gdf: GeoDataFrame
            layer_id: unique id for base layer geometries
            nox_df: emission data
            rd_id: smallest geometric unit of emission data
            nox1_col: small vehicle emission
            nox2_col: big vehicle emission
            nox_col: total vehicle emission
        """
        super().__init__(base_layer_gdf, nox_df, nox1_col, nox2_col, nox_col, [layer_id, rd_id], layer_id)

# class RoadBasedMap(LayerBasedMap):
#
#     def __init__(self, base_layer_gdf, nox_df, nox1_col, nox2_col, nox_col, rd_id):
#         super().__init__(base_layer_gdf, nox_df, nox1_col, nox2_col, nox_col, [rd_id], rd_id)


########################################################################################################################

def plot_feature_importance(rf, features, path, model_name, version):
    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    # plot feature importance
    f_imt_df = pd.DataFrame({"name": features, "importance": importances})
    f_imt_df.sort_values('importance', ascending=False, inplace=True)
    select_imt_df = f_imt_df.head(5)
    select_imt_df['importance'] = select_imt_df['importance'].apply(lambda x: round(x, 2) * 100)

    fig = plt.figure(figsize=(9, 5))
    # Make a bar chart
    plt.bar(select_imt_df['name'], select_imt_df['importance'], orientation='vertical')
    plt.ylabel('Variable Importance')
    plt.xlabel('Variable')
    plt.title('Importance')

    plt.savefig(os.path.join(path, 'FeatureImportance_{}_v{}.png'.format(model_name, version)), bbox_inches="tight", dpi=500)

    return


def plot_truth_pred(df, name, path, model_name, version):
    fig = plt.figure(figsize=(8, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']

    x = df[name]
    y = df['pred_{}'.format(name)]
    m, b = np.polyfit(x, y, 1)
    plt.scatter(x, y)
    # add linear regression line to scatterplot
    plt.plot(x, m * x + b, color='red', label='y={:.2f}x+{:.2f}'.format(m, b))
    plt.axis('square')
    plt.ylabel('预测车流量')
    plt.xlabel('真实车流量')
    plt.title('真实值v.预测值 散点图')
    plt.legend()

    plt.savefig(os.path.join(path, 'TruthPredScatter_{}_v{}.png'.format(model_name, version)), bbox_inches="tight", dpi=500)

    return


def stacked_plots(df, kind, groupby, col1, col2, path):
    if kind == 'bar':
        stacked = True
    else:
        stacked = False
    # fig = plt.figure(figsize=(9, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plot_df = df.groupby(groupby, as_index=False).agg({col1: 'mean', col2: 'mean'}).set_index(groupby)
    plot_df.plot(kind=kind, stacked=stacked, color=['green', 'orange'])
    plt.xticks(rotation=0, ticks=range(len(plot_df)), labels=plot_df.index)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, '{}_by_{}_{}.png'.format(col1, groupby, kind)), bbox_inches="tight", dpi=500)

    return


# def stacked_bars(df, groupby, col1, col2, path):
#     # fig = plt.figure(figsize=(9, 5))
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#
#     plot_df = df.groupby(groupby, as_index=False).agg({col1: 'mean', col2: 'mean'})
#     plot_df.plot(kind='bar', stacked=True, color=['green', 'orange'])
#     plt.xticks(rotation=0)
#     plt.legend(loc='upper right')
#     plt.savefig(os.path.join(path, '{}_by_{}_{}.png'.format(col1, groupby, 'bar')), bbox_inches="tight", dpi=500)
#
#     return


def plot_sample_rd(nrows, ncols, groupby, dict, col1, col2, path):
    plt.rcParams.update({'font.size': 22})
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig, axs = plt.subplots(nrows, ncols, sharey=True)
    fig.set_size_inches(50, 20)
    count = 0
    for i in range(nrows):
        for j in range(ncols):
            df = dict[count]['df']
            name = dict[count]['name']
            plot_df = df.groupby(groupby, as_index=False).agg({col1: 'mean', col2: 'mean'})
            axs[i, j].plot(plot_df[groupby], plot_df[col1], label=col1, color='green')
            axs[i, j].plot(plot_df[groupby], plot_df[col2], label=col2, color='orange')
            # plt.xticks(rotation=0)
            axs[i, j].set_title(name)
            axs[i, j].legend(loc='upper right')
            count += 1
    plt.savefig(os.path.join(path, 'SampleRoad_by_{}.png'.format(groupby)), bbox_inches="tight", dpi=500)

    return

