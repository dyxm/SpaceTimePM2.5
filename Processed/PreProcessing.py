# Created by Yuexiong Ding
# Date: 2018/9/30
# Description: 处理原始数据
import pandas as pd
import numpy as np
import os
import datetime
from MyModule import draw


def extract_data_by_stata(df_raw):
    save_root_path = '../DataSet/Processed/'
    states = set(df_raw['State Code'])
    file_names = {}
    for state in states:
        file_names[state] = []
        df_state_data = df_raw[df_raw['State Code'] == state]
        if not os.path.exists(save_root_path + 'state_' + state):
            os.makedirs(save_root_path + 'state_' + state)
        df_state_data['site'] = df_raw['State Code'] + df_raw['County Code'] + df_raw['Site Num']
        df_state_data.pop('State Code')
        df_state_data.pop('County Code')
        df_state_data.pop('Site Num')
        sites = set(df_state_data['site'])
        for site in sites:
            df_state_site_data = df_state_data[df_state_data['site'] == site]
            df_state_site_data.pop('site')
            df_state_site_data['Date Time'] = df_state_site_data.pop('Date Local') + ' ' + df_state_site_data.pop(
                'Time Local')
            df_state_site_data = df_state_site_data.sort_values(by="Date Time")
            print(site, len(df_state_site_data))
            file_name = save_root_path + 'state_' + state + '/' + site + '.csv'
            file_names[state].append(file_name)
            df_state_site_data.to_csv(file_name, index=False)
    return file_names


def fill_data(file_names):
    """
    用前面数据填充缺失数据
    :param file_names:
    :return:
    """
    print('用前面数据填充缺失数据...')
    for file_name in file_names:
        df_all = pd.read_csv(file_name, dtype=str)
        df_all['Date Time'] = pd.to_datetime(df_all['Date Time'])
        print(file_name)

        i = 1
        while i < len(df_all):
            delta_hours = int((df_all.loc[i]['Date Time'] - df_all.loc[i - 1]['Date Time']).seconds / 3600)
            delta_days = int((df_all.loc[i]['Date Time'] - df_all.loc[i - 1]['Date Time']).days)
            if delta_days > 0:
                delta_hours += delta_days * 24
            if delta_hours > 1:
                print('missing hours：', delta_hours)
                df_prior = df_all.loc[:i - 1]
                df_last = df_all.loc[i:]
                for h in range(delta_hours - 1):
                    df_prior_time = df_prior.loc[len(df_prior) - 1]['Date Time']
                    loc = i - delta_hours + h + 1
                    # if loc < 0:
                    #     loc = i - 1
                    df_temp = df_prior.loc[loc]
                    df_temp['Date Time'] = df_prior_time + datetime.timedelta(hours=1)
                    df_prior = df_prior.append(df_temp)
                    df_prior.index = [x for x in range(len(df_prior))]
                    i = len(df_prior)
                df_all = df_prior.append(df_last)
                df_all.index = [x for x in range(len(df_all))]
            else:
                i += 1
        print(len(df_all))
        df_all.to_csv(file_name, index=False)


def merge(file_names, usecols, how='inner'):
    site = 'Site_' + file_names[0][-13: -4]
    df_all = pd.read_csv(file_names[0], usecols=usecols, dtype=str)
    df_all[site] = df_all.pop('Sample Measurement')
    for i in range(1, len(file_names)):
        site = 'Site_' + file_names[i][-13: -4]
        df_temp = pd.read_csv(file_names[i], usecols=usecols, dtype=str)
        df_temp[site] = df_temp.pop('Sample Measurement')
        df_all = pd.merge(df_all, df_temp, how=how, on=['Date Time'])
        print(file_names[i])
    df_all.to_csv('../DataSet/Processed/Train/train_v1.csv', index=False)


def check_data(data_path):
    df_raw = pd.read_csv(data_path, dtype=str)
    start_datetime = datetime.datetime.strptime('2016-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    for i in range(17544):
        if not (df_raw.loc[i]['Date Time'] == start_datetime.strftime('%Y-%m-%d %H:%M:%S')):
            print(df_raw.loc[i]['Date Time'], start_datetime.strftime('%Y-%m-%d %H:%M:%S'))
            print(i)
        start_datetime = start_datetime + datetime.timedelta(hours=1)


def draw_time_series():
    """
    画出时间序列
    :return:
    """
    df_raw = pd.read_csv('../DataSet/Processed/Train/train_v1.csv')
    draw.draw_time_series(df_raw,
                          ['Site_060370016', 'Site_060371103', 'Site_060371201', 'Site_060374004', 'Site_060376012'])


def lag_correlation():
    main_col = 'Site_060371103'
    other_cols = ['Site_060371103', 'Site_060370016', 'Site_060371201', 'Site_060374004', 'Site_060376012']
    max_time_lag = 100
    min_corr_coeff = 0.3
    df_raw = pd.read_csv('../DataSet/Processed/Train/train_v1.csv')
    draw.draw_lag_correlation(df_raw, main_col, other_cols, max_time_lag, min_corr_coeff)


def main():
    # data_path = '../DataSet/Raw/state_hourly_88502_2016_2017.csv'
    # df_raw = pd.read_csv(data_path, dtype=str)
    # file_names = extract_data_by_stata(df_raw)
    # print(file_names)
    file_names = ['../DataSet/Processed/state_06/060370016.csv',
                  '../DataSet/Processed/state_06/060371103.csv',
                  '../DataSet/Processed/state_06/060371201.csv',
                  '../DataSet/Processed/state_06/060374004.csv',
                  '../DataSet/Processed/state_06/060376012.csv']
    # fill_data(file_names)
    merge(file_names, ['Sample Measurement', 'Date Time'])
    # draw_time_series()
    # lag_correlation()


if __name__ == '__main__':
    main()
    # check_data('../DataSet/Processed/state_06/060376012.csv')
