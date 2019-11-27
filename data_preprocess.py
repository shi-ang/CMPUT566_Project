# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:54:49 2019

@author: Shiang Qi
"""
import xlrd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def load_raw_data(file = "20180127NEGV7.xlsx"):
    sheets = xlrd.open_workbook(file)
    table = sheets.sheet_by_index(0)
    num_rows = table.nrows
    num_columns = table.ncols
    excel_list = []
    for row in range(1, num_rows):
        for col in range(num_columns):
            # get the value from each cell
            cell_value = table.cell(row, col).value
            # append the data to excel_list
            excel_list.append(cell_value)
    data = np.reshape(excel_list, (num_rows - 1, num_columns))
    return data

def trim_data(data):
    useless_rows = []
    for row_index in range(data.shape[0]):
        if data[row_index, 0] != 'PH' and data[row_index, 0] != 'SH':
            useless_rows.append(row_index)
        elif data[row_index, 0] == 'PH':
            data[row_index, 0] = 0
        elif data[row_index, 0] =='SH':
            data[row_index, 0] = 1
    data = np.delete(data, useless_rows, axis = 0)
    
    data = data.astype(np.float)
    
    label = data[:, 0]
    data = np.delete(data, 0, axis = 1)
    
    return data, label

def abandon_useless_features(data, non_zero_rate_threshold = 1):
    count_non_zero = np.count_nonzero(data, axis = 0)
    non_zero_rate = count_non_zero / data.shape[0]
    useless_feature_index = np.argwhere(non_zero_rate < non_zero_rate_threshold)
    data = np.delete(data, useless_feature_index.squeeze(), axis = 1)
    return data

def standardization(data):
    return (data - data.mean(axis = 0))/data.std(axis = 0)

if __name__ == '__main__':

    data = load_raw_data()
    data = data.T
    data, label = trim_data(data)
    data = abandon_useless_features(data)
    data = StandardScaler().fit_transform(data)
    
    pickle_file = 'SH.pickle'
    if not os.path.isfile(pickle_file):    #判断是否存在此文件，若无则存储
        print('Saving data to pickle file...')
        try:
            with open('SH.pickle', 'wb') as pfile:
                pickle.dump(
                        {
                            'dataset': data,
                            'labels': label,
                        },
                        pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

    print('Data cached in pickle file.')
    