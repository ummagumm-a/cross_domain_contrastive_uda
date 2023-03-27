import pandas as pd
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import os
from functools import reduce
from tqdm import tqdm

def get_values(arr):
    return list(map(lambda x: x.value, arr))

def make_dataframe_for_logs(path):
    event_acc = EventAccumulator(path)
    event_acc.Reload()

    columns = []
    column_names = []
    stride = 13
    window = np.ones(stride) / stride
    columns_that_need_stride = ['Contrastive loss grad norm', 'Classification loss grad norm', 'Total loss grad norm']
    for scalar_tag in event_acc.Tags()['scalars']:
        column_names.append(scalar_tag)
        data = event_acc.Scalars(scalar_tag)
        data = get_values(data)
        if scalar_tag in columns_that_need_stride:
            data = np.convolve(data, window, 'same')
            data = data[np.arange(0, len(data), stride)]
            data = data.tolist()

        columns.append(data)

    df = pd.DataFrame(list(zip(*columns)), columns=column_names)
    # df = df.iloc[np.arange(0, len(df), 2)]
    df['epoch'] = df['epoch'] * 2 + np.arange(len(df)) % 2
    df.set_index('epoch', inplace=True)
    return df

def extract_name(path, logs_dir):
    pattern = fr'{logs_dir}/([\w\s,]+, )fold_\d'
    match = re.match(pattern, path)
    return match.group(1)

def combine_dfs(dfs):
    common_columns = reduce(lambda x1, x2: x1.intersection(x2), 
                            map(lambda x: set(x.columns), dfs))

    df_mean = sum(map(lambda x: x[list(common_columns)], dfs)) / len(dfs)
    for df in dfs:
        other_columns = list(set(df.columns).difference(common_columns))
        if len(other_columns) != 0:
            df_mean[other_columns] = df[other_columns]

    return df_mean

def combine_folds(folds, logs_dir):
    dfs = []
    for fold_path in folds:
        dfs.append(make_dataframe_for_logs(fold_path))      
        
    return extract_name(folds[0], logs_dir), combine_dfs(dfs)

def logs_to_csvs(logs_dir):
    logs = os.listdir(logs_dir)
    logs = list(map(lambda x: os.path.join(logs_dir, x), logs))
    names = set(map(lambda x: extract_name(x, logs_dir), logs))
    name_groups = [[y for y in logs if x in y] for x in names]
    dfs = list(map(lambda x: combine_folds(x, logs_dir), tqdm(name_groups)))
    return dfs

def append_name_to_cols(pair):
    name, df = pair
    df.columns = pd.Index(map(lambda column_name: name.replace(', ', '_') + '_' + column_name, df.columns))
    return df

def merge_dfs_between_runs(dfs):
    dfs = list(map(append_name_to_cols, dfs))
    df = pd.concat(dfs, axis=1)
    return df

def save_dfs(dfs, destination_dir):
    for name, df in dfs:
        df.to_csv(os.path.join(destination_dir, name + '.csv'))

if __name__ == '__main__':
    logs_dir = os.path.join('lightning_logs', 'kfold_run')
    
    dfs = logs_to_csvs(logs_dir)
    df = merge_dfs_between_runs(dfs)
    df.to_csv('logs.csv', na_rep='nan')
#     destination_dir = 'logs_csvs'
#     os.makedirs('logs_csvs', exist_ok=True)
#     save_dfs(dfs, destination_dir)
