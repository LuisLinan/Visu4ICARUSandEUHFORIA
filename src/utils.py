
import pandas as pd
import os
import re
from bisect import bisect_left
from datetime import datetime

def get_sorted_datetimes(folder_path):
    file_list = os.listdir(folder_path)

    vts_files = [f for f in file_list if f.endswith('.vts')]

    file_datetime_tuples = []
    for file_name in vts_files:
        datetime_str = file_name.split('_')[1].replace('T', ' ').replace('-', ':').split('.')[0]
        file_datetime = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
        file_datetime_tuples.append((file_datetime, file_name))

    file_datetime_tuples.sort()

    sorted_datetimes = [t[0] for t in file_datetime_tuples]
    sorted_file = [t[1] for t in file_datetime_tuples]

    return sorted_datetimes, sorted_file

def find_nearest_date_index(date_list_e, target_date):
    pos = bisect_left(date_list_e, target_date)
    if pos == 0:
        return 0
    if pos == len(date_list_e):
        return len(date_list_e) - 1
    before = date_list_e[pos - 1]
    after = date_list_e[pos]

    if after - target_date < target_date - before:
        return pos
    else:
        return pos - 1

def extract_number(filename):
    match = re.search(r'data_(\d+)\.vts', filename)
    if match:
        return int(match.group(1))
    return None

def read_dsv_file(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True)
    data['date'] = pd.to_datetime(data['date'])
    return data