from src.slicer import slice2D
from src.utils import find_nearest_date_index, get_sorted_datetimes
import pandas as pd
import numpy as np

w_S =  2.66622373e-6
start_date = pd.to_datetime("2019-07-02T12:04:37")
interval_hours = 0.402 * 20 * 0.005
# date_list = [start_date + pd.to_timedelta(interval_hours * i, unit='h') for i in range(600)]
date_list = [start_date + pd.to_timedelta(interval_hours * 0, unit='h'), start_date + pd.to_timedelta(interval_hours * 118, unit='h'), start_date + pd.to_timedelta(interval_hours * 186, unit='h')]

dsvdirectory='data/dsv/'

folder_path='data/vts/'
sorted_datetimes, sorted_file = get_sorted_datetimes(folder_path)

seconds_since_start = [(date - start_date).total_seconds() for date in sorted_datetimes]

outputdir = f'image/coupling/'

idx=0 #50 
file=sorted_file[idx]
target_date = sorted_datetimes[idx]
nearest_index = find_nearest_date_index(date_list, target_date)
filevtu = f'data/vtu_filter/data_{nearest_index}.vtu'
filevts = f'data/vts/{file}'
print(f'{idx} {filevtu} {filevts}')
print(f"Target date: {target_date} nearest date {date_list[nearest_index]}")
time_delta = target_date - date_list[nearest_index]
diff = time_delta.total_seconds() / 60
print(f'diff {diff:.1f}')

plane_type = 'both'
slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, diff, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=False, variable='Vr', plane_type = plane_type)
slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, diff, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=False, variable='density', plane_type = plane_type)
slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, diff, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=False, variable='T', plane_type = plane_type)


""" 
for idx, file in enumerate(sorted_file[:]):
    target_date = sorted_datetimes[idx]
    nearest_index = find_nearest_date_index(date_list, target_date)
    filevtu = f'vtu_filter/data_{nearest_index}.vtu'
    filevts = f'vts/{file}'
    print(f'{idx} {filevtu} {filevts}')
    print(f"Target date: {target_date} nearest date {date_list[nearest_index]}")
    time_delta = target_date - date_list[nearest_index]
    diff = time_delta.total_seconds() / 60
    print(f'diff {diff:.1f}')
    #slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, diff, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=True, variable='density')
    #slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, diff, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=True, variable='Vr')
    slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, diff, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=True, variable='T')


outputdir = f'image/coupling/'
name='density'
movie(outputdir, name)

outputdir = f'image/coupling/'
name='Vr'
movie(outputdir, name)

outputdir = f'image/coupling/'
name='T'
movie(outputdir, name)
"""