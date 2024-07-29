import numpy as np
import pyvista as pv
import pandas as pd
import os
from bisect import bisect_left
from itertools import cycle
import cmocean
from cmap import Colormap
import re
from glob import glob
from datetime import datetime
import cv2
from color import citrus, citrus_low

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

def extract_number(filename):
    match = re.search(r'data_(\d+)\.vts', filename)
    if match:
        return int(match.group(1))
    return None

def read_dsv_file(file_path):
    data = pd.read_csv(file_path, delim_whitespace=True)
    data['date'] = pd.to_datetime(data['date'])
    return data

def create_dictplanet(folder):
    planet_data = {}

    for file_name in os.listdir(folder):
        if file_name.endswith(".dsv"):
            # Extract the name of the planet/satellite
            planet_name = file_name.split('_')[1].split('.')[0]

            # Read the file and structure the data
            file_path = os.path.join(folder, file_name)
            data = read_dsv_file(file_path)

            # Save the data in the dictionary
            position_data = data[['date', 'r[AU]', 'clt[rad]', 'lon[rad]','vr[km/s]']]
            planet_data[planet_name] = position_data.set_index('date').to_dict('index')
    return planet_data

def slice2D(filevtu,filevts, dsvdirectory, target_date, outputdir, idx, diff, angle, save=False, variable='Vr'):
    mesh = pv.read(filevtu)
    radii = np.sqrt(mesh.points[:, 0] ** 2 + mesh.points[:, 1] ** 2 + mesh.points[:, 2] ** 2)
    mask = radii <= 20.5
    mesh = mesh.extract_points(mask)

    transform = pv.transformations.axis_angle_rotation([0, 0, 1], angle)
    mesh.transform(transform)

    clip_sphere = pv.Sphere(radius=1.01, center=(0, 0, 0))

    # Clipping the mesh with the sphere
    clipped_mesh = mesh.clip_surface(clip_sphere)

    planet_data=create_dictplanet(dsvdirectory)

    planet, data = list(planet_data.items())[0]

    # Find the closest position for the first planet at target_date
    dates = list(data.keys())
    nearest_date_index = find_nearest_date_index(dates, target_date)
    nearest_date = dates[nearest_date_index]

    planet_positions = {}
    for planet, data in planet_data.items():
        try:
            planet_positions[planet] = data[nearest_date]
        except:
            continue

    mesh_vts = pv.read(filevts)

    conversion_factor = 215
    mesh_vts.points *= conversion_factor

    inner_radius_vts = 21.5
    radii_vts = np.sqrt(mesh_vts.points[:, 0] ** 2 + mesh_vts.points[:, 1] ** 2 + mesh_vts.points[:, 2] ** 2)
    mask_vts = radii_vts > inner_radius_vts
    mesh_vts = mesh_vts.extract_points(mask_vts)

    outer_radius = 230
    radii = np.sqrt(mesh_vts.points[:, 0] ** 2 + mesh_vts.points[:, 1] ** 2)
    mask = radii < outer_radius
    mesh_vts = mesh_vts.extract_points(mask)

    if variable=='density':

        #EUHFORIA cm-3 to m-3
        n = mesh_vts.cell_data['n']
        rho =n*1e6

        cell_centers = mesh_vts.cell_centers()
        cell_points = cell_centers.points
        radii_squared = np.sum(cell_points[:, :2] ** 2, axis=1)

        #vmin=np.min(rho*radii_squared)
        #vmax=np.max(rho*radii_squared)

        vmin=2.39e11
        vmax=1.85e13

        mesh_vts.cell_data['rho'] = rho * radii_squared

        #COCONUT vtu g/m-3 to m-3
        n = mesh.point_data['rho']
        rho = n / 1.67e-30
        radii_squared = np.sum(mesh.points[:, :2] ** 2, axis=1)

        mesh.point_data['rho'] = rho * radii_squared
    elif variable=='T':

        n = mesh_vts.cell_data['n']

        # cm-3 to m-3
        rho =n*1e6

        #P  ( Pa ? )
        P = mesh_vts.cell_data['P']

        vmin=5.3e4
        vmax=3.7e6

        # T = P/(nkb)
        mesh_vts.cell_data['T'] = 1.27*P / (rho*1.38e-23)



    slice_vts = mesh_vts.slice(normal='z', origin=(0, 0, 0))
    slice_vts = slice_vts.smooth(n_iter=50, relaxation_factor=0.01)

    # Extraire la coupe équatoriale à z = 0
    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1))

    # Filtrer le maillage pour ne conserver que les points où z = 0
    slice_equatorial = mesh.slice(normal='z', origin=(0, 0, 0))

    color_palette = cycle(["blue", "red", "green", "magenta", "purple", "orange", "cyan", "lime", "yellow", "pink"])
    color_dict = {planet: next(color_palette) for planet in planet_positions.keys()}

    plotter = pv.Plotter(off_screen=True)
    plotter.set_background('white')
    plotter.add_mesh(clipped_mesh, scalars='Br', cmap='bwr', show_scalar_bar=False, clim=[-10, 10])

    if variable=='Vr':
        plotter.add_mesh(slice_equatorial, scalars='Vr', cmap=Colormap('tol:nightfall'), show_scalar_bar=False,
                         clim=[300, 800])
    elif variable=='density':
        plotter.add_mesh(slice_equatorial, scalars='rho', cmap=citrus, show_scalar_bar=False,clim=[vmin, vmax])
    elif variable=='T':
        plotter.add_mesh(slice_equatorial, scalars='T', cmap=Colormap('cmocean:balance'), show_scalar_bar=False,clim=[vmin, vmax])

        #plotter.add_mesh(slice_equatorial, scalars='rho', cmap=Colormap('matplotlib:nipy_spectral'), show_scalar_bar=False,clim=[vmin, vmax])

    else:
        print('not valid variable')

    output_image_path = 'visualization.png'
    plotter.view_xy()
    plotter.camera.position = (0, 0, 150)

    plotter.screenshot(output_image_path, transparent_background=True)

    plotter.close()


    if save:
        plotter = pv.Plotter(off_screen=True, window_size=[1920,1080])
    else:
        plotter = pv.Plotter(window_size=[1920, 1080])

    plotter.add_text(f"{target_date.strftime('%Y-%m-%d %H:%M:%S')}\n {diff:.1f}", position='upper_right', color='black',
                     font_size=13)

    w_S = 2.6 * 10 ** (-6)
    rayon_solaire_km = 695700
    for planet, pos in planet_positions.items():
        r = pos['r[AU]']*215
        AU_to_km = 149597870.7
        if r > 230:
            continue
        clt = pos['clt[rad]']
        lon = pos['lon[rad]']
        vr = pos['vr[km/s]'] / rayon_solaire_km

        x = r * np.sin(clt) * np.cos(lon)
        y = r * np.sin(clt) * np.sin(lon)
        z = r * np.cos(clt)

        r_line = np.linspace(0, 230, 200)  # Radial distance from 0 to r
        phi_line = -w_S * (r_line - r) / vr + lon  # Angle phi as a function of radial distance

        # x, y, z coordinates of points along the spiral
        x_spiral = r_line * np.sin(clt) * np.cos(phi_line)
        y_spiral = r_line * np.sin(clt) * np.sin(phi_line)
        z_spiral = r_line * np.cos(clt)

        # z=0
        color = color_dict.get(planet, "white")  # Uses "white" if the planet is not in the color dictionary
        plotter.add_mesh(pv.Sphere(radius=4, center=(x, y, z)), color=color, label=planet)

        points = np.column_stack((x_spiral, y_spiral, z_spiral))
        line = pv.lines_from_points(points)
        plotter.add_mesh(line, color=color, line_width=2)

    plotter.add_mesh(clipped_mesh, scalars='Br', cmap='bwr', show_scalar_bar=False,clim=[-10, 10])

    if variable == 'Vr':
        plotter.add_mesh(slice_equatorial, scalars='Vr', cmap=Colormap('tol:nightfall'), show_scalar_bar=False,
                         clim=[300, 800])
        plotter.add_mesh(slice_vts, scalars='vr', cmap=Colormap('tol:nightfall'), show_scalar_bar=False,
                         clim=[300, 800])
        plotter.add_scalar_bar(title='Vr (km/s)', vertical=True, title_font_size=22, label_font_size=18)

    elif variable=='density':
        plotter.add_mesh(slice_equatorial, scalars='rho',cmap=citrus, show_scalar_bar=False,clim=[vmin, vmax])
        plotter.add_mesh(slice_vts, scalars='rho', cmap=citrus, show_scalar_bar=False,clim=[vmin, vmax])
        plotter.add_scalar_bar(title='Density * r²  (Rs² * m-3)', vertical=True, title_font_size=22, label_font_size=18)
    elif variable=='T':
        plotter.add_mesh(slice_vts, scalars='T', cmap=Colormap('cmocean:balance'), show_scalar_bar=False,clim=[vmin, vmax])
        plotter.add_mesh(slice_equatorial, scalars='T', cmap=Colormap('cmocean:balance'), show_scalar_bar=False,clim=[vmin, vmax])
        plotter.add_scalar_bar(title='Temperature (K)', vertical=True, title_font_size=22, label_font_size=18)
    else:
        print('not valid variable')



    theta = np.linspace(0, 2 * np.pi, 720)
    r = np.array([0, 21.5, 115, 230])
    theta_grid, r_grid = np.meshgrid(theta, r)
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    z_grid = np.zeros_like(x_grid)

    for i in range(r_grid.shape[0]):
        points = np.c_[x_grid[i, :], y_grid[i, :], z_grid[i, :]]
        line = pv.lines_from_points(points)
        if i == 1:  # Index 1 pour la deuxième ligne
            # Add the second line with a different color and try a visual effect for dotted line
            plotter.add_mesh(line, color="black", line_width=1.2)
        else:
            plotter.add_mesh(line, color="gray", line_width=1.5)

    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.array([0, 21.5, 115, 234])
    theta_grid, r_grid = np.meshgrid(theta, r)
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    z_grid = np.zeros_like(x_grid)

    # Adding angular lines
    angle_labels = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
    extended_radius = 231 + 20
    for i in range(0, theta_grid.shape[1], 45):
        points = np.c_[x_grid[:, i], y_grid[:, i], z_grid[:, i]]
        line = pv.lines_from_points(points)
        plotter.add_mesh(line, color="gray", line_width=1.5)

    correctif_r={0:-10, 45:-10, 90:-10, 135:0, 180:10, 225:10,270:0, 315:-1}
    correctif_angle={0:0, 45:0, 90:2, 135:3, 180:1, 225:0,270:-2, 315:-1}
    for i, value in enumerate([0, 45, 90, 135, 180, 225,270, 315]):

        new_x = (extended_radius+correctif_r[value]) * np.cos(np.radians(value+correctif_angle[value]))
        new_y = (extended_radius+correctif_r[value]) * np.sin(np.radians(value+correctif_angle[value]))

        text_position = np.array([[new_x, new_y, 0]])

        plotter.add_point_labels(text_position, [angle_labels[i]], font_size=18, point_size=1, text_color='grey',
                                 show_points=False,shape=None)


    plane = pv.Plane(center=(-300, -150, 0), direction=(0, 0, 1), i_size=400, j_size=400)
    texture = pv.read_texture(output_image_path)
    plotter.add_mesh(plane, texture=texture)

    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.position = (-38.18563118811632, 0.8302754252406501, 970.1723378487242)
    plotter.camera.focal_point = (-38.18563118811632, 0.8302754252406501, 0.0)

    def update_text():
        cam = plotter.camera
        info = f"Position: {cam.position}\nFocal Point: {cam.focal_point}\nUp Vector: {cam.up}"
        print(info)

    def on_mouse_click(*args):
        update_text()
        plotter.render()

    #plotter.track_click_position(on_mouse_click, side="left")
    plotter.add_legend(face='circle', bcolor=None,loc="upper left")
    if save:
        plotter.show(auto_close=False)
        plotter.screenshot(f'{outputdir}pyvista_{variable}_{idx:04d}.bmp')
        #plotter.save_graphic(f'{outputdir}pyvista_{idx}.eps')
        plotter.close()
    else:
        plotter.show()

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

def movie(outputdir, name):
    image_files = sorted(glob(os.path.join(outputdir, f'pyvista_{name}_*.bmp')))

    # Check if any files were found
    if not image_files:
        raise ValueError("No image files found in the specified directory.")

    # Read the first image to get the dimensions
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # Define the name and codec for the output video
    output_video = f'{outputdir}{name}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5

    # Initialize the VideoWriter object
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Add each image to the video
    for image_file in image_files:
        frame = cv2.imread(image_file)
        video.write(frame)

    # Release the VideoWriter object
    video.release()


if __name__ == "__main__":

    w_S =  2.66622373e-6
    start_date = pd.to_datetime("2019-07-02T12:04:37")
    interval_hours = 0.402 * 20 * 0.005
    # date_list = [start_date + pd.to_timedelta(interval_hours * i, unit='h') for i in range(600)]
    date_list = [start_date + pd.to_timedelta(interval_hours * 0, unit='h'), start_date + pd.to_timedelta(interval_hours * 118, unit='h'), start_date + pd.to_timedelta(interval_hours * 186, unit='h')]

    dsvdirectory='dsv/'

    folder_path='vts/'
    sorted_datetimes, sorted_file = get_sorted_datetimes(folder_path)

    seconds_since_start = [(date - start_date).total_seconds() for date in sorted_datetimes]

    outputdir = f'image/coupling/'

    idx=0 #50 
    file=sorted_file[idx]
    target_date = sorted_datetimes[idx]
    nearest_index = find_nearest_date_index(date_list, target_date) #???????
    filevtu = f'vtu_filter/data_{nearest_index}.vtu' #???????
    filevts = f'vts/{file}'
    print(f'{idx} {filevtu} {filevts}')
    print(f"Target date: {target_date} neares date {date_list[nearest_index]}")
    time_delta = target_date - date_list[nearest_index]
    diff = time_delta.total_seconds() / 60
    print(f'diff {diff:.1f}')
    slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, diff, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=False, variable='Vr')
    slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, diff, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=False, variable='density')
    slice2D(filevtu, filevts, dsvdirectory, target_date, outputdir, idx, diff, angle=180+np.degrees(w_S*seconds_since_start[idx]), save=False, variable='T')
    

""" 
    for idx, file in enumerate(sorted_file[:]):
        target_date = sorted_datetimes[idx]
        nearest_index = find_nearest_date_index(date_list, target_date)
        filevtu = f'vtu_filter/data_{nearest_index}.vtu'
        filevts = f'vts/{file}'
        print(f'{idx} {filevtu} {filevts}')
        print(f"Target date: {target_date} neares date {date_list[nearest_index]}")
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