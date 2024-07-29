import pyvista as pv

# Load datasets
coconut_data = pv.read("vts\hsphere_2019-07-02T12-04-22.vts")
euhforia_data = pv.read("vtu_filter\data_0.vtu")

# Create a plotter object
plotter = pv.Plotter()

# Add datasets to the plotter
plotter.add_mesh(coconut_data, cmap="viridis", opacity=0.7)
plotter.add_mesh(euhforia_data, cmap="plasma", opacity=0.7)

# Customize the visualization
plotter.camera_position = 'iso'
plotter.add_legend(['COCONUT', 'EUHFORIA'], [coconut_data, euhforia_data], bcolor='w')
plotter.set_background("black")

# Show the plot
plotter.show()
