import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# Read CSV file
# filename = "../../problem_data/plot_data/static_space_orbit.csv"  # Change this to your CSV file path
filename = "../../problem_data/plot_data/static_earth_orbit.csv"  # Change this to your CSV file path
x_coords = []
y_coords = []
z_coords = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        try:
            x_coords.append(float(row[1]))  # Second column
            y_coords.append(float(row[2]))  # Third column
            z_coords.append(float(row[3]))  # Fourth column
        except (IndexError, ValueError):
            continue  # Skip rows that don't have valid data

# Create figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot orbit
ax.plot(x_coords, y_coords, z_coords, 'b-', label='Orbit')
ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=100, label='Start')
ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, label='End')

# Create Earth sphere (assuming coordinates are in km)
earth_radius = 6371  # Earth radius in km
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = earth_radius * np.outer(np.cos(u), np.sin(v))
y = earth_radius * np.outer(np.sin(u), np.sin(v))
z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot Earth
ax.plot_surface(x, y, z, color='blue', alpha=0.3, label='Earth')

# Add labels and legend
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Orbit Visualization')
ax.legend()

# Equal aspect ratio
max_range = np.array([x_coords, y_coords, z_coords]).max()
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

plt.tight_layout()
plt.show()