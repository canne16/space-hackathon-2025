import numpy as np

# Parameters for the sphere
earth_radius = 6371  # in km
theta = np.linspace(0, np.pi, 100)  # Latitude range (0 to pi)
phi = np.linspace(0, 2 * np.pi, 100)  # Longitude range (0 to 2pi)

# Meshgrid for sphere surface
theta, phi = np.meshgrid(theta, phi)
x = earth_radius * np.sin(theta) * np.cos(phi)
y = earth_radius * np.sin(theta) * np.sin(phi)
z = earth_radius * np.cos(theta)

# Flatten the grid for export
points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

# Create polygon connectivity (triangles) by creating faces
faces = []
for i in range(len(theta) - 1):
    for j in range(len(phi) - 1):
        # Points (i,j), (i+1,j), (i,j+1), (i+1,j+1)
        p0 = i * len(phi) + j
        p1 = (i + 1) * len(phi) + j
        p2 = i * len(phi) + (j + 1)
        p3 = (i + 1) * len(phi) + (j + 1)
        
        # Two triangles per quad
        faces.append([3, p0, p1, p2])
        faces.append([3, p1, p3, p2])

# Save points and faces to VTK format
vtk_path = "earth_mesh.vtk"

# Write the VTK file
with open(vtk_path, 'w') as file:
    file.write("# vtk DataFile Version 3.0\n")
    file.write("Earth Mesh\n")
    file.write("ASCII\n")
    file.write("DATASET POLYDATA\n")
    file.write(f"POINTS {points.shape[0]} float\n")
    np.savetxt(file, points, fmt='%.6f')
    file.write(f"POLYGONS {len(faces)} {len(faces) * 4}\n")
    for face in faces:
        file.write(f"{' '.join(map(str, face))}\n")

vtk_path
