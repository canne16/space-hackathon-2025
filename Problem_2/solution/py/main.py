import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import csv

# Constants
mu = 398600.4415e9
R = 6371302.0  # Radius of Earth (m)
omega_E = 7.29211e-5
J_2 = 1082.8e-6

#Start parameters
altitude = 600e3  # 400 km altitude
angle = 98.0 * np.pi / 180.0
velocity = np.sqrt(mu / (R + altitude))  # Circular orbit velocity

def two_body_equations(t, y):
    """Differential equations for the 2-body problem"""
    r = y[:3]  # Position vector
    v = y[3:]  # Velocity vector
    
    r_mag = np.linalg.norm(r)  # Magnitude of position vector
    
    # Derivatives
    drdt = v
    dvdt = -mu * r / (r_mag ** 3)
    
    return np.concatenate((drdt, dvdt))


def simulate_orbit(initial_position, initial_velocity, t_span, t_eval):
    """Simulate satellite orbit and write to CSV every 100 seconds"""
    y0 = np.concatenate((initial_position, initial_velocity))
    
    # Solve the differential equations
    sol = solve_ivp(two_body_equations, t_span, y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    
    # Write to CSV file
    with open('../../problem_data/plot_data/output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ'])
        
        # Write data every 100 seconds
        for i, t in enumerate(sol.t):
            if t % 100 < 1e-6 or i == 0 or i == len(sol.t)-1:  # Every 100 seconds plus first/last points
                x, y, z = sol.y[0][i], sol.y[1][i], sol.y[2][i]
                vx, vy, vz = sol.y[3][i], sol.y[4][i], sol.y[5][i]
                writer.writerow([t, x, y, z, vx, vy, vz])
    
    return sol


def get_earth_solution():
    earth_solution = {
    "t": [],
    "x": []
}

for i in range(len(solution.t)):
    t = solution.t[i]
    angle = -t * omega_E
    earth_solution["t"].append(t)
    earth_solution["x"].append([
        solution.y[0][i] * np.cos(angle) - solution.y[1][i] * np.sin(angle),
        solution.y[0][i] * np.sin(angle) + solution.y[1][i] * np.cos(angle),
        solution.y[2][i],
        0.0,
        0.0,
        0.0,
    ])
    return earth_solution

# Initial conditions for a circular orbit
initial_position = np.array([R + altitude, 0, 0])  # Start at x-axis
initial_velocity = np.array([0, velocity*np.cos(angle), velocity*np.sin(angle)])  # Circular orbit velocity

# Time parameters
orbital_period = 2 * np.pi * (R + altitude) / np.linalg.norm(initial_velocity)
t_span = (0, 100*60*60)  # Simulate for 2 orbits
t_eval = np.arange(t_span[0], t_span[1], 100)  # Evaluate every 100 seconds

# Run simulation
print("Simulating orbit... Output will be saved to output.csv")
solution = simulate_orbit(initial_position, initial_velocity, t_span, t_eval)
print("Simulation complete! Data saved to output.csv")

# Corrected Earth rotation transformation
earth_solution = get_earth_solution()
    
with open('../../problem_data/plot_data/earth_output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ'])
    for i in range(len(earth_solution["t"])):
        t = earth_solution["t"][i]
        x, y, z, vx, vy, vz = earth_solution["x"][i]
        writer.writerow([t, x, y, z, vx, vy, vz])
print("Earth rotation simulation complete! Data saved to earth_output.csv")

for alt in range(400, 600, 10):
    altitude = alt * 1e3
    initial_position = np.array([R + altitude, 0, 0])  # Start at x-axis
    initial_velocity = np.array([0, velocity*np.cos(angle), velocity*np.sin(angle)])  # Circular orbit velocity

    # Time parameters
    t_span = (0, 100*60*60)  # Simulate for 2 orbits
    t_eval = np.arange(t_span[0], t_span[1], 100)  # Evaluate every 100 seconds

    # Run simulation
    
    solution = simulate_orbit(initial_position, initial_velocity, t_span, t_eval)
    