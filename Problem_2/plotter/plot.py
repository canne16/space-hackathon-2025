import pandas as pd
import plotly.graph_objects as go
import numpy as np

df = pd.read_csv("../problem_data/plot_data/earth_output.csv")
df.columns = [col.strip() for col in df.columns]

# Main trajectory line
trajectory = go.Scatter3d(
    x=df['X'], y=df['Y'], z=df['Z'],
    mode='lines',
    line=dict(color='royalblue', width=2),
    name='Trajectory'
)

# Start point (green)
start_marker = go.Scatter3d(
    x=[df['X'].iloc[0]],
    y=[df['Y'].iloc[0]],
    z=[df['Z'].iloc[0]],
    mode='markers',
    marker=dict(color='green', size=6),
    name='Start'
)

# End point (red)
end_marker = go.Scatter3d(
    x=[df['X'].iloc[-1]],
    y=[df['Y'].iloc[-1]],
    z=[df['Z'].iloc[-1]],
    mode='markers',
    marker=dict(color='red', size=6),
    name='End'
)

# Create Earth sphere
r = 6700e3  # Radius in km
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2 * np.pi, 50)
theta, phi = np.meshgrid(theta, phi)

x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

earth = go.Surface(
    x=x, y=y, z=z,
    colorscale=[[0, '#f0dda9'], [1, '#f0dda9']],
    opacity=1,
    showscale=False,
    name='Earth'
)

# Combine all traces
fig = go.Figure(data=[earth, trajectory, start_marker, end_marker])

# Layout
fig.update_layout(
    scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)'
    ),
    title='Satellite Trajectory (Interactive)',
    margin=dict(l=0, r=0, b=0, t=30),
    legend=dict(x=0, y=1)
)

fig.show()
