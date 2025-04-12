import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("../problem_data/plot_data/static_earth_orbit.csv")
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

# Combine all traces
fig = go.Figure(data=[trajectory, start_marker, end_marker])

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
