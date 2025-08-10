import plotly.graph_objects as go
import numpy as np

def plot_attractor(series, title="Attractor"):
    x,y,z = series[:,0], series[:,1], series[:,2]
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(width=2))])
    fig.update_layout(title=title, scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))
    return fig
