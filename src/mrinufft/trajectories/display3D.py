from mrinufft import get_operator, get_density
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px

from mrinufft.io import read_trajectory

def view_sliced_3d_data(shots, shape, osf=1):
    samples = shots.reshape(-1, shots.shape[-1])
    dcomp = get_density("pipe")(samples, shape)
    grid_op = get_operator("gpunufft")(samples, [sh*osf for sh in shape], density=dcomp, upsampfac=1)
    gridded_ones = grid_op.raw_op.adj_op(np.ones(samples.shape[0]), None, True)
    time = grid_op.raw_op.adj_op(
        np.tile(np.linspace(1, 10, shots.shape[1]), (shots.shape[0], )),
        None,
        True,
    )
    turbo_factor = 176
    shot_specific = grid_op.raw_op.adj_op(
        np.repeat(np.linspace(1, 10, turbo_factor), samples.shape[0]//turbo_factor+1)[:samples.shape[0]],
        None,
        True,
    )
    shot_grid = np.abs(shot_specific) / np.abs(gridded_ones)
    time_grid = np.abs(time) / np.abs(gridded_ones)
    return np.abs(gridded_ones)
    

traj, params = read_trajectory("/neurospin/metric/Chaithya/Projects/MP2RAGE/2024_06_24__SENIOR_LowAF/traj/d3_ICar_TW1_Nc83_Ns512_G39_S100_N240_OSF0.5_c39.46_d1.27_cOS1.0__D13M6Y2024T137.bin")

traj[np.where(traj>0.5)] = 0.5
traj[np.where(traj<-0.5)] = -0.5

data = np.squeeze(view_sliced_3d_data(traj, params['img_size'], 1))



# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Slider(id='x-slider', min=0, max=data.shape[0]-1, step=1, value=data.shape[0]//2),
        dcc.Slider(id='y-slider', min=0, max=data.shape[1]-1, step=1, value=data.shape[1]//2),
        dcc.Slider(id='z-slider', min=0, max=data.shape[2]-1, step=1, value=data.shape[1]//2),
        dcc.Dropdown(id='cmap-dropdown', options=[
            {'label': 'Rainbow', 'value': 'rainbow'},
            {'label': 'Plasma', 'value': 'plasma'},
            {'label': 'Inferno', 'value': 'inferno'},
            {'label': 'Magma', 'value': 'magma'},
            {'label': 'Cividis', 'value': 'cividis'},
            {'label': 'Greys', 'value': 'Greys'},
            {'label': 'Purples', 'value': 'Purples'},
            {'label': 'Blues', 'value': 'Blues'},
            {'label': 'Greens', 'value': 'Greens'},
            {'label': 'Oranges', 'value': 'Oranges'},
            {'label': 'Reds', 'value': 'Reds'}
        ], value='rainbow', clearable=False, style={'width': '50%'})
    ]),
    html.Div([
        html.Div([dcc.Graph(id='slice-z')], style={'display': 'inline-block', 'width': '32%'}),
        html.Div([dcc.Graph(id='slice-y')], style={'display': 'inline-block', 'width': '32%'}),
        html.Div([dcc.Graph(id='slice-x')], style={'display': 'inline-block', 'width': '32%'})
    ], style={'display': 'flex', 'justify-content': 'space-between'}),
    html.Div([
        dcc.Graph(id='trajectory-plot', figure={})
    ])
])

# Define the callback to update the plots
@app.callback(
    [Output('slice-z', 'figure'),
     Output('slice-y', 'figure'),
     Output('slice-x', 'figure'),
     Output('trajectory-plot', 'figure')],
    [Input('x-slider', 'value'),
     Input('y-slider', 'value'),
     Input('z-slider', 'value'),
     Input('cmap-dropdown', 'value')]
)
def update_slices(x_idx, y_idx, z_idx, cmap):
    # Slice along z-axis
    slice_z = data[:, :, z_idx]
    fig_z = px.imshow(slice_z, color_continuous_scale=cmap, zmax=np.nanmax(data), zmin=np.nanmin(data))
    
    # Slice along y-axis
    slice_y = data[:, y_idx, :]
    fig_y = px.imshow(slice_y, color_continuous_scale=cmap, zmax=np.nanmax(data), zmin=np.nanmin(data))
    
    # Slice along x-axis
    slice_x = data[x_idx, :, :]
    fig_x = px.imshow(slice_x, color_continuous_scale=cmap, zmax=np.nanmax(data), zmin=np.nanmin(data))
    
    # Create 3D trajectories plot
    fig_trajectory = go.Figure()
    for i in range(19):
        fig_trajectory.add_trace(go.Scatter3d(
            x=traj[i, :, 0],
            y=traj[i, :, 1],
            z=traj[i, :, 2],
            mode='lines',
        ))
    fig_trajectory.update_layout(
        title='3D Trajectories',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    return fig_z, fig_y, fig_x, fig_trajectory

@app.callback(
    [Output('x-slider', 'value'),
     Output('y-slider', 'value'),
     Output('z-slider', 'value')],
    [Input('slice-x', 'clickData'),
     Input('slice-y', 'clickData'),
     Input('slice-z', 'clickData')],
    [State('x-slider', 'value'),
     State('y-slider', 'value'),
     State('z-slider', 'value')]
)
def update_sliders(click_data_x, click_data_y, click_data_z, x_val, y_val, z_val):
    ctx = dash.callback_context
    if not ctx.triggered:
        return x_val, y_val, z_val

    prop_id = ctx.triggered[0]['prop_id']
    if 'slice-x' in prop_id and click_data_x:
        z_val = click_data_x['points'][0]['x']
        y_val = click_data_x['points'][0]['y']
    elif 'slice-y' in prop_id and click_data_y:
        z_val = click_data_y['points'][0]['x']
        x_val = click_data_y['points'][0]['y']
    elif 'slice-z' in prop_id and click_data_z:
        x_val = click_data_z['points'][0]['x']
        y_val = click_data_z['points'][0]['y']
    
    return x_val, y_val, z_val

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")

