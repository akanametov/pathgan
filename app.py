from typing import Optional
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np

import glob
import os



def collect_first_best(pattern: str, neg: Optional[str] = None):
    files = []
    for file in sorted(glob.glob(pattern), reverse=True):
        if neg and neg in file:
            continue
        files.append(file)

    figures = []
    for file in files:
        title = os.path.split(file)[1].split('.')[0]
        df1 = pd.read_csv(file)
        figures.append(px.box(df1, x="map_name", y="counts",
                              color="algo_type",
                              notched=True, title=title))
    return figures


def collect_line_plots(pattern: str, neg: Optional[str] = None):
    files = []
    for file in sorted(glob.glob(pattern), reverse=True):
        if neg and neg in file:
            continue
        files.append(file)

    lineplots = []
    for file in files:
        with open(file) as f:
            data = json.loads(f.readline().rstrip('\n'))
            title = os.path.split(file)[1].split('.')[0]
            fig = go.Figure()
            fig.update_layout(dict(title=title))
            for map_name, counts in data.items():
                uniform_mean = np.array(counts['uniform']['mean'])
                uniform_std = np.array(counts['uniform']['std'])

                y_u1 = (uniform_mean - uniform_std).tolist()
                y_u2 = (uniform_mean + uniform_std).tolist()

                roi_mean = np.array(counts['roi']['mean'])
                roi_std = np.array(counts['roi']['std'])
                y_r1 = (roi_mean - 0.5 * roi_std).tolist()
                y_r2 = (roi_mean + roi_std).tolist()

                x1 = list(range(len(uniform_mean)))
                x2 = list(range(len(roi_mean)))

                fig.add_trace(go.Scatter(x=x1 + x1[::-1], y=y_u1 + y_u2[::-1], fill='toself', name='RRT*-uniform_' + map_name))
                fig.add_trace(go.Scatter(x=x2 + x2[::-1], y=y_r1 + y_r2[::-1], fill='toself', name='RRT*-roi_' + map_name))

                #fig.add_trace(go.Scatter(x=x1, y=uniform_mean, name='RRT*-uniform_' + map_name))
                #fig.add_trace(go.Scatter(x=x2, y=roi_mean, name='RRT*-roi_' + map_name))
            fig.update_traces(mode='lines')
            lineplots.append(fig)
    return lineplots


figures = collect_first_best('logs/collected_stats_gan*.csv')
figures.extend(collect_first_best('logs/collected_stats_pix2pix*.csv'))
figures.extend(collect_first_best('logs/gan*.csv', neg='moving_ai'))
figures.extend(collect_first_best('logs/pix2pix*.csv', neg='moving_ai'))
figures.extend(collect_first_best('logs/gan_moving_ai*.csv'))
figures.extend(collect_first_best('logs/pix2pix_moving_ai*.csv'))

lineplots = collect_line_plots('logs/collected_stats_gan*.plot')
lineplots.extend(collect_line_plots('logs/collected_stats_pix2pix*.plot'))
lineplots.extend(collect_line_plots('logs/gan*.plot', neg='moving_ai'))
lineplots.extend(collect_line_plots('logs/pix2pix*.plot', neg='moving_ai'))
lineplots.extend(collect_line_plots('logs/gan_moving_ai*.plot'))
lineplots.extend(collect_line_plots('logs/pix2pix_moving_ai*.plot'))

with open('box_plots.html', 'w') as f:
    for fig in figures:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
with open('line_plots.html', 'w') as f:
    for line in lineplots:
        f.write(line.to_html(full_html=False, include_plotlyjs='cdn'))


app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([dcc.Graph(figure=fig) for fig in figures]),
    html.Div([dcc.Graph(figure=lineplot) for lineplot in lineplots])
])
app.run_server(debug=True)"
