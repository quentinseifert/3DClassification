import base64
import io
import os
import numpy as np
import dash
from dash import no_update
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import app_helper

files = [f for f in os.listdir('models') if os.path.isfile(os.path.join('models', f))]
print(files)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([

    html.Div(id='chosen file', children='''
        Dash: A web application framework for Python.
        '''),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Img(id='Tree'),

    dcc.Dropdown(
        id='choose_model',
        options=[
            {'label': i, 'value': i} for i in files
        ],
        value=''
    ),
    html.Div(id='dd-output-container'),

    html.Button('Predict', id='button'),
    html.Div(id='output-container-button',
             children='Press "Predict" to classify tree')
])





def get_array(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'xyz' in filename:
            # Assume that the user uploaded a CSV file
            df = np.loadtxt(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df




@app.callback(Output('Tree', 'src'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def plot_tree(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        tree = [get_array(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        tree = tree[0]
        fig = plt.figure()
        ax = Axes3D(fig)
        plt.axis('off')
        ax.scatter(tree[:, 0], tree[:, 1], tree[:, 2], s=20, alpha=1, color='black')
        fig.savefig('./test.png')
        test_base64 = base64.b64encode(open('./test.png', 'rb').read()).decode('ascii')
        src = 'data:image/png;base64,{}'.format(test_base64)
        return src

    else:
        return no_update


@app.callback(
    Output('dd-output-container', 'children'),
    [Input('choose_model', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)

@app.callback(
    Output('output-container-button', 'children'),
    [Input('button', 'n_clicks'),
     Input('choose_model', 'value')]
)
def show_file(n_clicks, value):
    print(value)
    if n_clicks == 0:
        return no_update

    if n_clicks is not None and not os.path.exists('test.png'):
        return 'Please upload a point cloud'

    if n_clicks is not None and os.path.exists('test.png'):
        print(n_clicks)
        img = app_helper.prep_prediction('test.png')
        model = load_model('models/' + value)
        pred = model.predict(img)
        os.remove('test.png')
        return f'The tree belongs to class {np.argmax(pred)}'





if __name__ == '__main__':
    app.run_server(debug=True)