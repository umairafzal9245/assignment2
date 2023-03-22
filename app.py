from flask import Flask, jsonify, request
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import random

app = Flask(_name_)

data_df = None  # Global variable to store the data

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

dashboard_app = dash.Dash(_name_, server=app, url_base_pathname='/dashboard/')

dashboard_app.layout = html.Div([
    html.H1('Stock Price Prediction'),
    html.Div([
        html.Div([
            dcc.Graph(id='stock-chart')
        ], className='twelve columns')
    ], className='row'),

    html.Div([
        html.Div([
            html.H3('Performance Metrics'),
            html.H4('Mean Squared Error'),
            html.Div(id='mse'),
            html.H4('R2 Score'),
            html.Div(id='r2')
        ], className='six columns')
    ], className='row'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,
        n_intervals=0
    )
])


@dashboard_app.callback(
    Output('stock-chart', 'figure'), Output('mse',
                                            'children'), Output('r2', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph_live(n):

    random_open = random.uniform(100, 150)
    random_high = random.uniform(100, 150)
    random_low = random.uniform(100, 150)
    random_close = random.uniform(100, 150)
    random_adj_close = random.uniform(100, 150)
    random_volume = random.uniform(100, 150)

    point = pd.DataFrame({"Open": [random_open], "High": [random_high], "Low": [random_low], "Close": [
                         random_close], "Adj Close": [random_adj_close], "Volume": [random_volume]})
    current_date = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')
    point.index = [current_date]

    global data_df
    data_df = pd.concat([data_df, point])

    features = ["Open", "High", "Low", "Volume"]
    scaler = MinMaxScaler()
    data_transform = scaler.fit_transform(data_df[features])
    data_transform = pd.DataFrame(
        columns=features, data=data_transform, index=data_df.index)

    predictions = model.predict(data_transform)

    figure = {
        'data': [
            {'x': data_df.index,
                'y': data_df['Adj Close'], 'type': 'line', 'name': 'Actual'},
            {'x': data_df.index, 'y': predictions,
                'type': 'line', 'name': 'Predicted'}
        ],
        'layout': {
            'title': 'Stock Prices',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Price'}
        }
    }

    error = mean_squared_error(data_df['Adj Close'], predictions)
    r2score = r2_score(data_df['Adj Close'], predictions)

    return figure, error, r2score


@app.route('/', methods=['GET'])
def home():
    return "<h1>Stock Prediction API</h1><p>This API predicts stock prices based on input data.</p>"


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json(force=True)

    data_df = pd.DataFrame({"Open": [data['Open']], "High": [data['High']], "Low": [data['Low']], "Close": [
                           data['Close']], "Adj Close": [data['Adj Close']], "Volume": [data['Volume']]})

    features = ["Open", "High", "Low", "Volume"]
    scaler = MinMaxScaler()
    data_transform = scaler.fit_transform(data_df[features])
    data_transform = pd.DataFrame(
        columns=features, data=data_transform, index=data_df.index)

    predictions = model.predict(data_transform)

    actual = [data['Adj Close']]

    error = mean_squared_error(actual, predictions)

    output = {'predicted': predictions.tolist()[0], 'MSE': error}

    return jsonify(output)


if _name_ == '_main_':

    app.run(host='localhost', port=5000, debug=True)
