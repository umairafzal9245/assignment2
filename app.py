from flask import Flask, jsonify, request
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

app = Flask(_name_)

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create the Dash app
dashboard_app = dash.Dash(_name_, server=app, url_base_pathname='/dashboard/')

# Create the layout
dashboard_app.layout = html.Div([
    html.H1('Stock Price Prediction'),
    html.Div([
        html.Div([
            dcc.Graph(id='stock-chart')
        ], className='twelve columns')
    ], className='row'),

    # write a code for performance metrics
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
        interval=1*1000,  # in milliseconds
        n_intervals=0
    )
])

# Create the callback


@dashboard_app.callback(
    Output('stock-chart', 'figure'), Output('mse',
                                            'children'), Output('r2', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph_live(n):

    # Fetch the data for the selected stock
    # Replace this with your own logic to fetch live data for the selected stock
    yf.pdr_override()

    data_df = yf.download('AAPL', interval='1m', period='1wk', ignore_tz=True)

    # Transform the features using the same scaler used during training
    features = ["Open", "High", "Low", "Volume"]
    scaler = MinMaxScaler()
    data_transform = scaler.fit_transform(data_df[features])
    data_transform = pd.DataFrame(
        columns=features, data=data_transform, index=data_df.index)

    # Use the model to make predictions
    predictions = model.predict(data_transform)

    # Create the figure
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

    # Calculate the error
    error = mean_squared_error(data_df['Adj Close'], predictions)
    r2score = r2_score(data_df['Adj Close'], predictions)

    return figure, error, r2score


@app.route('/', methods=['GET'])
def home():
    return "<h1>Stock Prediction API</h1><p>This API predicts stock prices based on input data.</p>"


@app.route('/predict', methods=['POST'])
def predict():
    # Get the request data as a dictionary
    data = request.get_json(force=True)

    # Convert the dictionary to a pandas dataframe
    data_df = pd.DataFrame({"Open": [data['Open']], "High": [data['High']], "Low": [data['Low']], "Close": [
                           data['Close']], "Adj Close": [data['Adj Close']], "Volume": [data['Volume']]})

    # Transform the features using the same scaler used during training
    features = ["Open", "High", "Low", "Volume"]
    scaler = MinMaxScaler()
    data_transform = scaler.fit_transform(data_df[features])
    data_transform = pd.DataFrame(
        columns=features, data=data_transform, index=data_df.index)

    # Use the model to make predictions
    predictions = model.predict(data_transform)

    actual = [data['Adj Close']]

    # Calculate the error
    error = mean_squared_error(actual, predictions)

    # create a json object to return
    output = {'predicted': predictions.tolist()[0], 'MSE': error}
    # Convert the predictions to a list and return as JSON
    return jsonify(output)


if _name_ == '_main_':

    app.run(host='localhost', port=5000, debug=True)
