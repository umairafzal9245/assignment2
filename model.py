import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle


if __name__ == '__main__':

    symbols = ['AAPL', 'AMZN', 'GOOG', 'TSLA']

    yf.pdr_override()

    data = pd.DataFrame(
        columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

    for symbol in symbols:
        data = data.append(yf.download(
            symbol, interval='1m', period='1wk', ignore_tz=True))

    print("data shpae",data.shape)

    data.index = pd.to_datetime(data.index)
    data.index = data.index.tz_localize(None)
    data.index = data.index.strftime('%Y-%m-%d %H:%M:%S')

    features = ["Open", "High", "Low", "Volume"]
    target = "Adj Close"

    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(data[features])
    feature_transform = pd.DataFrame(
        columns=features, data=feature_transform, index=data.index)

    X = feature_transform
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print('LR Coefficients: \n', model.coef_)
    print('LR Intercept: \n', model.intercept_)

    y_test_pred = model.predict(X_test)

    print('Mean squared error: ', mean_squared_error(y_test, y_test_pred))
    print('Coefficient of determination: ', r2_score(y_test, y_test_pred))
