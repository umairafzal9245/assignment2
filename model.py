from psx import stocks
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, help='Ticker symbol')
    args = parser.parse_args()

    if args.ticker is None:
        raise ValueError('No ticker symbol provided')

    symbol = args.ticker

    data = stocks(symbol, start=datetime.date(2015, 1, 1), end=datetime.date.today())

    features = ["Open", "High", "Low"]
    target = "Close"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print('LR Coefficients: \n', model.coef_)
    print('LR Intercept: \n', model.intercept_)

    y_test_pred = model.predict(X_test)

    print('Mean squared error: ',mean_squared_error(y_test, y_test_pred))
    print('Coefficient of determination: ',r2_score(y_test, y_test_pred))