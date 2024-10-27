import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet




# evaluate function
def evaluate_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ElasticNet")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--l1_ratio", type=float, default=0.5)
    args = parser.parse_args()

    # read data
    data = pd.read_csv("wine-quality.csv")

    # split data
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # The Predicted column is "quality" which is a score between 0 and 10
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)

    # The Target column is "quality" which is a score between 0 and 10
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # train model
    model = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
    model.fit(train_x, train_y)

    # evaluate model
    y_pred = model.predict(test_x) # predict on test set (predicted qualities for test wine)
    rmse, mae, r2 = evaluate_metrics(test_y, y_pred)

    print(f"ELASTICNET MODEL (alpha={args.alpha}, l1_ratio={args.l1_ratio}):")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    print("Model logged successfully!")

