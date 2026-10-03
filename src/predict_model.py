import pandas as pd
import joblib

model = joblib.load("model/house_price_model.pkl")

test_data = pd.read_csv("dataset/test.csv")

X_test = test_data[["GrLivArea", "BedroomAbvGr", "FullBath"]]

predictions = model.predict(X_test)

submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": predictions
})

submission.to_csv("submission.csv", index=False)

print(submission.head())