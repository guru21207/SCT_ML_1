import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/train.csv")

X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

comparison =pd.DataFrame({"Actual":y_test,"Predicted": y_pred})
from  sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test,y_pred)
print("MAE: ",mae)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R²:", r2)

import numpy as np
rmse = np.sqrt(mse)
print("RMSE:", rmse)

print("Intercept:", model.intercept_) 
for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)
print("Score:", model.score(X_test, y_test))

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="#008000"
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

import joblib
joblib.dump(model, "model/house_price_model.pkl")

