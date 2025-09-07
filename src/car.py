import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso

pd.set_option('display.max_rows', None)     
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)
pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

df = pd.get_dummies(df, columns=['fuel','seller_type','transmission','owner'], drop_first=True)
df['log_price'] = np.log1p(df['selling_price'])

print(df.head())
correlation = df.drop(columns = 'name', axis = 1)
correlation = correlation.corr()

X = df.drop(['name', 'selling_price'], axis=1)
Y = df['selling_price']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

model = Lasso(fit_intercept=True)
model.fit(X_train, Y_train)

train_preds = model.predict(X_train)
r2 = r2_score(Y_train, train_preds)
mse = mean_squared_error(Y_train, train_preds)
print(f'R² на обучающей выборке: {r2:.4f}')
print(f'MSE на обучающей выборке: {mse:.2f}')


plt.scatter(Y_train, train_preds)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()

