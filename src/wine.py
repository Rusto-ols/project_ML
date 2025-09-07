import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine_dataset = pd.read_csv('winequality-red.csv')

print(wine_dataset.describe())
sns.set_theme(style='ticks')

graph = sns.catplot(x='quality', data=wine_dataset, kind='count')

correlation = wine_dataset.corr()

plt.show()

X = wine_dataset.drop(columns = 'quality', axis = 1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

model = RandomForestClassifier()
model.fit(X_train, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy : ', test_data_accuracy)

input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

input_data = (7.5,0.5,0.36,6.1,0.071,17.0,102.0,0.9978,3.35,0.8,10.5)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
if (prediction[0]==1):
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')
