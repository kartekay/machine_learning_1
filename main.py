import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
bike_database= pd.read_csv('Used_Bikes.csv')
bike_database.drop(columns=['owner','city','brand','bike_name'],inplace=True)
bike_database.info()
print(bike_database)
features_bike = bike_database[['kms_driven','age','power']]
prediction = bike_database[['price']]
X_train, X_test, y_train, y_test = train_test_split(features_bike, prediction, test_size=0.1)
k=7
final = KNeighborsRegressor(n_neighbors=k,weights='distance',metric='euclidean')
final.fit(X_train,y_train)
pred=final.predict(X_test)
score=final.score(X_test,y_test)
print("Scored accuracy:",score)
mse = mean_squared_error(y_test,pred)
print("Mean Squared Error:", mse)
plt.figure(figsize=(10,8))
plt.scatter(y_test, pred, color='black',label='predictions made')
plt.xlabel('Actual Price of Bike')
plt.ylabel('Predicted Price of Bike')
plt.title('Actual vs. Predicted Bike Prices')
ideal_min = min(min(y_test.values)[0], min(pred)[0]) - 200000
ideal_max = max(max(y_test.values)[0], max(pred)[0]) + 200000
plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], color='red', linestyle='--', linewidth=2, label='Ideal Line')
c=min(y_test.values)[0]
d=max(y_test.values)[0]
plt.xlim(c - 200000, d + 200000)
plt.ylim(ideal_min, ideal_max)
plt.legend()
plt.grid(True)
plt.show()
plt.subplot(2,1,1)
plt.bar(['Accuracy Score'],[score],color='blue')
plt.ylim(0,1)
plt.subplot(2,1,2)
plt.bar(['Mean Squared Error'],[mse],color='gray')
plt.grid(True)
plt.show()




