import numpy as np
import pandas as pd
import matplotlib as mptlb
import matplotlib.pyplot as plt
import random as rn

d = pd.read_csv('/kaggle/input/real-estate-price-prediction/Real estate.csv')

d.head()
d.shape
hse_age = np.array(d['X2 house age'])
hse_age = (hse_age - np.mean(hse_age)) / np.std(hse_age)
cnv_stre = np.array(d['X4 number of convenience stores'])
cnv_stre = (cnv_stre - np.mean(cnv_stre)) / np.std(cnv_stre)
#to normalize the data
latt = np.array(d['X5 latitude'] / 10000)
longt = np.array(d['X6 longitude'] / 10000)
n = len(longt)
y_price = np.array(d['Y house price of unit area'])
y_price = ((y_price) - np.mean(y_price)) / np.std(y_price)
y_pred = w1 * hse_age + w2 * cnv_stre + b

# to plot bar graph
plt.bar(cnv_stre, y_price, color ='maroon', width = 0.4)
 
plt.xlabel("House Price")
plt.ylabel("House Age")
plt.title("Linear Regression")
plt.show()

# to plot scatterplot
plt.scatter(hse_age, y_price)
plt.show()

plt.plot(cnv_stre, y_price, color = "green")
plt.show()

# gradient descent algorithm
alpha = 0.0000001
cost_fn = []
for i in range(10000000):
    w1_derivative = (1 / n) * np.sum((y_pred - y_price) * hse_age)
    w2_derivative = (1 / n) * np.sum((y_pred - y_price) * cnv_stre)
    #w3_derivative = (1 / n) * np.sum((y_pred - y_price) * latt)
    #w4_derivative = (1 / n) * np.sum((y_pred - y_price) * longt)
    b_derivative = (1 / n) * np.sum(y_pred - y_price)
    
    w1 = w1 + alpha * w1_derivative
    w2 = w2 + alpha * w2_derivative
    #w3 = w3 + w3_derivative
    #w4 = w4 + w4_derivative
    b = b + b_derivative
    cost_fn.append(i)

y_pred = ((w1 * hse_age[4] + w2 * cnv_stre[4]) + b)
y_pred
