# What's Next?
The final project of my Foundations of Data Science course was to predict the movement of stock returns. I applied ensemble methods to build models that (1) predict stock returns and (2) predict the movement of the stock.

## Regression Models (Predict Stock Returns)
In order to predict the stock price, I built regression models and measured their performance by the correlation of predicted values with the actual values in the test set.
* Forward Selection Regression
* Principal Component Regression
* Random Forest Regression

## Classification Models (Predict Stock Movements)
In order to predict the movement of the stock, I built classifier models and measured their performance by the percentage of erroneous predictions.
* Logistic Regression Model
* Support Vector Machine
* Decision Tree & Random Forest
* Voting Ensemble
* Boosting

#### Final Classification Model
Noticing that the individual models performed poorly, and realizing that the logistic regression model and the support vector machine with only the target company’s lagged returns as the predictor variables performed the best, I decided that the final model should only use the target company’s lagged returns as the predictor variables. 

Furthermore, financial theory suggests that prediction models will perform better over longer horizons. Hence, I decided to build a model that would not just predict the returns at t+1, but rather predict returns at even further out horizons.

First, I divided the lags into 4 windows, t to t-7, t-5 to t-12, t-8 to t-15 and t-13 to t-20. Then, I used these variables to predict the return at t+1 using  logistic regression models. 

Each of these models gave a prediction for the returns at t+1. I averaged the predictions, and if the average was >= 0.5, I would classify that as the stock returns being positive and negative otherwise.
