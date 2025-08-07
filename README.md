This project demonstrates Simple and Multiple Linear Regression using the Housing.csv dataset. The goal is to predict house prices based on features such as area, number of bedrooms, and amenities using Python libraries like Scikit-learn, Pandas, and Matplotlib.

Libraries Used
pandas – for data loading and manipulation

scikit-learn – for model training, preprocessing, and evaluation

matplotlib – for plotting the regression line

 Project Steps
1. Load & Preprocess the Data
Read the Housing.csv file

Encode categorical variables using OneHotEncoder

2. Split the Dataset
Split into training and test sets using train_test_split

 3. Multiple Linear Regression
Train a model using all available features

Evaluate using:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

R² Score

Print model intercept and coefficients

4. Simple Linear Regression
Train a model using only one feature: area

Plot the regression line against actual values

Print the intercept and slope
