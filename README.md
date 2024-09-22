# Linear Regression for Real Estate Price Prediction: A Technical Overview

This repository presents a comprehensive set of exercises on using linear regression to predict real estate prices. Each exercise builds on the previous one, introducing new datasets, concepts, and methods to deepen the understanding of regression analysis.

## 1. Real Estate Price Prediction

In this exercise, we implement a basic linear regression model to predict real estate prices using a set of property features. The features include:

- **Property Size (in square meters)**
- **Number of Rooms**
- **Proximity to Public Amenities (in meters)**

### Dataset

The dataset consists of the following entries:

| Property Size (sqm) | Number of Rooms | Number of Bathrooms | Distance to Amenities (m) | Price (€) |
|---------------------|-----------------|---------------------|---------------------------|-----------|
| 120                 | 3               | 1                   | 500                       | 350,000   |
| 85                  | 2               | 1                   | 300                       | 250,000   |
| 200                 | 4               | 2                   | 1000                      | 650,000   |
| 150                 | 3               | 1                   | 400                       | 450,000   |


### Model Implementation

```r
import numpy as np

Real estate dataset: Features (x) and corresponding prices (y)
x = np.array([
    [120, 3, 1, 500],
    [85, 2, 1, 300],
    [200, 4, 2, 1000],
    [150, 3, 1, 400]
])
y = np.array([350000, 250000, 650000, 450000])

# Fit the linear regression model using least squares
c = np.linalg.lstsq(x, y, rcond=None)[0]

# Predict the property prices
predictions = x @ c
print(c)
print(predictions)
```

### Explanation of the Method

- **Feature Matrix (X)**: This matrix contains the independent variables (features) used to predict the price. Each row represents a property, and each column corresponds to a specific feature.

- **Target Vector (y)**: The dependent variable (price) for each property.

- **Coefficient Calculation**: The `np.linalg.lstsq` function computes the coefficients that minimize the sum of the squared differences between the predicted and actual prices.

- **Prediction**: Once the coefficients are computed, the property prices are predicted by multiplying the feature matrix by the coefficient vector.

## 2. Least Squares Regression

This exercise explores the mathematical basis of linear regression by focusing on the least squares method. It introduces a more detailed cabin dataset for regression analysis, and walks through the calculation of regression coefficients.

###Dataset

The dataset includes five cabins with various features:

| Cabin Size (sqm) | Sauna Size (sqm) | Distance to Water (m) | Number of Bathrooms | Proximity to Neighbors (m) | Price (€) |
|------------------|------------------|-----------------------|---------------------|----------------------------|-----------|
| 25               | 2                | 50                    | 1                   | 500                        | 127,900   |
| 39               | 3                | 10                    | 1                   | 1000                       | 222,100   |
| 13               | 2                | 13                    | 1                   | 1000                       | 143,750   |
| 82               | 5                | 20                    | 2                   | 120                        | 268,000   |
| 130              | 6                | 10                    | 2                   | 600                        | 460,700   |


### Model Implementation

```r
import numpy as np

# Cabin dataset: Features (X) and corresponding prices (y)
x = np.array([
    [25, 2, 50, 1, 500], 
    [39, 3, 10, 1, 1000], 
    [13, 2, 13, 1, 1000], 
    [82, 5, 20, 2, 120], 
    [130, 6, 10, 2, 600]
])
y = np.array([127900, 222100, 143750, 268000, 460700])

# Compute the coefficients using least squares method
c = np.linalg.lstsq(x, y, rcond=None)[0]

# Predict cabin prices
predicted_prices = x @ c
print(c)
print(predicted_prices)
```

### Explanation

- **Linear Model**: The relationship between the input features and the output (price) is assumed to be linear. Each feature contributes to the price by a fixed coefficient.

- **Least Squares Solution**: The least squares method finds the coefficients that minimize the squared error between the predicted prices and the actual prices in the dataset.

- **Interpretation**: The coefficients can be interpreted as the price impact per unit change in each feature (e.g., cabin size, distance to water, etc.).


## 3. Predictions with Additional Data
In this exercise, we extend the previous model by introducing additional data points. This allows us to evaluate how well the model scales with more data and examine the robustness of the predictions.

### Dataset

Additional cabins are introduced:

| Cabin Size (sqm) | Sauna Size (sqm) | Distance to Water (m) | Number of Bathrooms | Proximity to Neighbors (m) | Price (€) |
|------------------|------------------|-----------------------|---------------------|----------------------------|-----------|
| 115              | 6                | 10                    | 1                   | 550                        | 407,000   |


### Model Implementation

```r
import numpy as np
from io import StringIO

# Simulated input file containing the dataset
input_string = '''
25 2 50 1 500 127900
39 3 10 1 1000 222100
13 2 13 1 1000 143750
82 5 20 2 120 268000
130 6 10 2 600 460700
115 6 10 1 550 407000
'''

# Function to read and fit the model
def fit_model(input_file):
    data = np.genfromtxt(input_file, skip_header=0)
    x = data[:, :-1]
    y = data[:, -1]
    
    # Fit the linear model
    c = np.linalg.lstsq(x, y, rcond=None)[0]
    
    # Predict the prices
    predicted_prices = x @ c
    return c, predicted_prices

# Read data and run the model
input_file = StringIO(input_string)
coefficients, predictions = fit_model(input_file)
print(coefficients)
print(predictions)
```

### Explanation

- **Dataset Expansion**: We introduced one additional cabin to increase the dataset size and observe how it affects the regression results.

- **Prediction Accuracy**: With more data, the regression model becomes more robust, and the predictions are more reliable. This highlights the importance of having sufficient data for training regression models.


## 4. Training Data vs Test Data

In this final exercise, we explore the distinction between **training data** and **test data**, which is crucial for evaluating machine learning models. We train the model using one dataset and predict prices on another dataset that the model has not seen before.

### Training Dataset

| Cabin Size (sqm) | Sauna Size (sqm) | Distance to Water (m) | Number of Bathrooms | Proximity to Neighbors (m) | Price (€) |
|------------------|------------------|-----------------------|---------------------|----------------------------|-----------|
| 25               | 2                | 50                    | 1                   | 500                        | 127,900   |
| 39               | 3                | 10                    | 1                   | 1000                       | 222,100   |
| 13               | 2                | 13                    | 1                   | 1000                       | 143,750   |
| 82               | 5                | 20                    | 2                   | 120                        | 268,000   |
| 130              | 6                | 10                    | 2                   | 600                        | 460,700   |
| 115              | 6                | 10                    | 1                   | 550                        | 407,000   |


### Test Dataset

| Cabin Size (sqm) | Sauna Size (sqm) | Distance to Water (m) | Number of Bathrooms | Proximity to Neighbors (m) | Actual Price (€) |
|------------------|------------------|-----------------------|---------------------|----------------------------|------------------|
| 36               | 3                | 15                    | 1                   | 850                        | 196,000          |
| 75               | 5                | 18                    | 2                   | 540                        | 290,000          |


### Model Implementation

```r
import numpy as np

# Training data
x_train = np.array([
    [25, 2, 50, 1, 500],
    [39, 3, 10, 1, 1000],
    [13, 2, 13, 1, 1000],
    [82, 5, 20, 2, 120],
    [130, 6, 10, 2, 600],
    [115, 6, 10, 1, 550]
])
y_train = np.array([127900, 222100, 143750, 268000, 460700, 407000])

# Test data
x_test = np.array([
    [36, 3, 15, 1, 850],
    [75, 5, 18, 2, 540]
])
y_test = np.array([196000, 290000])

# Fit the model on training data
c = np.linalg.lstsq(x_train, y_train, rcond=None)[0]

# Predict on test data
predictions = x_test @ c
print(predictions)
```

### Explanation

- **Training vs Test Data**: In this exercise, the model is trained on the training data, and then it is used to predict the prices of properties in the test dataset.

- **Model Generalization**: This exercise shows how well the model can generalize to unseen data. The comparison of predicted prices with actual test data provides an estimate of the model’s performance on new data.

## Conclusion

This repository demonstrates the step-by-step application of linear regression using the least squares method for real estate price prediction. From understanding basic regression models to scaling them with more data and evaluating them using training and test datasets, these exercises lay the foundation for more advanced machine learning techniques.

These methods can be applied to various datasets, offering predictive solutions for many real-world applications, including real estate, finance, and healthcare. Future extensions could explore non-linear models or advanced machine learning algorithms like decision trees, random forests, and neural networks for improved accuracy and generalization.

## Acknowledgments

This project was inspired by the "Build AI" course, which provided the initial code framework and theoretical foundations. Building on this, the author independently developed and modified all core code implementations to meet the specific needs of the project.

While some algorithms and datasets are consistent with those from the course, the author has completely rewritten the content and explanations based on their own understanding to better align with the project’s objectives, enhancing both clarity and functionality.

I am sincerely grateful to the "Build AI" course team for providing invaluable resources. All borrowed content respects the rights of the original creators, and their contributions are acknowledged with deep appreciation.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


