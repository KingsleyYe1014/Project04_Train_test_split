# Project 4: Census Data Analysis and Modeling

## Introduction

This project analyzes the relationship between household income and various demographic factors using data from the U.S. Census Bureau. The goal is to predict median household income based on education levels and labor force participation rates across different states.

## Data Source and Collection

The data is collected from the U.S. Census Bureau's American Community Survey (ACS) 5-year estimates for 2021. The following variables are used:

- Median household income (B19013_001E)
- Education levels (B15003_022E through B15003_025E)
- Labor force participation (B23025_002E, B23025_007E)
- Total population (B01003_001E)

## Methodology

1. Data Collection:
   - Used the Census API to fetch data for all states
   - Calculated derived features:
     - Education level (percentage with bachelor's degree or higher)
     - Labor force participation rate
     - Per capita income

2. Model Implementation:
   - Split data into training (80%) and testing (20%) sets
   - Implemented two models:
     - Linear Regression
     - Decision Tree Regressor
   - Evaluated models using R² score and Mean Squared Error (MSE)
   - Analyzed feature importance and correlations

## Results

The analysis was performed on data from 52 states, and the following results were obtained:

### Model Performance

1. Linear Regression Model:
   - R² Score: 0.7563 (75.63%)
   - Mean Squared Error: 19,876,998.70
   - This model shows strong predictive power, explaining about 76% of the variance in household income.
   - Both features show significant positive coefficients, with labor force participation having a stronger effect.

2. Decision Tree Model:
   - R² Score: 0.2106 (21.06%)
   - Mean Squared Error: 64,391,949.09
   - This model performed significantly worse than the linear regression model.
   - Education level is slightly more important than labor force participation in this model.

### Feature Importance and Correlations

The analysis revealed several important relationships:

1. Education Level:
   - Shows strong positive correlation with household income (0.731)
   - Linear Regression coefficient: 107,138.41
   - Decision Tree importance: 0.5841
   - Higher education levels consistently associated with higher incomes

2. Labor Force Participation:
   - Strongest correlation with household income (0.804)
   - Linear Regression coefficient: 175,054.20
   - Decision Tree importance: 0.4159
   - Most significant predictor in both models

3. Per Capita Income:
   - Moderate correlation with household income (0.261)
   - Provides additional context for income distribution
   - Shows weaker relationship than education and labor force participation

### Visualization Results

The project generated several visualization files:

1. Model Predictions:
   - `linear_regression_predictions.png`: Shows the relationship between actual and predicted values for the linear regression model
   - `decision_tree_predictions.png`: Shows the relationship between actual and predicted values for the decision tree model

2. Feature Analysis:
   - `correlation_heatmap.png`: Displays the correlation matrix between key variables
   - `feature_importance.png`: Compares the importance of features in both models

## Discussion

### Model Performance Analysis
The linear regression model outperformed the decision tree model by a significant margin. This suggests that the relationship between household income and our selected features (education level and labor force participation rate) is more linear than non-linear. The high R² score of 0.7563 indicates that these two features alone can explain a substantial portion of the variation in household income across states.

### Key Findings
1. Education is the strongest predictor of household income
2. Labor force participation has a significant but secondary effect
3. The relationship between these factors and income is primarily linear
4. State-level variations show clear patterns in income distribution

### Limitations
1. Limited to state-level data
2. Only considers two main features
3. May not capture complex interactions between variables
4. Decision tree model's poor performance suggests that the relationship might be too simple for a tree-based approach
5. Does not account for regional cost of living differences

### Future Improvements
1. Include more demographic variables such as:
   - Age distribution
   - Industry composition
   - Housing costs
   - Cost of living
2. Try more sophisticated models:
   - Random Forest
   - Gradient Boosting
   - Neural Networks
3. Use county-level data for more granular analysis
4. Add feature engineering for better predictive power
5. Consider non-linear transformations of features
6. Include temporal analysis to track changes over time

## Conclusion

The analysis demonstrates that education level and labor force participation rate are strong predictors of household income at the state level. The linear regression model achieved a good fit with an R² score of 0.7563, indicating that these two factors alone can explain about 76% of the variation in household income across states. This finding has important implications for understanding the relationship between education, employment, and income at the state level.

The project successfully implemented the train-test split methodology and compared different modeling approaches. While the linear regression model performed well, the decision tree model's poor performance suggests that the relationship between our features and the target variable is primarily linear in nature.

The results highlight the importance of education in determining household income and suggest that policies aimed at improving educational attainment could have significant impacts on income levels. The strong linear relationship also indicates that simple, interpretable models can be effective for understanding and predicting household income patterns at the state level. 