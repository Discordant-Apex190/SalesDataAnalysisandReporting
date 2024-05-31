# Sales Data Analysis and Reporting with Sales Prediction

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tools Used](#tools-used)
3. [Project Phases](#project-phases)

## Project Overview

Welcome to the Sales Data Analysis and Reporting project repository. This project aims to provide a comprehensive analysis of historical sales data, visualize sales trends, and make predictions for future sales performance.

## Tools Used

- **Python**: For data analysis, preprocessing, and machine learning.
- **SQL**: For data storage and retrieval.
- **JupyterLab**: For data analysis and exploration.
- **Excel** : For data preprocessing

## Project Phases

### Data Collection and Preparation

- **Data Collection**: We sourced a real-world sales order dataset from Kaggle, a reputable platform for data sets. The dataset contained historical sales order information, providing a realistic basis for our analysis.

- **Data Cleaning and Transformation**: The raw data from Kaggle required initial data cleaning. We performed tasks like handling missing values, addressing duplicates, and ensuring consistent date formatting. This clean data was vital for accurate analysis and predictions.

- **Data Import into MySQL**: We created a MySQL database named `sales_analysis` and established a table within it, named `sales`. This SQL database served as a structured repository for our cleaned sales data. We employed MySQL's data import capabilities to efficiently load our data into the database, ensuring its organized storage.

### Data Analysis with Python

- Use JupyterLab to perform data analysis.
- Explore the dataset using Python libraries like pandas, and matplotlib
- Perform statistical analysis to gain insights into the dataset.

### Data Visualization with Python

- Create visualizations using libraries like and matplotlib to represent my findings.

### Sales Prediction with Random Forest Regression

- Choose Random Forest Regression as the regression model for sales prediction. Random Forest Regression is a powerful machine learning algorithm known for its robustness and ability to handle complex relationships in the data.
- Preprocess data for machine learning, including feature engineering and data splitting.
- Train and evaluate the Random Forest Regression model using Python libraries such as scikit-learn.
