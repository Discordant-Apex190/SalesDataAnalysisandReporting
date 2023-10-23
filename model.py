# The model I made in Jupyter Lab

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import cross_val_score
from sales_data_extractor import DatabaseConnector


class SalesAnalyzer:
    def __init__(self, config_file):
        self.model = None
        self.df = None
        self.db_connector = DatabaseConnector(config_file)
        self.x_columns = []  # Initialize x_columns
        self.y_column = ""  # Initialize y_column

    def fetch_sales_data(self, query):
        query_results = self.db_connector.execute_query(query)
        results, column_names = query_results
        self.df = pd.DataFrame(results, columns=column_names)

    def select_best_predictors(self, x_columns, y_column, n_features_to_select=2):
        self.x_columns = x_columns  # Store x_columns
        self.y_column = y_column  # Store y_column

        x = self.df[x_columns]
        y = self.df[y_column]

        model = RandomForestRegressor(n_estimators=100)

        rfe = RFE(model, n_features_to_select=n_features_to_select)
        rfe = rfe.fit(x, y)

        top_predictors = [col for col, support in zip(x_columns, rfe.support_) if support]

        return top_predictors

    def clean_data(self):
        self.df[self.x_columns] = self.df[self.x_columns].apply(pd.to_numeric, errors='coerce')
        self.df[self.y_column] = pd.to_numeric(self.df[self.y_column], errors='coerce')

    def perform_regression_analysis(self):
        model = RandomForestRegressor(n_estimators=100)
        x = self.df[self.x_columns]
        y = self.df[self.y_column]

        # Use cross-validation to assess model performance
        scores = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')

        # Calculate the RMSE based on the cross-validation scores
        rmse = np.sqrt(-scores.mean())  # Take the negative mean squared error
        print(f"Cross-Validated RMSE: {rmse:.2f}")

        # Fit the model on the entire dataset
        model.fit(x, y)
        self.model = model

    def evaluate_model(self):
        actual_values = self.df[self.y_column]
        predicted_values = self.model.predict(self.df[self.x_columns])

        rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
        mae = mean_absolute_error(actual_values, predicted_values)
        mape = mean_absolute_percentage_error(actual_values, predicted_values)
        r_squared = r2_score(actual_values, predicted_values)

        print("Predicted vs. Actual Values:")
        for i in range(0, 5):
            print(f"expected={actual_values[i]:.1f}, predicted={predicted_values[i]:.1f}")

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R-squared: {r_squared:.2f}")

    def visualize_predictions(self, num_data_points):
        self.df['ORDERDATE'] = pd.to_datetime(self.df['ORDERDATE'])  # Convert ORDERDATE to datetime
        actual_values = self.df[self.y_column]
        predicted_values = self.model.predict(self.df[self.x_columns])

        # Filter the data to show only the first 'num_data_points' values
        actual_values = actual_values[:num_data_points]
        predicted_values = predicted_values[:num_data_points]

        plt.figure(figsize=(12, 6))
        plt.plot(self.df['ORDERDATE'][:num_data_points], actual_values, label='Actual', marker='o', markersize=4,
                 linewidth=0.5)
        plt.plot(self.df['ORDERDATE'][:num_data_points], predicted_values, label='Predicted', marker='o', markersize=4,
                 linewidth=0.5)

        plt.title = "Actual vs. Predicted Sales Over Time"
        plt.xlabel = "Date"
        plt.ylabel = "Sales"
        plt.legend()
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()

    def close_connection(self):
        self.db_connector.close_connection()


if __name__ == "__main__":
    user_query = "SELECT * FROM sales"
    config_file = 'config.json'  # Specify the config file here
    sales_analyzer = SalesAnalyzer(config_file)
    sales_analyzer.fetch_sales_data(user_query)

    x_columns = ['QUANTITYORDERED', 'PRICEEACH']
    y_column = 'SALES'

    top_predictors = sales_analyzer.select_best_predictors(x_columns, y_column, n_features_to_select=2)

    sales_analyzer.clean_data()
    sales_analyzer.perform_regression_analysis()
    sales_analyzer.evaluate_model()
    sales_analyzer.visualize_predictions(num_data_points=25)

    sales_analyzer.close_connection()
