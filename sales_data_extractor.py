import mysql.connector
import json
import pandas as pd


class DatabaseConnector:
    def __init__(self, config_file_path):
        with open(config_file_path) as config_file:
            config_data = json.load(config_file)
            db_config = config_data['DBConnection']
            self.connection = mysql.connector.connect(
                host=db_config['host'],
                user=db_config['user'],
                password=db_config['password'],
                database=db_config['database']
            )

    # resultsog is the original results variable without usage
    def execute_query(self, query):
        cursor = self.connection.cursor()
        cursor.execute(query)
        resultsog = cursor.fetchall()
        cursor.close()
        return resultsog

    def close_connection(self):
        self.connection.close()


if __name__ == "__main__":
    # Create a DatabaseConnector instance with the path to your config file
    db_connector = DatabaseConnector('config.json')

    # Define your query
    user_query = str(input("Enter the query you would like to run: "))

    # Execute the query and get the results
    results = db_connector.execute_query(user_query)

    # Convert the results to a DataFrame
    df = pd.DataFrame(results)

    # Display the DataFrame (table) in PyCharm
    print(df)

    # Close the database connection
    db_connector.close_connection()
