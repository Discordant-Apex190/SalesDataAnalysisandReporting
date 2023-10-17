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

    def execute_query(self, query):
        cursor = self.connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        cursor.close()
        return results, column_names

    def close_connection(self):
        self.connection.close()


if __name__ == "__main__":
    # Create a DatabaseConnector instance with the path to your config file
    db_connector = DatabaseConnector('config.json')

    # Define your query
    user_query = str(input("Enter the query you would like to run: "))

    # Execute the query and get the results and column names
    query_results = db_connector.execute_query(user_query)

    # Unpack the results and column names from the tuple
    results, column_names = query_results

    # Convert the results and column names to a DataFrame
    df = pd.DataFrame(results, columns=column_names)

    # Set the option to display all columns
    pd.set_option('display.max_columns', None)

    # Display the DataFrame (table) in PyCharm
    print(df.head(6))

    # Close the database connection
    db_connector.close_connection()
