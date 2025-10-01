# app/utils.py

import os
import requests
from langchain_community.utilities import SQLDatabase

def get_db_connection(db_uri: str):
    """
    Establishes a connection to a SQL database from a given URI.
    If the URI points to a local SQLite file that doesn't exist, it attempts
    to download the sample Chinook.db.

    Args:
        db_uri (str): The database URI.

    Returns:
        An instance of SQLDatabase or None if connection fails.
    """
    if db_uri.startswith("sqlite:///"):
        db_file = db_uri.split("sqlite:///")[1]
        if not os.path.exists(db_file) and db_file == "Chinook.db":
            print(f"Database file '{db_file}' not found. Attempting to download.")
            url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(db_file, "wb") as file:
                    file.write(response.content)
                print(f"File downloaded and saved as {db_file}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download the file: {e}")
                return None
    try:
        db = SQLDatabase.from_uri(db_uri)
        return db
    except Exception as e:
        print(f"Failed to connect to the database: {e}")
        return None
