# app/utils.py

import os
import requests
import pandas as pd
from langchain_community.utilities.sql_database import SQLDatabase

CHINOOK_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

def get_db_connection(db_uri: str) -> SQLDatabase | None:
    """
    Establish a connection to a SQL database from a given URI.
    If the URI points to a local SQLite Chinook.db and it doesn't exist,
    it will attempt to download it automatically.
    """
    if db_uri.startswith("sqlite:///"):
        db_file = db_uri.split("sqlite:///")[1]
        if not os.path.exists(db_file) and db_file == "Chinook.db":
            print(f"Database file '{db_file}' not found. Attempting to download...")
            try:
                response = requests.get(CHINOOK_URL, timeout=30)
                response.raise_for_status()
                with open(db_file, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded and saved as {db_file}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download the file: {e}")
                return None
    try:
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        print(f"Failed to connect to the database: {e}")
        return None


def get_schema(db: SQLDatabase) -> str:
    """
    Returns the database schema (tables + columns) as a string.
    """
    try:
        return db.get_table_info()
    except Exception as e:
        return f"Error getting schema: {e}"


def is_safe_query(query: str) -> bool:
    """
    Basic safeguard: allow only SELECT statements.
    """
    return query.strip().upper().startswith("SELECT")


def execute_query(db: SQLDatabase, query: str):
    """
    Executes a query safely and returns results as a pandas DataFrame
    or error message.
    """
    if not is_safe_query(query):
        return {"error": "Only SELECT queries are allowed."}
    try:
        result = db.run(query)
        if isinstance(result, list):
            return pd.DataFrame(result)
        return result
    except Exception as e:
        return {"error": str(e)}

