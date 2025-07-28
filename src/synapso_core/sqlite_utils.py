import os
import sqlite3


def create_sqlite_db_if_not_exists(db_path):
    # Check if the database file already exists
    if not os.path.exists(db_path):
        # Connect to the database (this will create the file)
        conn = sqlite3.connect(db_path)
        conn.close()
        print(f"Database created at: {db_path}")
    else:
        print(f"Database already exists at: {db_path}")
