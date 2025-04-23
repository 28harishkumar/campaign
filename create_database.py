import sqlite3
import json
import os


def create_database():
    # Read the schema from JSON file
    with open("database_schema.json", "r") as f:
        schema = json.load(f)["database_schema"]

    # Connect to SQLite database (creates it if it doesn't exist)
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # Create tables based on schema
    for table_name, columns in schema.items():
        # Start building the CREATE TABLE statement
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ("

        # Add columns
        column_definitions = []
        for column_name, column_type in columns.items():
            # Convert JSON schema types to SQLite types
            sqlite_type = {
                "PK": "TEXT PRIMARY KEY",
                "FK": "TEXT",
                "varchar": "TEXT",
                "datetime": "TEXT",
                "date": "TEXT",
                "decimal": "REAL",
                "json": "TEXT",
            }.get(column_type, "TEXT")

            column_definitions.append(f"{column_name} {sqlite_type}")

        # Add foreign key constraints
        for column_name, column_type in columns.items():
            if column_type == "FK":
                # Find the referenced table (table that has this column as PK)
                ref_table = None
                for t, cols in schema.items():
                    if column_name.replace("_id", "") in t and "PK" in cols.values():
                        ref_table = t
                        break

                if ref_table:
                    column_definitions.append(
                        f"FOREIGN KEY ({column_name}) REFERENCES {ref_table}({column_name})"
                    )

        # Complete the CREATE TABLE statement
        create_table_sql += ", ".join(column_definitions) + ")"

        # Execute the CREATE TABLE statement
        cursor.execute(create_table_sql)

    # Commit changes and close connection
    conn.commit()
    conn.close()
    print("Database created successfully!")


if __name__ == "__main__":
    create_database()
