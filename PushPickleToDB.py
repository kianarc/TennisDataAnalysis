import pickle
import pandas as pd
import psycopg2
from psycopg2 import sql
from pathlib import Path
import os

# Database connection parameters
DB_CONFIG = {
    'host': '87.248.130.241',
    'port': 5885,
    'database': 'cleaned_tennis_db',
    'user': 'postgres',
    'password': 'dEngR7eOfT33J84eds'
}

# Path to the folder containing pickle files
# Update this path to where you extracted the files manually
PICKLE_DIR = 'cleaned_data_pickle'

def load_pickle_files():
    """Load all pickle files into a dictionary"""
    pickle_data = {}
    pickle_dir = Path(PICKLE_DIR)
    
    for pkl_file in pickle_dir.glob('*.pkl'):
        table_name = pkl_file.stem.replace('cleaned_', '')
        print(f"Loading {pkl_file.name}...")
        
        with open(pkl_file, 'rb') as f:
            df = pickle.load(f)
            pickle_data[table_name] = df
            print(f"  Loaded {len(df)} rows for table '{table_name}'")
    
    return pickle_data

def get_postgresql_type(dtype):
    """Map pandas dtypes to PostgreSQL types"""
    dtype_str = str(dtype)
    
    if 'int' in dtype_str:
        return 'INTEGER'
    elif 'float' in dtype_str:
        return 'REAL'
    elif 'datetime' in dtype_str or 'timestamp' in dtype_str:
        return 'TIMESTAMP'
    elif 'date' in dtype_str:
        return 'DATE'
    elif 'bool' in dtype_str:
        return 'BOOLEAN'
    else:
        return 'TEXT'

def identify_primary_keys(table_name, df):
    """Identify likely primary key columns"""
    # Common primary key patterns
    pk_patterns = [f'{table_name}_id', 'id', f'{table_name}id']
    
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in pk_patterns:
            return col
    
    # Check for columns with unique values that might be IDs
    for col in df.columns:
        if 'id' in col.lower() and df[col].nunique() == len(df):
            return col
    
    return None

def identify_foreign_keys(table_name, df, all_tables):
    """Identify likely foreign key relationships"""
    foreign_keys = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check if column name suggests a foreign key
        for other_table in all_tables:
            if other_table == table_name:
                continue
            
            # Pattern: other_table_id or other_tableid
            fk_patterns = [f'{other_table}_id', f'{other_table}id']
            
            if col_lower in fk_patterns:
                # Verify the referenced table has a matching primary key
                pk = identify_primary_keys(other_table, all_tables[other_table])
                if pk:
                    foreign_keys.append({
                        'column': col,
                        'ref_table': other_table,
                        'ref_column': pk
                    })
    
    return foreign_keys

def create_tables(conn, pickle_data):
    """Create tables with primary keys and foreign keys"""
    cursor = conn.cursor()
    
    # First pass: Create tables with primary keys only
    print("\nCreating tables...")
    for table_name, df in pickle_data.items():
        print(f"\nCreating table: {table_name}")
        
        # Drop existing table
        cursor.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
            sql.Identifier(table_name)
        ))
        
        # Build CREATE TABLE statement
        columns = []
        pk_column = identify_primary_keys(table_name, df)
        
        for col in df.columns:
            pg_type = get_postgresql_type(df[col].dtype)
            col_def = f"{col} {pg_type}"
            
            # Add PRIMARY KEY constraint
            if col == pk_column:
                col_def += " PRIMARY KEY"
            
            columns.append(col_def)
        
        create_stmt = f"CREATE TABLE {table_name} ({', '.join(columns)})"
        cursor.execute(create_stmt)
        print(f"  Created with primary key: {pk_column}")
    
    conn.commit()
    
    # Second pass: Add foreign key constraints
    print("\nAdding foreign key constraints...")
    for table_name, df in pickle_data.items():
        fks = identify_foreign_keys(table_name, df, pickle_data)
        
        for fk in fks:
            try:
                alter_stmt = f"""
                ALTER TABLE {table_name}
                ADD CONSTRAINT fk_{table_name}_{fk['column']}
                FOREIGN KEY ({fk['column']})
                REFERENCES {fk['ref_table']}({fk['ref_column']})
                """
                cursor.execute(alter_stmt)
                print(f"  {table_name}.{fk['column']} -> {fk['ref_table']}.{fk['ref_column']}")
            except Exception as e:
                print(f"  Warning: Could not create FK {table_name}.{fk['column']}: {e}")
    
    conn.commit()
    cursor.close()

def insert_data(conn, pickle_data):
    """Insert data into tables"""
    cursor = conn.cursor()
    
    print("\nInserting data...")
    for table_name, df in pickle_data.items():
        print(f"Inserting {len(df)} rows into {table_name}...")
        
        # Replace NaN with None for proper NULL handling
        df = df.where(pd.notna(df), None)
        
        # Prepare INSERT statement
        cols = ', '.join(df.columns)
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_stmt = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
        
        # Insert in batches
        batch_size = 1000
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            values = [tuple(row) for row in batch.values]
            cursor.executemany(insert_stmt, values)
            conn.commit()
            print(f"  Inserted {min(i+batch_size, len(df))}/{len(df)} rows")
    
    cursor.close()

def main():
    try:
        # Load pickle files
        pickle_data = load_pickle_files()
        
        if not pickle_data:
            print("No pickle files found!")
            return
        
        # Connect to database
        print("\nConnecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected!")
        
        # Create tables with relationships
        create_tables(conn, pickle_data)
        
        # Insert data
        insert_data(conn, pickle_data)
        
        print("\n✓ Database import completed successfully!")
        print("\nYou can now generate an ERD diagram using tools like:")
        print("- pgAdmin (Tools -> ERD Tool)")
        print("- DBeaver (Database -> ER Diagram)")
        print("- dbdiagram.io (export schema and import)")
        
        conn.close()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()