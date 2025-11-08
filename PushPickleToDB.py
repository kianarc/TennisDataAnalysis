import pickle
import pandas as pd
import numpy as np
import psycopg2
from psycopg2 import sql
from pathlib import Path
import os

# Database connection parameters
DB_CONFIG = {
    'host': '87.248.130.241',
    'port': 5885,
    'database': 'cleaned_tennis_db',
    'user': 'developer',
    'password': 'm4dtls64soe'
}

# Path to the folder containing pickle files
PICKLE_DIR = 'cleaned_data_pickle'

# Define the correct table relationships based on the ERD
TABLE_RELATIONSHIPS = {
    'matcheventinfo': {
        'pk': 'match_id',
        'references': []
    },
    'gameinfo': {
        'pk': None,
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    },
    'matchawayteaminfo': {
        'pk': 'match_id',
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    },
    'matchhometeaminfo': {
        'pk': 'match_id',
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    },
    'matchroundinfo': {
        'pk': 'match_id',
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    },
    'matchseasoninfo': {
        'pk': 'match_id',
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    },
    'matchtimeinfo': {
        'pk': 'match_id',
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    },
    'matchtournamentinfo': {
        'pk': 'match_id',
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    },
    'matchvenueinfo': {
        'pk': 'match_id',
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    },
    'matchvotesinfo': {
        'pk': 'match_id',
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    },
    'oddsinfo': {
        'pk': None,
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    },
    'periodinfo': {
        'pk': None,
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    },
    'powerinfo': {
        'pk': None,
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    },
    'matchwayteaminfo': {
        'pk': 'match_id',
        'references': [('match_id', 'matcheventinfo', 'match_id')]
    }
}

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
    
    if 'int64' in dtype_str or 'int32' in dtype_str:
        return 'INTEGER'
    elif 'int' in dtype_str:
        return 'INTEGER'
    elif 'float' in dtype_str:
        return 'DOUBLE PRECISION'
    elif 'datetime' in dtype_str or 'timestamp' in dtype_str:
        return 'TIMESTAMP'
    elif 'date' in dtype_str:
        return 'DATE'
    elif 'bool' in dtype_str:
        return 'BOOLEAN'
    else:
        return 'TEXT'

def convert_to_python_types(value):
    """Convert numpy types to native Python types"""
    import numpy as np
    
    # Handle None and NaN first
    if value is None:
        return None
    if pd.isna(value):
        return None
    
    # Convert numpy types
    if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
        return float(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, str):
        return value
    else:
        return value

def get_table_order(pickle_data):
    """Determine the correct order to create tables (parent tables first)"""
    # matcheventinfo must be created first as it's the parent
    order = ['matcheventinfo']
    
    # Add all other tables
    for table_name in pickle_data.keys():
        if table_name not in order:
            order.append(table_name)
    
    return order

def create_tables(conn, pickle_data):
    """Create tables with primary keys and foreign keys"""
    cursor = conn.cursor()
    
    # Get correct table creation order
    table_order = get_table_order(pickle_data)
    
    # First pass: Create tables with primary keys only
    print("\nCreating tables...")
    for table_name in table_order:
        if table_name not in pickle_data:
            continue
            
        df = pickle_data[table_name]
        print(f"\nCreating table: {table_name}")
        
        # Drop existing table
        cursor.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
            sql.Identifier(table_name)
        ))
        
        # Build CREATE TABLE statement
        columns = []
        pk_column = TABLE_RELATIONSHIPS.get(table_name, {}).get('pk')
        
        for col in df.columns:
            pg_type = get_postgresql_type(df[col].dtype)
            col_def = f'"{col}" {pg_type}'
            
            # Add PRIMARY KEY constraint
            if col == pk_column:
                col_def += " PRIMARY KEY"
            
            columns.append(col_def)
        
        create_stmt = f"CREATE TABLE {table_name} ({', '.join(columns)})"
        cursor.execute(create_stmt)
        
        if pk_column:
            print(f"  Created with primary key: {pk_column}")
        else:
            print(f"  Created without primary key")
    
    conn.commit()
    
    # Second pass: Add foreign key constraints
    print("\nAdding foreign key constraints...")
    for table_name in table_order:
        if table_name not in pickle_data:
            continue
            
        relationships = TABLE_RELATIONSHIPS.get(table_name, {}).get('references', [])
        
        for fk_col, ref_table, ref_col in relationships:
            try:
                # Check if both tables exist and have the required columns
                if ref_table not in pickle_data:
                    print(f"  Warning: Referenced table {ref_table} not found, skipping FK")
                    continue
                
                if fk_col not in pickle_data[table_name].columns:
                    print(f"  Warning: Column {fk_col} not found in {table_name}, skipping FK")
                    continue
                
                alter_stmt = f"""
                ALTER TABLE {table_name}
                ADD CONSTRAINT fk_{table_name}_{fk_col}
                FOREIGN KEY ("{fk_col}")
                REFERENCES {ref_table}("{ref_col}")
                ON DELETE CASCADE
                """
                cursor.execute(alter_stmt)
                print(f"  ✓ {table_name}.{fk_col} -> {ref_table}.{ref_col}")
            except Exception as e:
                print(f"  ✗ Could not create FK {table_name}.{fk_col}: {e}")
    
    conn.commit()
    cursor.close()

def insert_data(conn, pickle_data):
    """Insert data into tables"""
    cursor = conn.cursor()
    
    # Get correct insertion order (parent tables first)
    table_order = get_table_order(pickle_data)
    
    print("\nInserting data...")
    for table_name in table_order:
        if table_name not in pickle_data:
            continue
            
        df = pickle_data[table_name].copy()
        print(f"Inserting {len(df)} rows into {table_name}...")
        
        # Convert all numpy types to Python native types for each column
        for col in df.columns:
            if df[col].dtype == 'object':
                # For object columns, apply conversion element by element
                df[col] = df[col].apply(lambda x: convert_to_python_types(x))
            elif pd.api.types.is_integer_dtype(df[col]):
                # Convert integer columns
                df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else None)
            elif pd.api.types.is_float_dtype(df[col]):
                # Convert float columns
                df[col] = df[col].apply(lambda x: float(x) if pd.notna(x) else None)
            elif pd.api.types.is_bool_dtype(df[col]):
                # Convert boolean columns
                df[col] = df[col].apply(lambda x: bool(x) if pd.notna(x) else None)
        
        # Prepare INSERT statement with quoted column names
        cols = ', '.join([f'"{col}"' for col in df.columns])
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_stmt = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
        
        # Insert in batches
        batch_size = 1000
        successful_rows = 0
        for i in range(0, len(df), batch_size):
            try:
                batch = df.iloc[i:i+batch_size]
                # Convert to list of tuples with explicit type conversion
                values = []
                for _, row in batch.iterrows():
                    row_values = tuple(convert_to_python_types(v) for v in row.values)
                    values.append(row_values)
                
                cursor.executemany(insert_stmt, values)
                conn.commit()
                successful_rows += len(batch)
                print(f"  Inserted {min(i+batch_size, len(df))}/{len(df)} rows")
            except Exception as e:
                conn.rollback()
                print(f"  Error inserting batch at row {i}: {e}")
                # Try inserting one by one for this batch
                for j, (_, row) in enumerate(batch.iterrows()):
                    try:
                        row_values = tuple(convert_to_python_types(v) for v in row.values)
                        cursor.execute(insert_stmt, row_values)
                        conn.commit()
                        successful_rows += 1
                    except Exception as e2:
                        conn.rollback()
                        print(f"    Skipped row {i+j}: {e2}")
        
        print(f"  Successfully inserted {successful_rows}/{len(df)} rows")
    
    cursor.close()

def verify_relationships(conn):
    """Verify that foreign key relationships are created correctly"""
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("VERIFYING FOREIGN KEY RELATIONSHIPS")
    print("="*60)
    
    cursor.execute("""
        SELECT 
            tc.table_name, 
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name 
        FROM information_schema.table_constraints AS tc 
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY' 
        AND tc.table_schema = 'public'
        ORDER BY tc.table_name;
    """)
    
    fks = cursor.fetchall()
    
    if fks:
        print(f"\nFound {len(fks)} foreign key relationships:\n")
        for fk in fks:
            print(f"  {fk[0]}.{fk[1]} -> {fk[2]}.{fk[3]}")
    else:
        print("\n⚠ WARNING: No foreign key relationships found!")
    
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
        
        # Verify relationships
        verify_relationships(conn)
        
        print("\n" + "="*60)
        print("[SUCCESS] Database import completed!")
        print("="*60)
        print("\nYou can now generate an ERD diagram using:")
        print("  - pgAdmin (Tools -> ERD Tool)")
        print("  - DBeaver (Database -> ER Diagram)")
        print("  - dbdiagram.io (export schema and import)")
        
        conn.close()
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()