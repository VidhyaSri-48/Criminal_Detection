# dbHandler.py
import sqlite3
import os

DB_PATH = 'criminals.db'

def init_db():
    """Create the database and table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS criminals
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  father TEXT,
                  mother TEXT,
                  gender TEXT,
                  dob TEXT,
                  blood_group TEXT,
                  id_mark TEXT,
                  nationality TEXT,
                  religion TEXT,
                  crimes TEXT)''')
    conn.commit()
    conn.close()
    print("Database initialized.")

def insertData(data):
    """
    Insert criminal data into database.
    Returns row id on success, None on failure.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO criminals 
                     (name, father, mother, gender, dob, blood_group,
                      id_mark, nationality, religion, crimes)
                     VALUES (?,?,?,?,?,?,?,?,?,?)''',
                  (data['Name'].lower(),
                   data["Father's Name"].lower(),
                   data["Mother's Name"].lower(),
                   data['Gender'].lower(),
                   data['DOB'],
                   data['Blood Group'].lower(),
                   data['Identification Mark'].lower(),
                   data['Nationality'].lower(),
                   data['Religion'].lower(),
                   data['Crimes Done'].lower()))
        conn.commit()
        row_id = c.lastrowid
        conn.close()
        return row_id
    except Exception as e:
        print(f"Database insert error: {e}")
        return None

def retrieveData(name):
    """
    Retrieve criminal data by name (case‑insensitive).
    Returns (id, data_dict) or (None, None) if not found.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT * FROM criminals WHERE name=?", (name.lower(),))
        row = c.fetchone()
        conn.close()
        if row:
            # Map row to dict (adjust column order as per your table)
            columns = ['id', 'name', 'father', 'mother', 'gender', 'dob',
                       'blood_group', 'id_mark', 'nationality', 'religion', 'crimes']
            data = dict(zip(columns[1:], row[1:]))   # skip id
            return row[0], data
        else:
            return None, None
    except Exception as e:
        print(f"Database retrieve error: {e}")
        return None, None