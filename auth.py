import sqlite3
import hashlib
import secrets
import re


DATABASE = "users.db"


# ==========================================
# DATABASE INITIALIZATION
# ==========================================

def initialize_database():

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# ==========================================
# PASSWORD HASHING
# ==========================================

def hash_password(password, salt=None):

    if salt is None:
        salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    )

    return password_hash.hex(), salt


# ==========================================
# PASSWORD VERIFICATION
# ==========================================

def verify_password(password, stored_hash, salt):

    password_hash, _ = hash_password(
        password,
        salt
    )

    return secrets.compare_digest(
        password_hash,
        stored_hash
    )


# ==========================================
# PHONE VALIDATION
# ==========================================

def valid_phone(phone):

    phone = phone.strip()

    # Accepts 10-15 digit numbers
    pattern = r"^\+?[0-9]{10,15}$"

    return bool(
        re.fullmatch(pattern, phone)
    )


# ==========================================
# CREATE ACCOUNT
# ==========================================

def create_user(name, phone, password):

    name = name.strip()
    phone = phone.strip()

    if name == "":
        return False, "Please enter your name."

    if not valid_phone(phone):
        return False, "Please enter a valid phone number."

    if len(password) < 6:
        return False, "Password must contain at least 6 characters."

    password_hash, salt = hash_password(password)

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    try:

        cursor.execute("""
            INSERT INTO users
            (name, phone, password_hash, salt)
            VALUES (?, ?, ?, ?)
        """, (
            name,
            phone,
            password_hash,
            salt
        ))

        connection.commit()

        return True, "Account created successfully!"

    except sqlite3.IntegrityError:

        return False, "This phone number is already registered."

    finally:

        connection.close()


# ==========================================
# LOGIN
# ==========================================

def login_user(phone, password):

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, name, phone, password_hash, salt
        FROM users
        WHERE phone = ?
    """, (phone.strip(),))

    user = cursor.fetchone()

    connection.close()

    if user is None:
        return None

    user_id, name, phone, password_hash, salt = user

    if verify_password(
        password,
        password_hash,
        salt
    ):

        return {
            "id": user_id,
            "name": name,
            "phone": phone
        }

    return None