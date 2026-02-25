import os
import psycopg2
import psycopg2.extras

def get_connection():
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise Exception("DATABASE_URL is not set")

    conn = psycopg2.connect(
        database_url,
        cursor_factory=psycopg2.extras.RealDictCursor
    )

    return conn