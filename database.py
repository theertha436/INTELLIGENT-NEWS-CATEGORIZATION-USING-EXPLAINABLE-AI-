# ============================================================
# DATABASE
# ============================================================

import sqlite3
import pandas as pd

from datetime import datetime


DB_NAME = "users.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    return sqlite3.connect(
        DB_NAME
    )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_database():

    conn = get_connection()

    cursor = conn.cursor()

    # --------------------------------------------------------
    # NEWS HISTORY
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS news_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            title TEXT,

            article_text TEXT NOT NULL,

            language TEXT DEFAULT 'English',

            source TEXT,

            created_at TEXT NOT NULL
        )
        """
    )

    # --------------------------------------------------------
    # ANALYSIS RESULTS
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_results (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            news_id INTEGER NOT NULL,

            category TEXT,

            sentiment TEXT,

            sentiment_score REAL,

            summary TEXT,

            fake_prediction TEXT,

            fake_confidence REAL,

            FOREIGN KEY (news_id)
                REFERENCES news_history(id)
                ON DELETE CASCADE
        )
        """
    )

    # --------------------------------------------------------
    # USER PREFERENCES
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_preferences (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER UNIQUE NOT NULL,

            preferred_language TEXT DEFAULT 'English',

            preferred_category TEXT DEFAULT 'All',

            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
        )
        """
    )

    conn.commit()

    conn.close()


# ============================================================
# SAVE NEWS
# ============================================================

def save_news(
    user_id,
    title,
    article_text,
    language="English",
    source=None
):

    conn = get_connection()

    cursor = conn.cursor()

    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute(
        """
        INSERT INTO news_history
        (
            user_id,
            title,
            article_text,
            language,
            source,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            title,
            article_text,
            language,
            source,
            created_at
        )
    )

    news_id = cursor.lastrowid

    conn.commit()

    conn.close()

    return news_id


# ============================================================
# SAVE ANALYSIS
# ============================================================

def save_analysis(
    news_id,
    category=None,
    sentiment=None,
    sentiment_score=None,
    summary=None,
    fake_prediction=None,
    fake_confidence=None
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO analysis_results
        (
            news_id,
            category,
            sentiment,
            sentiment_score,
            summary,
            fake_prediction,
            fake_confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            news_id,
            category,
            sentiment,
            sentiment_score,
            summary,
            fake_prediction,
            fake_confidence
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# GET USER NEWS
# ============================================================

def get_user_news(user_id):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            n.id,
            n.title,
            n.language,
            n.source,
            n.created_at,
            a.category,
            a.sentiment,
            a.sentiment_score,
            a.fake_prediction,
            a.fake_confidence,
            a.summary

        FROM news_history n

        LEFT JOIN analysis_results a
            ON n.id = a.news_id

        WHERE n.user_id = ?

        ORDER BY n.created_at DESC
        """,
        (user_id,)
    )

    results = cursor.fetchall()

    conn.close()

    return results


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

def get_dashboard_stats(user_id):

    conn = get_connection()

    cursor = conn.cursor()

    # Total articles
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM news_history
        WHERE user_id = ?
        """,
        (user_id,)
    )

    total_articles = cursor.fetchone()[0]

    # Category counts
    cursor.execute(
        """
        SELECT
            a.category,
            COUNT(*)

        FROM analysis_results a

        JOIN news_history n
            ON a.news_id = n.id

        WHERE n.user_id = ?

        GROUP BY a.category
        """,
        (user_id,)
    )

    category_counts = cursor.fetchall()

    # Sentiment counts
    cursor.execute(
        """
        SELECT
            a.sentiment,
            COUNT(*)

        FROM analysis_results a

        JOIN news_history n
            ON a.news_id = n.id

        WHERE n.user_id = ?

        GROUP BY a.sentiment
        """,
        (user_id,)
    )

    sentiment_counts = cursor.fetchall()

    # Fake news counts
    cursor.execute(
        """
        SELECT
            a.fake_prediction,
            COUNT(*)

        FROM analysis_results a

        JOIN news_history n
            ON a.news_id = n.id

        WHERE n.user_id = ?

        GROUP BY a.fake_prediction
        """,
        (user_id,)
    )

    fake_news_counts = cursor.fetchall()

    conn.close()

    return {

        "total_articles": total_articles,

        "category_counts": category_counts,

        "sentiment_counts": sentiment_counts,

        "fake_news_counts": fake_news_counts
    }


# ============================================================
# SAVE USER PREFERENCES
# ============================================================

def save_preferences(
    user_id,
    language="English",
    category="All"
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO user_preferences
        (
            user_id,
            preferred_language,
            preferred_category
        )

        VALUES (?, ?, ?)

        ON CONFLICT(user_id)

        DO UPDATE SET

            preferred_language =
                excluded.preferred_language,

            preferred_category =
                excluded.preferred_category
        """,
        (
            user_id,
            language,
            category
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# GET USER PREFERENCES
# ============================================================

def get_preferences(user_id):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            preferred_language,
            preferred_category

        FROM user_preferences

        WHERE user_id = ?
        """,
        (user_id,)
    )

    result = cursor.fetchone()

    conn.close()

    return result


# ============================================================
# GET ALL NEWS
# Used for Similar News Detection
# ============================================================

def get_all_news():

    conn = get_connection()

    query = """
        SELECT
            id,
            user_id,
            title,
            article_text,
            language,
            created_at

        FROM news_history

        ORDER BY created_at DESC
    """

    df = pd.read_sql_query(
        query,
        conn
    )

    conn.close()

    return df