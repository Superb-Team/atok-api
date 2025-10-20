"""
Database helper functions
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def get_github_token(user_id: str) -> Optional[str]:
    """
    Get GitHub access token for user from mcp_auth table
    
    Args:
        user_id: User identifier
    
    Returns:
        GitHub access token or None if not found
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT access_token
            FROM mcp_auth
            WHERE user_id = %s AND provider = 'github'
        """, (user_id,))
        
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result:
            return result['access_token']
        return None
        
    except Exception as e:
        print(f"Error fetching GitHub token: {e}")
        return None


def verify_user(user_id: str) -> bool:
    """
    Verify if user exists in database
    
    Args:
        user_id: User identifier
    
    Returns:
        True if user exists, False otherwise
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id FROM users WHERE id = %s
        """, (user_id,))
        
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        return result is not None
        
    except Exception as e:
        print(f"Error verifying user: {e}")
        return False


def get_user(user_id: str) -> Optional[dict]:
    """
    Get user information from database
    
    Args:
        user_id: User identifier
    
    Returns:
        User dict or None if not found
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, email, username, full_name, avatar_url, is_active, is_verified
            FROM users
            WHERE id = %s
        """, (user_id,))
        
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        return dict(result) if result else None
        
    except Exception as e:
        print(f"Error fetching user: {e}")
        return None
