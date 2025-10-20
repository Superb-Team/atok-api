"""
Task Management Tools
CRUD operations for task management with PostgreSQL database
"""

from strands import tool
from typing import Optional
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

# Database connection string
DATABASE_URL = os.getenv('DATABASE_URL')
def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


@tool
def create_task(user_id: str, title: str, description: str = "", priority: str = "medium", status: str = "backlog") -> str:
    """
    Create a new task with title, description, and priority for a specific user.
    
    Args:
        user_id: User identifier
        title: Task title
        description: Detailed task description (optional)
        priority: Task priority (low, medium, high) - default: medium
        status: Task status (backlog, this_week, today, done) - default: backlog
    
    Returns:
        Confirmation message with task ID
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Insert task
        cur.execute("""
            INSERT INTO tasks (user_id, title, description, priority, status, position)
            VALUES (%s, %s, %s, %s, %s, 
                (SELECT COALESCE(MAX(position), 0) + 1 FROM tasks WHERE user_id = %s AND status = %s)
            )
            RETURNING id, title, priority, status
        """, (user_id, title, description, priority, status, user_id, status))
        
        task = cur.fetchone()
        conn.commit()
        
        cur.close()
        conn.close()
        
        return f"✓ Task created successfully!\nTask ID: {task['id']}\nTitle: {task['title']}\nPriority: {task['priority']}\nStatus: {task['status']}\nUser: {user_id}"
        
    except Exception as e:
        return f"✗ Error creating task: {str(e)}"


@tool
def read_task(user_id: str, task_id: Optional[int] = None, status: Optional[str] = None) -> str:
    """
    Read task details for a specific user. If task_id is provided, returns specific task. 
    Otherwise returns all user's tasks, optionally filtered by status.
    
    Args:
        user_id: User identifier
        task_id: Optional task ID to retrieve specific task
        status: Optional status filter (backlog, this_week, today, done)
    
    Returns:
        Task details or list of all user's tasks
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        if task_id:
            # Get specific task
            cur.execute("""
                SELECT id, title, description, priority, status, 
                       created_at, completed_at, due_date, tags
                FROM tasks
                WHERE user_id = %s AND id = %s AND is_deleted = false
            """, (user_id, task_id))
            
            task = cur.fetchone()
            cur.close()
            conn.close()
            
            if not task:
                return f"✗ Task {task_id} not found for user {user_id}."
            
            tags_str = ", ".join(task['tags']) if task['tags'] else "None"
            return f"""Task Details:
ID: {task['id']}
Title: {task['title']}
Description: {task['description'] or 'No description'}
Priority: {task['priority']}
Status: {task['status']}
Tags: {tags_str}
Due Date: {task['due_date'] or 'Not set'}
Created: {task['created_at']}
Completed: {task['completed_at'] or 'Not completed'}"""
        
        # Get all tasks (optionally filtered by status)
        if status:
            cur.execute("""
                SELECT id, title, priority, status, due_date, tags
                FROM tasks
                WHERE user_id = %s AND status = %s AND is_deleted = false
                ORDER BY position, created_at DESC
            """, (user_id, status))
        else:
            cur.execute("""
                SELECT id, title, priority, status, due_date, tags
                FROM tasks
                WHERE user_id = %s AND is_deleted = false
                ORDER BY 
                    CASE status
                        WHEN 'today' THEN 1
                        WHEN 'this_week' THEN 2
                        WHEN 'backlog' THEN 3
                        WHEN 'done' THEN 4
                    END,
                    position, created_at DESC
            """, (user_id,))
        
        tasks = cur.fetchall()
        cur.close()
        conn.close()
        
        if not tasks:
            status_msg = f" with status '{status}'" if status else ""
            return f"No tasks found for user {user_id}{status_msg}."
        
        # Group by status
        tasks_by_status = {}
        for task in tasks:
            task_status = task['status']
            if task_status not in tasks_by_status:
                tasks_by_status[task_status] = []
            
            status_icon = "✓" if task_status == "done" else "○"
            tags_str = f" [{', '.join(task['tags'])}]" if task['tags'] else ""
            due_str = f" (Due: {task['due_date'].strftime('%Y-%m-%d')})" if task['due_date'] else ""
            
            tasks_by_status[task_status].append(
                f"{status_icon} [#{task['id']}] {task['title']} - {task['priority']}{tags_str}{due_str}"
            )
        
        # Format output
        result = [f"Tasks for {user_id}:\n"]
        status_order = ['today', 'this_week', 'backlog', 'done']
        status_labels = {
            'today': '📅 TODAY',
            'this_week': '📆 THIS WEEK',
            'backlog': '📋 BACKLOG',
            'done': '✅ DONE'
        }
        
        for status_key in status_order:
            if status_key in tasks_by_status:
                result.append(f"\n{status_labels[status_key]}:")
                result.extend(tasks_by_status[status_key])
        
        return "\n".join(result)
        
    except Exception as e:
        return f"✗ Error reading tasks: {str(e)}"


@tool
def complete_task(user_id: str, task_id: int) -> str:
    """
    Mark a task as completed for a specific user.
    
    Args:
        user_id: User identifier
        task_id: Task ID to complete
    
    Returns:
        Confirmation message
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if task exists and get current status
        cur.execute("""
            SELECT id, title, status
            FROM tasks
            WHERE user_id = %s AND id = %s AND is_deleted = false
        """, (user_id, task_id))
        
        task = cur.fetchone()
        
        if not task:
            cur.close()
            conn.close()
            return f"✗ Task {task_id} not found for user {user_id}."
        
        if task['status'] == 'done':
            cur.close()
            conn.close()
            return f"Task #{task_id} '{task['title']}' is already completed."
        
        # Update task to done
        cur.execute("""
            UPDATE tasks
            SET status = 'done', completed_at = CURRENT_TIMESTAMP
            WHERE user_id = %s AND id = %s
            RETURNING id, title
        """, (user_id, task_id))
        
        updated_task = cur.fetchone()
        conn.commit()
        
        cur.close()
        conn.close()
        
        return f"✓ Task #{updated_task['id']} marked as completed!\nTitle: {updated_task['title']}\nUser: {user_id}"
        
    except Exception as e:
        return f"✗ Error completing task: {str(e)}"


@tool
def delete_task(user_id: str, task_id: int) -> str:
    """
    Delete a task (soft delete) for a specific user.
    
    Args:
        user_id: User identifier
        task_id: Task ID to delete
    
    Returns:
        Confirmation message
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Soft delete task
        cur.execute("""
            UPDATE tasks
            SET is_deleted = true
            WHERE user_id = %s AND id = %s AND is_deleted = false
            RETURNING id, title
        """, (user_id, task_id))
        
        task = cur.fetchone()
        
        if not task:
            cur.close()
            conn.close()
            return f"✗ Task {task_id} not found for user {user_id}."
        
        conn.commit()
        cur.close()
        conn.close()
        
        return f"✓ Task #{task['id']} deleted successfully!\nDeleted task: {task['title']}\nUser: {user_id}"
        
    except Exception as e:
        return f"✗ Error deleting task: {str(e)}"


@tool
def update_task_status(user_id: str, task_id: int, new_status: str) -> str:
    """
    Update task status (move between backlog, this_week, today, done).
    
    Args:
        user_id: User identifier
        task_id: Task ID to update
        new_status: New status (backlog, this_week, today, done)
    
    Returns:
        Confirmation message
    """
    valid_statuses = ['backlog', 'this_week', 'today', 'done']
    
    if new_status not in valid_statuses:
        return f"✗ Invalid status. Must be one of: {', '.join(valid_statuses)}"
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Update status
        cur.execute("""
            UPDATE tasks
            SET status = %s,
                completed_at = CASE WHEN %s = 'done' THEN CURRENT_TIMESTAMP ELSE NULL END
            WHERE user_id = %s AND id = %s AND is_deleted = false
            RETURNING id, title, status
        """, (new_status, new_status, user_id, task_id))
        
        task = cur.fetchone()
        
        if not task:
            cur.close()
            conn.close()
            return f"✗ Task {task_id} not found for user {user_id}."
        
        conn.commit()
        cur.close()
        conn.close()
        
        return f"✓ Task #{task['id']} status updated!\nTitle: {task['title']}\nNew Status: {task['status']}\nUser: {user_id}"
        
    except Exception as e:
        return f"✗ Error updating task status: {str(e)}"
