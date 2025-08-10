#!/usr/bin/env python3
# noinspection SqlNoDataSourceInspection,SqlResolve,PyUnresolvedReferences
"""
Database initialization script - PyCharm compatible
Run this first to create the database and tables
"""

import sqlite3
import datetime
import json
from pathlib import Path


def create_tables(conn):
    """Create database tables with clean SQL"""

    # Artifacts table
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS artifacts
                 (
                     id            TEXT PRIMARY KEY,
                     title         TEXT NOT NULL,
                     artifact_type TEXT NOT NULL,
                     language      TEXT,
                     tags          TEXT,
                     created       TEXT NOT NULL,
                     modified      TEXT NOT NULL,
                     size          INTEGER DEFAULT 0,
                     checksum      TEXT,
                     chat_context  TEXT,
                     project       TEXT    DEFAULT 'default',
                     favorite      INTEGER DEFAULT 0,
                     version       INTEGER DEFAULT 1,
                     parent_id     TEXT,
                     filepath      TEXT
                 )
                 """)

    # Projects table
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS projects
                 (
                     name        TEXT PRIMARY KEY,
                     description TEXT,
                     created     TEXT NOT NULL,
                     color       TEXT DEFAULT '#007acc'
                 )
                 """)

    # Usage stats table
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS usage_stats
                 (
                     id          INTEGER PRIMARY KEY AUTOINCREMENT,
                     artifact_id TEXT NOT NULL,
                     accessed    TEXT NOT NULL,
                     action      TEXT NOT NULL
                 )
                 """)


def create_indexes(conn):
    """Create database indexes"""

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_artifacts_tags ON artifacts(tags)",
        "CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project)",
        "CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type)",
        "CREATE INDEX IF NOT EXISTS idx_artifacts_language ON artifacts(language)",
        "CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts(created)",
        "CREATE INDEX IF NOT EXISTS idx_usage_stats_artifact ON usage_stats(artifact_id)"
    ]

    for index_sql in indexes:
        conn.execute(index_sql)


def insert_default_data(conn):
    """Insert default project and sample artifact"""

    current_time = datetime.datetime.now().isoformat()

    # Insert default project
    conn.execute(
        "INSERT OR IGNORE INTO projects (name, description, created) VALUES (?, ?, ?)",
        ('default', 'Default project for artifacts', current_time)
    )

    # Sample artifact content
    sample_content = '''print("Hello from Claude Artifact Manager!")

def welcome():
    """Sample function to test the artifact system"""
    return "Welcome to the robust artifact management system!"

if __name__ == "__main__":
    print(welcome())
'''

    # Generate sample artifact data
    sample_id = f"sample_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sample_tags = json.dumps(["sample", "welcome", "python"])
    sample_filepath = f"files/{sample_id}.py"

    # Insert sample artifact
    artifact_data = (
        sample_id,
        "Sample Python Script",
        "code",
        "python",
        sample_tags,
        current_time,
        current_time,
        len(sample_content),
        "sample_checksum",
        "Initial sample artifact to demonstrate the system",
        "default",
        sample_filepath
    )

    conn.execute("""
                 INSERT OR IGNORE INTO artifacts
                 (id, title, artifact_type, language, tags, created, modified,
                  size, checksum, chat_context, project, filepath)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                 """, artifact_data)

    return sample_id, sample_content


def create_directories(storage_dir):
    """Create required directories"""

    directories = ["files", "backups", "exports", "temp"]

    for dir_name in directories:
        dir_path = storage_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_path}")


def init_database(storage_path="claude_artifacts"):
    """Initialize the artifact manager database"""

    try:
        # Setup storage directory
        storage_dir = Path(storage_path)
        storage_dir.mkdir(exist_ok=True)
        print(f"📁 Storage directory: {storage_dir}")

        # Create subdirectories
        create_directories(storage_dir)

        # Database path
        db_path = storage_dir / "artifacts.db"
        print(f"🔧 Initializing database: {db_path}")

        # Connect to database
        with sqlite3.connect(str(db_path)) as conn:
            # Configure SQLite
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")

            # Create schema
            create_tables(conn)
            create_indexes(conn)

            # Insert default data
            sample_id, sample_content = insert_default_data(conn)

            # Save sample file
            sample_file_path = storage_dir / "files" / f"{sample_id}.py"
            try:
                with open(sample_file_path, 'w', encoding='utf-8') as f:
                    f.write(sample_content)
                print(f"✅ Created sample file: {sample_file_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not create sample file: {e}")

            # Commit all changes
            conn.commit()

        print("✅ Database initialized successfully!")
        print(f"✅ Sample artifact: {sample_id}")

        # Verify setup
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM artifacts")
            artifact_count = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM projects")
            project_count = cursor.fetchone()[0]

        print(f"📊 Artifacts: {artifact_count}")
        print(f"📊 Projects: {project_count}")

        print("\n🎯 Next steps:")
        print("1. Test: python artifact_manager.py --test")
        print("2. Stats: python artifact_manager.py --stats")
        print("3. Connect PyCharm Database tool to:", str(db_path))

        return True

    except PermissionError as e:
        print(f"❌ Permission denied: {e}")
        print("💡 Try running as administrator or check folder permissions")
        return False

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        print("💡 Database file might be locked or corrupted")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Check Python installation and dependencies")
        return False


def main():
    """Main function with proper error handling"""

    print("🚀 Claude Artifact Manager - Database Initialization")
    print("=" * 60)

    success = init_database()

    if success:
        print("\n🎉 Initialization completed successfully!")
        print("The artifact management system is ready to use.")
    else:
        print("\n❌ Initialization failed!")
        print("Please check the error messages above.")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)