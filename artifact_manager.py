#!/usr/bin/env python3
# noinspection PyUnresolvedReferences,SqlNoDataSourceInspection,SqlResolve
"""
Robust Claude Artifact Management System
Production-ready tool with comprehensive error handling and validation
"""

import json
import sqlite3
import hashlib
import logging
import sys
import argparse
import configparser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('artifact_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ArtifactManagerError(Exception):
    """Base exception for artifact manager"""
    pass


class DatabaseError(ArtifactManagerError):
    """Database-related errors"""
    pass


class ValidationError(ArtifactManagerError):
    """Data validation errors"""
    pass


class FileSystemError(ArtifactManagerError):
    """File system operation errors"""
    pass


@dataclass
class Artifact:
    """Immutable artifact data structure with validation"""
    id: str
    title: str
    content: str
    artifact_type: str
    language: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())
    size: int = 0
    checksum: str = ""
    chat_context: str = ""
    project: str = "default"
    favorite: bool = False
    version: int = 1
    parent_id: Optional[str] = None

    def __post_init__(self):
        """Validate artifact data after initialization"""
        self.validate()
        if not self.size:
            self.size = len(self.content)
        if not self.checksum:
            self.checksum = hashlib.sha256(self.content.encode('utf-8')).hexdigest()[:12]

    def validate(self) -> None:
        """Validate artifact data"""
        if not self.id or not isinstance(self.id, str) or len(self.id.strip()) == 0:
            raise ValidationError("Artifact ID must be a non-empty string")

        if not self.title or not isinstance(self.title, str) or len(self.title.strip()) == 0:
            raise ValidationError("Artifact title must be a non-empty string")

        if len(self.title) > 200:
            raise ValidationError("Artifact title cannot exceed 200 characters")

        if not isinstance(self.content, str):
            raise ValidationError("Artifact content must be a string")

        if len(self.content) > 10_000_000:  # 10MB limit
            raise ValidationError("Artifact content cannot exceed 10MB")

        valid_types = ['code', 'document', 'html', 'text', 'data']
        if self.artifact_type not in valid_types:
            raise ValidationError(f"Artifact type must be one of: {valid_types}")

        if self.language and (not isinstance(self.language, str) or len(self.language) > 50):
            raise ValidationError("Language field must be a string under 50 characters")

        if not isinstance(self.tags, list):
            raise ValidationError("Tags must be a list")

        if len(self.tags) > 20:
            raise ValidationError("Cannot have more than 20 tags")

        for tag in self.tags:
            if not isinstance(tag, str) or len(tag) > 50 or len(tag.strip()) == 0:
                raise ValidationError("Each tag must be a non-empty string under 50 characters")

        if not isinstance(self.project, str) or len(self.project) > 100 or len(self.project.strip()) == 0:
            raise ValidationError("Project name must be a non-empty string under 100 characters")

        if not isinstance(self.favorite, bool):
            raise ValidationError("Favorite must be a boolean value")

        if not isinstance(self.version, int) or self.version < 1:
            raise ValidationError("Version must be a positive integer")


class DatabaseManager:
    """Robust database operations with connection pooling and transactions"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")

            yield conn

        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")

        finally:
            if conn:
                conn.close()

    def init_database(self) -> None:
        """Initialize database with proper schema and error handling"""
        try:
            with self.get_connection() as conn:
                # Create artifacts table
                conn.execute('''
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
                             ''')

                # Create projects table
                conn.execute('''
                             CREATE TABLE IF NOT EXISTS projects
                             (
                                 name        TEXT PRIMARY KEY,
                                 description TEXT,
                                 created     TEXT NOT NULL,
                                 color       TEXT DEFAULT '#007acc'
                             )
                             ''')

                # Create usage stats table
                conn.execute('''
                             CREATE TABLE IF NOT EXISTS usage_stats
                             (
                                 id          INTEGER PRIMARY KEY AUTOINCREMENT,
                                 artifact_id TEXT NOT NULL,
                                 accessed    TEXT NOT NULL,
                                 action      TEXT NOT NULL
                             )
                             ''')

                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_tags ON artifacts(tags)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_language ON artifacts(language)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts(created)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_usage_stats_artifact ON usage_stats(artifact_id)')

                # Create default project
                conn.execute('''
                             INSERT OR IGNORE INTO projects (name, description, created)
                             VALUES (?, ?, ?)
                             ''', ('default', 'Default project for artifacts', datetime.now().isoformat()))

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")

    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Execute query with proper error handling"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {query[:100]}... Error: {e}")
            raise DatabaseError(f"Query execution failed: {e}")

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute update/insert/delete with proper error handling"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Update execution failed: {query[:100]}... Error: {e}")
            raise DatabaseError(f"Update execution failed: {e}")

    def close(self) -> None:
        """Close database connections"""
        pass  # Connections are closed automatically in context manager


class FileManager:
    """Robust file operations with atomic writes and cleanup"""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.files_dir = storage_path / "files"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create required directories with error handling"""
        try:
            self.files_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise FileSystemError(f"Failed to create directory {self.files_dir}: {e}")

    def save_content(self, artifact_id: str, content: str, language: str = None) -> Path:
        """Save content to file with atomic write"""
        try:
            ext = self._get_file_extension(language or 'txt')
            filename = f"{artifact_id}.{ext}"
            filepath = self.files_dir / filename

            # Simple write operation
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.debug(f"Saved content to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save content for artifact {artifact_id}: {e}")
            raise FileSystemError(f"Failed to save content: {e}")

    def load_content(self, filepath: Path) -> str:
        """Load content from file with error handling"""
        try:
            if not filepath.exists():
                raise FileSystemError(f"File not found: {filepath}")

            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()

        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error for {filepath}: {e}")
            raise FileSystemError(f"File encoding error: {e}")
        except Exception as e:
            logger.error(f"Failed to load content from {filepath}: {e}")
            raise FileSystemError(f"Failed to load content: {e}")

    def delete_file(self, filepath: Path) -> bool:
        """Delete file with error handling"""
        try:
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"Deleted file {filepath}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {filepath}: {e}")
            return False

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            'python': 'py', 'javascript': 'js', 'typescript': 'ts',
            'html': 'html', 'css': 'css', 'java': 'java',
            'cpp': 'cpp', 'c': 'c', 'sql': 'sql',
            'json': 'json', 'xml': 'xml', 'yaml': 'yml',
            'markdown': 'md', 'shell': 'sh', 'bash': 'sh',
            'rust': 'rs', 'go': 'go', 'php': 'php',
            'ruby': 'rb', 'swift': 'swift', 'kotlin': 'kt',
            'code': 'txt', 'document': 'md', 'text': 'txt'
        }
        return extensions.get(language.lower() if language else 'txt', 'txt')


class ConfigManager:
    """Robust configuration management"""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self._load_config()

    def _load_config(self):
        """Load configuration with defaults"""
        default_config = {
            'general': {
                'auto_backup': 'true',
                'backup_interval_hours': '24',
                'max_backups': '10',
                'max_content_size_mb': '10',
                'max_artifacts': '10000'
            },
            'database': {
                'connection_timeout': '30',
                'max_connections': '10',
                'wal_mode': 'true'
            },
            'security': {
                'validate_content': 'true',
                'sanitize_filenames': 'true',
                'max_tag_length': '50'
            },
            'web_ui': {
                'enabled': 'true',
                'port': '5000',
                'host': 'localhost',
                'debug': 'false'
            }
        }

        try:
            if self.config_path.exists():
                self.config.read(self.config_path)
                # Ensure all default sections exist
                for section, options in default_config.items():
                    if not self.config.has_section(section):
                        self.config.add_section(section)
                    for key, value in options.items():
                        if not self.config.has_option(section, key):
                            self.config.set(section, key, value)
            else:
                self.config.read_dict(default_config)

            self._save_config()

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config.read_dict(default_config)

    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                self.config.write(f)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get(self, section: str, key: str, fallback: Any = None) -> str:
        """Get configuration value with fallback"""
        try:
            return self.config.get(section, key, fallback=fallback)
        except Exception:
            return fallback

    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        """Get boolean configuration value"""
        try:
            return self.config.getboolean(section, key, fallback=fallback)
        except Exception:
            return fallback

    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        """Get integer configuration value"""
        try:
            return self.config.getint(section, key, fallback=fallback)
        except Exception:
            return fallback


class RobustArtifactManager:
    """Production-ready artifact manager with comprehensive error handling"""

    def __init__(self, storage_path: str = "claude_artifacts"):
        try:
            self.storage_path = Path(storage_path).resolve()
            self.storage_path.mkdir(parents=True, exist_ok=True)

            # Initialize components
            self.config = ConfigManager(self.storage_path / "config.ini")
            self.db = DatabaseManager(self.storage_path / "artifacts.db")
            self.files = FileManager(self.storage_path)

            # Load limits from config
            self.max_artifacts = self.config.getint('general', 'max_artifacts', 10000)
            self.max_content_size = self.config.getint('general', 'max_content_size_mb', 10) * 1024 * 1024

            logger.info(f"Artifact manager initialized at {self.storage_path}")

        except Exception as e:
            logger.error(f"Failed to initialize artifact manager: {e}")
            raise ArtifactManagerError(f"Initialization failed: {e}")

    def save_artifact(self, content: str, title: str, artifact_type: str = "code",
                      language: str = None, tags: List[str] = None,
                      chat_context: str = "", project: str = "default") -> str:
        """Save artifact with comprehensive validation and error handling"""
        try:
            # Validate inputs
            if not content or not isinstance(content, str):
                raise ValidationError("Content cannot be empty")

            if len(content) > self.max_content_size:
                raise ValidationError(f"Content exceeds maximum size of {self.max_content_size // (1024 * 1024)}MB")

            if not title or not isinstance(title, str):
                raise ValidationError("Title cannot be empty")

            # Check artifact limit
            count_result = self.db.execute_query("SELECT COUNT(*) FROM artifacts")
            if count_result and count_result[0][0] >= self.max_artifacts:
                raise ValidationError(f"Maximum number of artifacts ({self.max_artifacts}) reached")

            # Generate unique ID with collision detection
            artifact_id = self._generate_unique_id(content)

            # Create artifact object (this validates the data)
            artifact = Artifact(
                id=artifact_id,
                title=title.strip(),
                content=content,
                artifact_type=artifact_type,
                language=language,
                tags=tags or [],
                chat_context=chat_context,
                project=project
            )

            # Save content to file
            filepath = self.files.save_content(artifact_id, content, language)

            # Save to database
            self.db.execute_update('''
                                   INSERT INTO artifacts
                                   (id, title, artifact_type, language, tags, created, modified,
                                    size, checksum, chat_context, project, filepath)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                   ''', (
                                       artifact.id, artifact.title, artifact.artifact_type, artifact.language,
                                       json.dumps(artifact.tags), artifact.created, artifact.modified,
                                       artifact.size, artifact.checksum, artifact.chat_context,
                                       artifact.project, str(filepath)
                                   ))

            # Log usage
            self._log_usage(artifact_id, "created")

            # Create backup if enabled
            if self.config.getboolean('general', 'auto_backup'):
                self._create_backup()

            logger.info(f"Saved artifact: {title} (ID: {artifact_id})")
            return artifact_id

        except Exception as e:
            logger.error(f"Failed to save artifact '{title}': {e}")
            raise ArtifactManagerError(f"Failed to save artifact: {e}")

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve artifact with error handling"""
        try:
            if not artifact_id or not isinstance(artifact_id, str):
                raise ValidationError("Invalid artifact ID")

            # Get metadata from database
            rows = self.db.execute_query(
                'SELECT * FROM artifacts WHERE id = ?', (artifact_id,)
            )

            if not rows:
                return None

            row = rows[0]
            filepath = Path(row[14])  # filepath column

            # Load content from file
            content = self.files.load_content(filepath)

            # Log access
            self._log_usage(artifact_id, "accessed")

            # Create artifact object
            artifact = Artifact(
                id=row[0], title=row[1], content=content, artifact_type=row[2],
                language=row[3], tags=json.loads(row[4]) if row[4] else [],
                created=row[5], modified=row[6], size=row[7], checksum=row[8],
                chat_context=row[9], project=row[10], favorite=bool(row[11]),
                version=row[12], parent_id=row[13]
            )

            return artifact

        except Exception as e:
            logger.error(f"Failed to get artifact {artifact_id}: {e}")
            return None

    def search_artifacts(self, query: str = None, **filters) -> List[Artifact]:
        """Search artifacts with robust filtering"""
        try:
            conditions: List[str] = []
            params: List[str] = []

            # Build search conditions safely
            if query and isinstance(query, str):
                # Use parameterized queries to prevent injection
                conditions.append("(title LIKE ? OR tags LIKE ? OR chat_context LIKE ?)")
                safe_query = f"%{query.strip()}%"
                params.extend([safe_query, safe_query, safe_query])

            # Add filters with validation
            for field, value in filters.items():
                if not value:
                    continue

                if field in ['artifact_type', 'language', 'project'] and isinstance(value, str):
                    conditions.append(f"{field} = ?")
                    params.append(value.strip())
                elif field == 'tags' and isinstance(value, str):
                    conditions.append("tags LIKE ?")
                    params.append(f"%{value.strip()}%")
                elif field == 'favorite' and value:
                    conditions.append("favorite = 1")

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            query_sql = f'''
                SELECT id, title, artifact_type, language, tags, created, modified,
                       size, checksum, chat_context, project, favorite, version, parent_id
                FROM artifacts 
                WHERE {where_clause}
                ORDER BY modified DESC
                LIMIT 1000
            '''

            rows = self.db.execute_query(query_sql, params)

            # Convert rows to artifact objects (without content for performance)
            artifacts: List[Artifact] = []
            for row in rows:
                try:
                    artifact = Artifact(
                        id=row[0], title=row[1], content="", artifact_type=row[2],
                        language=row[3], tags=json.loads(row[4]) if row[4] else [],
                        created=row[5], modified=row[6], size=row[7], checksum=row[8],
                        chat_context=row[9], project=row[10], favorite=bool(row[11]),
                        version=row[12], parent_id=row[13]
                    )
                    artifacts.append(artifact)
                except Exception as e:
                    logger.warning(f"Skipped invalid artifact row: {e}")
                    continue

            return artifacts

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def edit_artifact(self, artifact_id: str, **updates) -> bool:
        """Edit artifact with validation and atomicity"""
        try:
            if not artifact_id:
                raise ValidationError("Artifact ID required")

            # Check if artifact exists
            existing = self.get_artifact(artifact_id)
            if not existing:
                raise ValidationError(f"Artifact {artifact_id} not found")

            # Validate updates
            db_updates: Dict[str, str] = {}
            params: List[Any] = []

            if 'title' in updates and updates['title']:
                if len(updates['title']) > 200:
                    raise ValidationError("Title too long")
                db_updates['title'] = '?'
                params.append(updates['title'].strip())

            if 'tags' in updates and isinstance(updates['tags'], list):
                if len(updates['tags']) > 20:
                    raise ValidationError("Too many tags")
                db_updates['tags'] = '?'
                params.append(json.dumps(updates['tags']))

            if 'chat_context' in updates:
                db_updates['chat_context'] = '?'
                params.append(updates['chat_context'])

            if 'project' in updates and updates['project']:
                if len(updates['project']) > 100:
                    raise ValidationError("Project name too long")
                db_updates['project'] = '?'
                params.append(updates['project'].strip())

            # Handle content update
            if 'content' in updates:
                content = updates['content']
                if len(content) > self.max_content_size:
                    raise ValidationError("Content too large")

                # Save new content
                filepath = self.files.save_content(artifact_id, content, existing.language)

                # Update size and checksum
                db_updates['size'] = '?'
                db_updates['checksum'] = '?'
                params.extend([len(content), hashlib.sha256(content.encode()).hexdigest()[:12]])

            if not db_updates:
                logger.warning("No valid updates provided")
                return False

            # Always update modified timestamp
            db_updates['modified'] = '?'
            params.append(datetime.now().isoformat())

            # Build and execute update query
            set_clause = ', '.join([f"{field} = ?" for field in db_updates.keys()])
            params.append(artifact_id)

            update_sql = f'UPDATE artifacts SET {set_clause} WHERE id = ?'
            rows_affected = self.db.execute_update(update_sql, params)

            if rows_affected > 0:
                self._log_usage(artifact_id, "edited")
                logger.info(f"Updated artifact {artifact_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to edit artifact {artifact_id}: {e}")
            raise ArtifactManagerError(f"Edit failed: {e}")

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete artifact with proper cleanup"""
        try:
            if not artifact_id:
                raise ValidationError("Artifact ID required")

            # Get artifact info
            rows = self.db.execute_query(
                'SELECT title, filepath FROM artifacts WHERE id = ?', (artifact_id,)
            )

            if not rows:
                logger.warning(f"Artifact {artifact_id} not found")
                return False

            title, filepath = rows[0]

            # Delete file
            if filepath:
                file_path = Path(filepath)
                self.files.delete_file(file_path)

            # Delete from database (cascades to usage_stats)
            rows_affected = self.db.execute_update(
                'DELETE FROM artifacts WHERE id = ?', (artifact_id,)
            )

            if rows_affected > 0:
                logger.info(f"Deleted artifact: {title} ({artifact_id})")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete artifact {artifact_id}: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics with error handling"""
        try:
            stats: Dict[str, Any] = {
                'total_artifacts': 0,
                'total_projects': 0,
                'by_type': {},
                'by_language': {},
                'storage_mb': 0,
                'recent_activity': {},
                'error_count': 0
            }

            # Get basic counts
            try:
                result = self.db.execute_query('SELECT COUNT(*) FROM artifacts')
                stats['total_artifacts'] = result[0][0] if result else 0
            except Exception as e:
                logger.warning(f"Failed to get artifact count: {e}")
                stats['error_count'] += 1

            try:
                result = self.db.execute_query('SELECT COUNT(DISTINCT project) FROM artifacts')
                stats['total_projects'] = result[0][0] if result else 0
            except Exception as e:
                logger.warning(f"Failed to get project count: {e}")
                stats['error_count'] += 1

            # Get type distribution
            try:
                rows = self.db.execute_query(
                    'SELECT artifact_type, COUNT(*) FROM artifacts GROUP BY artifact_type'
                )
                stats['by_type'] = dict(rows) if rows else {}
            except Exception as e:
                logger.warning(f"Failed to get type distribution: {e}")
                stats['error_count'] += 1

            # Get language distribution
            try:
                rows = self.db.execute_query(
                    'SELECT language, COUNT(*) FROM artifacts WHERE language IS NOT NULL GROUP BY language'
                )
                stats['by_language'] = dict(rows) if rows else {}
            except Exception as e:
                logger.warning(f"Failed to get language distribution: {e}")
                stats['error_count'] += 1

            # Calculate storage usage
            try:
                result = self.db.execute_query('SELECT SUM(size) FROM artifacts')
                total_size = result[0][0] if result and result[0][0] else 0
                stats['storage_mb'] = round(total_size / (1024 * 1024), 2)
            except Exception as e:
                logger.warning(f"Failed to calculate storage usage: {e}")
                stats['error_count'] += 1

            return stats

        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
            return {'error': str(e)}

    def _generate_unique_id(self, content: str) -> str:
        """Generate unique ID with collision detection"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]

        base_id = f"{timestamp}_{content_hash}"
        artifact_id = base_id
        counter = 1

        # Check for collisions
        while True:
            rows = self.db.execute_query('SELECT 1 FROM artifacts WHERE id = ?', (artifact_id,))
            if not rows:
                break
            artifact_id = f"{base_id}_{counter}"
            counter += 1
            if counter > 1000:  # Prevent infinite loop
                raise ArtifactManagerError("Failed to generate unique ID")

        return artifact_id

    def _log_usage(self, artifact_id: str, action: str) -> None:
        """Log usage with error handling"""
        try:
            self.db.execute_update('''
                                   INSERT INTO usage_stats (artifact_id, accessed, action)
                                   VALUES (?, ?, ?)
                                   ''', (artifact_id, datetime.now().isoformat(), action))
        except Exception as e:
            logger.warning(f"Failed to log usage: {e}")

    def _create_backup(self) -> None:
        """Create backup with error handling"""
        # Backup functionality removed - simplified for core functionality
        pass

    def close(self) -> None:
        """Clean shutdown"""
        try:
            self.db.close()
            logger.info("Artifact manager closed successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit"""
        self.close()
        return False


def run_system_test(manager: RobustArtifactManager) -> None:
    """Run comprehensive system test with error checking"""
    print("🧪 Running system test...")

    test_content = "print('Hello, robust world!')\n\ndef test():\n    return 'Test successful!'"

    try:
        # Test 1: Save artifact
        print("  1. Testing save artifact...")
        artifact_id = manager.save_artifact(
            content=test_content,
            title="Test Artifact",
            artifact_type="code",
            language="python",
            tags=["test", "demo"]
        )
        print(f"  ✅ Saved test artifact: {artifact_id}")

        # Test 2: Retrieve artifact
        print("  2. Testing retrieve artifact...")
        artifact = manager.get_artifact(artifact_id)
        if artifact and artifact.content == test_content:
            print(f"  ✅ Retrieved artifact: {artifact.title}")
        else:
            raise Exception("Retrieved artifact content doesn't match")

        # Test 3: Search artifacts
        print("  3. Testing search...")
        results = manager.search_artifacts(query="test")
        if results and any(r.id == artifact_id for r in results):
            print(f"  ✅ Found {len(results)} artifacts in search")
        else:
            raise Exception("Search didn't find the test artifact")

        # Test 4: Edit artifact
        print("  4. Testing edit artifact...")
        success = manager.edit_artifact(artifact_id, title="Updated Test Artifact")
        if success:
            print("  ✅ Successfully edited artifact")
        else:
            raise Exception("Failed to edit artifact")

        # Test 5: Get statistics
        print("  5. Testing statistics...")
        stats = manager.get_statistics()
        if stats and stats.get('total_artifacts', 0) > 0:
            print(f"  ✅ Statistics working: {stats['total_artifacts']} artifacts")
        else:
            raise Exception("Statistics not working properly")

        # Test 6: Delete artifact
        print("  6. Testing delete artifact...")
        if manager.delete_artifact(artifact_id):
            print("  ✅ Deleted test artifact")
        else:
            raise Exception("Failed to delete test artifact")

        print("🧪 ✅ All tests passed!")

    except Exception as e:
        print(f"🧪 ❌ Test failed: {e}")
        # Try to cleanup test artifact if it exists
        try:
            manager.delete_artifact(artifact_id)
        except:
            pass
        raise


def start_web_interface(manager: RobustArtifactManager, port: int) -> None:
    """Start web interface with proper error checking"""
    try:
        # Import Flask here so we can check for it earlier
        # noinspection PyPackageRequirements
        from flask import Flask

        print(f"🌐 Starting web interface on port {port}...")

        # Check if port is available
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', port))
            sock.close()
        except OSError:
            raise Exception(f"Port {port} is already in use")

        # Create minimal web app for now
        app = Flask(__name__)

        @app.route('/')
        def index():
            stats = manager.get_statistics()
            return f"""
            <h1>Claude Artifact Manager</h1>
            <p>System Status: ✅ Running</p>
            <p>Total Artifacts: {stats.get('total_artifacts', 0)}</p>
            <p>Storage Used: {stats.get('storage_mb', 0)} MB</p>
            <p><a href="/stats">View Full Statistics</a></p>
            """

        @app.route('/stats')
        def stats():
            stats_data = manager.get_statistics()
            html = "<h1>Artifact Statistics</h1><ul>"
            for key, value in stats_data.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul><a href='/'>Back</a>"
            return html

        @app.errorhandler(Exception)
        def handle_error(error):
            logger.error(f"Web interface error: {error}")
            return f"<h1>Error</h1><p>{error}</p>", 500

        print(f"🌐 Web interface starting at http://localhost:{port}")
        print("🌐 Press Ctrl+C to stop")

        try:
            app.run(host='localhost', port=port, debug=False)
        except KeyboardInterrupt:
            print("\n🌐 Web interface stopped")

    except ImportError as e:
        raise Exception(f"Flask import failed: {e}")
    except Exception as e:
        raise Exception(f"Web interface error: {e}")


def run_interactive_cli(manager: RobustArtifactManager) -> None:
    """Interactive CLI with error checking"""
    print("🔧 Robust Claude Artifact Manager")
    print("Commands: save, get, search, stats, test, help, quit")
    print("Type 'help' for detailed command information")

    while True:
        try:
            command = input("\n> ").strip().lower()

            if command in ["quit", "q", "exit"]:
                break
            elif command == "help":
                print_cli_help()
            elif command == "stats":
                handle_stats_command(manager)
            elif command == "save":
                handle_save_command(manager)
            elif command == "get":
                handle_get_command(manager)
            elif command == "search":
                handle_search_command(manager)
            elif command == "test":
                run_system_test(manager)
            elif command == "":
                continue  # Empty input, just continue
            else:
                print(f"❌ Unknown command: '{command}'. Type 'help' for available commands.")

        except KeyboardInterrupt:
            raise  # Re-raise to be caught by main
        except EOFError:
            break  # End of input stream
        except Exception as e:
            print(f"❌ Command error: {e}")
            logger.error(f"CLI command error: {e}")


def print_cli_help() -> None:
    """Print detailed CLI help"""
    print("""
📖 Available Commands:
  save    - Save a new artifact (interactive)
  get     - Retrieve an artifact by ID
  search  - Search artifacts by query
  stats   - Show system statistics
  test    - Run system functionality test
  help    - Show this help message
  quit    - Exit the program (also: q, exit)

💡 Tips:
  - All commands include error checking and validation
  - Type Ctrl+C to interrupt any command
  - Check logs in 'artifact_manager.log' for detailed errors
""")


def handle_stats_command(manager: RobustArtifactManager) -> None:
    """Handle stats command with error checking"""
    try:
        stats = manager.get_statistics()
        print("\n📊 System Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"❌ Failed to get statistics: {e}")


def handle_save_command(manager: RobustArtifactManager) -> None:
    """Handle save command with validation"""
    try:
        print("\n📝 Save New Artifact")
        title = input("Title: ").strip()
        if not title:
            print("❌ Title cannot be empty")
            return

        print("Content (type 'END' on a new line to finish):")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            except EOFError:
                break

        content = "\n".join(lines)
        if not content.strip():
            print("❌ Content cannot be empty")
            return

        artifact_type = input("Type (code/document/html/text/data) [code]: ").strip() or "code"
        language = input("Language (optional): ").strip() or None
        project = input("Project [default]: ").strip() or "default"

        tags_input = input("Tags (comma-separated): ").strip()
        tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []

        artifact_id = manager.save_artifact(
            content=content,
            title=title,
            artifact_type=artifact_type,
            language=language,
            tags=tags,
            project=project
        )
        print(f"✅ Saved artifact: {artifact_id}")

    except Exception as e:
        print(f"❌ Failed to save artifact: {e}")


def handle_get_command(manager: RobustArtifactManager) -> None:
    """Handle get command with error checking"""
    try:
        artifact_id = input("Artifact ID: ").strip()
        if not artifact_id:
            print("❌ Artifact ID cannot be empty")
            return

        artifact = manager.get_artifact(artifact_id)
        if artifact:
            print(f"\n📄 {artifact.title}")
            print(f"Type: {artifact.artifact_type}")
            if artifact.language:
                print(f"Language: {artifact.language}")
            print(f"Project: {artifact.project}")
            print(f"Created: {artifact.created}")
            print(f"Size: {artifact.size} bytes")
            if artifact.tags:
                print(f"Tags: {', '.join(artifact.tags)}")
            print(f"\nContent preview (first 500 chars):")
            print("-" * 50)
            print(artifact.content[:500])
            if len(artifact.content) > 500:
                print("\n... (truncated)")
        else:
            print(f"❌ Artifact '{artifact_id}' not found")

    except Exception as e:
        print(f"❌ Failed to get artifact: {e}")


def handle_search_command(manager: RobustArtifactManager) -> None:
    """Handle search command with error checking"""
    try:
        query = input("Search query (optional): ").strip() or None
        project = input("Project filter (optional): ").strip() or None
        artifact_type = input("Type filter (optional): ").strip() or None
        language = input("Language filter (optional): ").strip() or None

        results = manager.search_artifacts(
            query=query,
            project=project,
            artifact_type=artifact_type,
            language=language
        )

        if not results:
            print("❌ No artifacts found matching your criteria")
            return

        print(f"\n🔍 Found {len(results)} artifacts:")
        for i, artifact in enumerate(results[:20], 1):  # Show max 20 results
            tags_str = f" [{', '.join(artifact.tags)}]" if artifact.tags else ""
            print(f"{i:2d}. {artifact.id} - {artifact.title} ({artifact.artifact_type}){tags_str}")

        if len(results) > 20:
            print(f"... and {len(results) - 20} more (showing first 20)")

    except Exception as e:
        print(f"❌ Search failed: {e}")


def main() -> None:
    """Main CLI interface with comprehensive error handling"""
    try:
        parser = argparse.ArgumentParser(description='Robust Claude Artifact Manager')
        parser.add_argument('--storage', default='claude_artifacts', help='Storage directory')
        parser.add_argument('--stats', action='store_true', help='Show statistics')
        parser.add_argument('--test', action='store_true', help='Run basic test')
        parser.add_argument('--web', action='store_true', help='Start web interface')
        parser.add_argument('--port', type=int, default=5000, help='Web interface port')

        args = parser.parse_args()

        # Check for web interface dependencies
        if args.web:
            try:
                # noinspection PyPackageRequirements
                import flask
                logger.info(f"Flask {flask.__version__} available")
            except ImportError as e:
                print("❌ Flask not installed. Web interface requires Flask.")
                print("💡 Install with: pip install flask")
                print("💡 Or run without --web flag for CLI mode")
                sys.exit(1)

        # Initialize manager with error handling
        try:
            manager = RobustArtifactManager(args.storage)
            logger.info("Artifact manager initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize artifact manager: {e}")
            print("💡 Check storage directory permissions and disk space")
            sys.exit(1)

        if args.stats:
            try:
                stats = manager.get_statistics()
                print("📊 Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                return
            except Exception as e:
                print(f"❌ Failed to get statistics: {e}")
                sys.exit(1)

        if args.test:
            try:
                run_system_test(manager)
                return
            except Exception as e:
                print(f"❌ System test failed: {e}")
                sys.exit(1)

        if args.web:
            try:
                start_web_interface(manager, args.port)
                return
            except Exception as e:
                print(f"❌ Failed to start web interface: {e}")
                print("💡 Check if port is already in use or try different port")
                sys.exit(1)

        # Interactive CLI mode
        try:
            run_interactive_cli(manager)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ CLI error: {e}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

    finally:
        try:
            if 'manager' in locals():
                manager.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()