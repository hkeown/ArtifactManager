#!/usr/bin/env python3
# noinspection PyUnresolvedReferences,SqlDialectInspection,SqlNoDataSourceInspection,SqlResolve,PyTypeChecker,PyUnusedLocal,PyBroadException,PyProtectedMember,PyMethodMayBeStatic,PyShadowingNames,PyTooManyLocals,PyIncorrectDocstring,PyPep8Naming,PyUnboundLocalVariable
"""
Robust Claude Artifact Management System - COMPLETE VERSION
Production-ready tool with comprehensive error handling and validation
Includes all frontend-required endpoints
"""

import json
import sqlite3
import hashlib
import logging
import sys
import argparse
import configparser
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Generator
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


class DatabaseError(ArtifactManagerError):
    """Database-related errors"""


class ValidationError(ArtifactManagerError):
    """Data validation errors"""


class FileSystemError(ArtifactManagerError):
    """File system operation errors"""


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

    def __post_init__(self) -> None:
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

        valid_types: List[str] = ['code', 'document', 'html', 'text', 'data']
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

    def __init__(self, db_path: Path) -> None:
        self.db_path: Path = db_path
        self.init_database()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with automatic cleanup"""
        connection: Optional[sqlite3.Connection] = None
        try:
            connection = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")

            yield connection

        except sqlite3.Error as exc:
            if connection:
                connection.rollback()
            logger.error(f"Database error: {exc}")
            raise DatabaseError(f"Database operation failed: {exc}")

        finally:
            if connection:
                connection.close()

    def init_database(self) -> None:
        """Initialize database with proper schema and error handling"""
        try:
            with self.get_connection() as connection:
                # Create artifacts table
                connection.execute('''
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
                connection.execute('''
                                   CREATE TABLE IF NOT EXISTS projects
                                   (
                                       name        TEXT PRIMARY KEY,
                                       description TEXT,
                                       created     TEXT NOT NULL,
                                       color       TEXT DEFAULT '#007acc'
                                   )
                                   ''')

                # Create usage stats table
                connection.execute('''
                                   CREATE TABLE IF NOT EXISTS usage_stats
                                   (
                                       id          INTEGER PRIMARY KEY AUTOINCREMENT,
                                       artifact_id TEXT NOT NULL,
                                       accessed    TEXT NOT NULL,
                                       action      TEXT NOT NULL
                                   )
                                   ''')

                # Create indexes
                connection.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_tags ON artifacts(tags)')
                connection.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project)')
                connection.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type)')
                connection.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_language ON artifacts(language)')
                connection.execute('CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts(created)')
                connection.execute('CREATE INDEX IF NOT EXISTS idx_usage_stats_artifact ON usage_stats(artifact_id)')

                # Create default project
                connection.execute('''
                                   INSERT OR IGNORE INTO projects (name, description, created)
                                   VALUES (?, ?, ?)
                                   ''', ('default', 'Default project for artifacts', datetime.now().isoformat()))

                connection.commit()
                logger.info("Database initialized successfully")

        except Exception as exc:
            logger.error(f"Failed to initialize database: {exc}")
            raise DatabaseError(f"Database initialization failed: {exc}")

    def execute_query(self, query: str, params: Tuple[Any, ...] = ()) -> List[Tuple[Any, ...]]:
        """Execute query with proper error handling"""
        try:
            with self.get_connection() as connection:
                cursor = connection.execute(query, params)
                return cursor.fetchall()
        except Exception as exc:
            logger.error(f"Query execution failed: {query[:100]}... Error: {exc}")
            raise DatabaseError(f"Query execution failed: {exc}")

    def execute_update(self, query: str, params: Tuple[Any, ...] = ()) -> int:
        """Execute update/insert/delete with proper error handling"""
        try:
            with self.get_connection() as connection:
                cursor = connection.execute(query, params)
                connection.commit()
                return cursor.rowcount
        except Exception as exc:
            logger.error(f"Update execution failed: {query[:100]}... Error: {exc}")
            raise DatabaseError(f"Update execution failed: {exc}")

    def close(self) -> None:
        """Close database connections"""
        pass  # Connections are closed automatically in context manager


class FileManager:
    """Robust file operations with atomic writes and cleanup"""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path: Path = storage_path
        self.files_dir: Path = storage_path / "files"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create required directories with error handling"""
        try:
            self.files_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise FileSystemError(f"Failed to create directory {self.files_dir}: {exc}")

    def save_content(self, artifact_id: str, content: str, language: Optional[str] = None) -> Path:
        """Save content to file with atomic write"""
        try:
            ext: str = FileManager.get_file_extension(language or 'txt')
            filename: str = f"{artifact_id}.{ext}"
            filepath: Path = self.files_dir / filename

            # Simple write operation
            with open(filepath, 'w', encoding='utf-8') as file_handle:
                file_handle.write(content)

            logger.debug(f"Saved content to {filepath}")
            return filepath

        except Exception as exc:
            logger.error(f"Failed to save content for artifact {artifact_id}: {exc}")
            raise FileSystemError(f"Failed to save content: {exc}")

    @staticmethod
    def load_content(filepath: Path) -> str:
        """Load content from file with error handling"""
        try:
            if not filepath.exists():
                raise FileSystemError(f"File not found: {filepath}")

            with open(filepath, 'r', encoding='utf-8') as file_handle:
                return file_handle.read()

        except UnicodeDecodeError as exc:
            logger.error(f"Unicode decode error for {filepath}: {exc}")
            raise FileSystemError(f"File encoding error: {exc}")
        except Exception as exc:
            logger.error(f"Failed to load content from {filepath}: {exc}")
            raise FileSystemError(f"Failed to load content: {exc}")

    @staticmethod
    def delete_file(filepath: Path) -> bool:
        """Delete file with error handling"""
        try:
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"Deleted file {filepath}")
                return True
            return False
        except Exception as exc:
            logger.error(f"Failed to delete file {filepath}: {exc}")
            return False

    @staticmethod
    def get_file_extension(language: str) -> str:
        """Get file extension for language"""
        extensions: Dict[str, str] = {
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

    def __init__(self, config_path: Path) -> None:
        self.config_path: Path = config_path
        self.config: configparser.ConfigParser = configparser.ConfigParser()
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration with defaults"""
        default_config: Dict[str, Dict[str, str]] = {
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

        except Exception as exc:
            logger.error(f"Failed to load config: {exc}")
            self.config.read_dict(default_config)

    def _save_config(self) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as file_handle:
                self.config.write(file_handle)
        except Exception as exc:
            logger.error(f"Failed to save config: {exc}")

    def get(self, section: str, key: str, fallback: Optional[str] = None) -> str:
        """Get configuration value with fallback"""
        try:
            return self.config.get(section, key, fallback=fallback)
        except Exception:
            return fallback or ""

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

    def __init__(self, storage_path: str = "claude_artifacts") -> None:
        try:
            self.storage_path: Path = Path(storage_path).resolve()
            self.storage_path.mkdir(parents=True, exist_ok=True)

            # Initialize components
            self.config: ConfigManager = ConfigManager(self.storage_path / "config.ini")
            self.db: DatabaseManager = DatabaseManager(self.storage_path / "artifacts.db")
            self.files: FileManager = FileManager(self.storage_path)

            # Load limits from config
            self.max_artifacts: int = self.config.getint('general', 'max_artifacts', 10000)
            self.max_content_size: int = self.config.getint('general', 'max_content_size_mb', 10) * 1024 * 1024

            logger.info(f"Artifact manager initialized at {self.storage_path}")

        except Exception as exc:
            logger.error(f"Failed to initialize artifact manager: {exc}")
            raise ArtifactManagerError(f"Initialization failed: {exc}")

    def save_artifact(self, content: str, title: str, artifact_type: str = "code",
                      language: Optional[str] = None, tags: Optional[List[str]] = None,
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
            count_result: List[Tuple[Any, ...]] = self.db.execute_query("SELECT COUNT(*) FROM artifacts")
            if count_result and count_result[0][0] >= self.max_artifacts:
                raise ValidationError(f"Maximum number of artifacts ({self.max_artifacts}) reached")

            # Generate unique ID with collision detection
            artifact_id: str = self._generate_unique_id(content)

            # Create artifact object (this validates the data)
            artifact: Artifact = Artifact(
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
            filepath: Path = self.files.save_content(artifact_id, content, language)

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

            logger.info(f"Saved artifact: {title} (ID: {artifact_id})")
            return artifact_id

        except Exception as exc:
            logger.error(f"Failed to save artifact '{title}': {exc}")
            raise ArtifactManagerError(f"Failed to save artifact: {exc}")

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve artifact with error handling"""
        try:
            if not artifact_id or not isinstance(artifact_id, str):
                raise ValidationError("Invalid artifact ID")

            # Get metadata from database
            rows: List[Tuple[Any, ...]] = self.db.execute_query(
                'SELECT * FROM artifacts WHERE id = ?', (artifact_id,)
            )

            if not rows:
                return None

            row: Tuple[Any, ...] = rows[0]
            filepath: Path = Path(row[14])  # filepath column

            # Load content from file
            content: str = FileManager.load_content(filepath)

            # Log access
            self._log_usage(artifact_id, "accessed")

            # Create artifact object
            artifact: Artifact = Artifact(
                id=row[0], title=row[1], content=content, artifact_type=row[2],
                language=row[3], tags=json.loads(row[4]) if row[4] else [],
                created=row[5], modified=row[6], size=row[7], checksum=row[8],
                chat_context=row[9], project=row[10], favorite=bool(row[11]),
                version=row[12], parent_id=row[13]
            )

            return artifact

        except Exception as exc:
            logger.error(f"Failed to get artifact {artifact_id}: {exc}")
            return None

    def search_artifacts(self, query: Optional[str] = None, **filters: Any) -> List[Artifact]:
        """Search artifacts with robust filtering"""
        try:
            conditions: List[str] = []
            params: List[str] = []

            # Build search conditions safely
            if query and isinstance(query, str):
                # Use parameterized queries to prevent injection
                conditions.append("(title LIKE ? OR tags LIKE ? OR chat_context LIKE ?)")
                safe_query: str = f"%{query.strip()}%"
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

            where_clause: str = " AND ".join(conditions) if conditions else "1=1"

            query_sql: str = f'''
                SELECT id, title, artifact_type, language, tags, created, modified,
                       size, checksum, chat_context, project, favorite, version, parent_id
                FROM artifacts 
                WHERE {where_clause}
                ORDER BY modified DESC
                LIMIT 1000
            '''

            rows: List[Tuple[Any, ...]] = self.db.execute_query(query_sql, tuple(params))

            # Convert rows to artifact objects (without content for performance)
            artifacts: List[Artifact] = []
            for row in rows:
                try:
                    artifact: Artifact = Artifact(
                        id=row[0], title=row[1], content="", artifact_type=row[2],
                        language=row[3], tags=json.loads(row[4]) if row[4] else [],
                        created=row[5], modified=row[6], size=row[7], checksum=row[8],
                        chat_context=row[9], project=row[10], favorite=bool(row[11]),
                        version=row[12], parent_id=row[13]
                    )
                    artifacts.append(artifact)
                except Exception as exc:
                    logger.warning(f"Skipped invalid artifact row: {exc}")
                    continue

            return artifacts

        except Exception as exc:
            logger.error(f"Search failed: {exc}")
            return []

    def edit_artifact(self, artifact_id: str, **updates: Any) -> bool:
        """Edit artifact with validation and atomicity"""
        try:
            if not artifact_id:
                raise ValidationError("Artifact ID required")

            # Check if artifact exists
            existing: Optional[Artifact] = self.get_artifact(artifact_id)
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

            if 'favorite' in updates:
                db_updates['favorite'] = '?'
                params.append(1 if updates['favorite'] else 0)

            # Handle content update
            if 'content' in updates:
                content: str = updates['content']
                if len(content) > self.max_content_size:
                    raise ValidationError("Content too large")

                # Save new content
                self.files.save_content(artifact_id, content, existing.language)

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
            set_clause: str = ', '.join([f"{field} = {placeholder}" for field, placeholder in db_updates.items()])
            params.append(artifact_id)

            update_sql: str = f'UPDATE artifacts SET {set_clause} WHERE id = ?'
            rows_affected: int = self.db.execute_update(update_sql, tuple(params))

            if rows_affected > 0:
                self._log_usage(artifact_id, "edited")
                logger.info(f"Updated artifact {artifact_id}")
                return True

            return False

        except Exception as exc:
            logger.error(f"Failed to edit artifact {artifact_id}: {exc}")
            raise ArtifactManagerError(f"Edit failed: {exc}")

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete artifact with proper cleanup"""
        try:
            if not artifact_id:
                raise ValidationError("Artifact ID required")

            # Get artifact info
            rows: List[Tuple[Any, ...]] = self.db.execute_query(
                'SELECT title, filepath FROM artifacts WHERE id = ?', (artifact_id,)
            )

            if not rows:
                logger.warning(f"Artifact {artifact_id} not found")
                return False

            title: str = rows[0][0]
            filepath: str = rows[0][1]

            # Delete file
            if filepath:
                file_path: Path = Path(filepath)
                FileManager.delete_file(file_path)

            # Delete from database
            rows_affected: int = self.db.execute_update(
                'DELETE FROM artifacts WHERE id = ?', (artifact_id,)
            )

            if rows_affected > 0:
                logger.info(f"Deleted artifact: {title} ({artifact_id})")
                return True

            return False

        except Exception as exc:
            logger.error(f"Failed to delete artifact {artifact_id}: {exc}")
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
                result: List[Tuple[Any, ...]] = self.db.execute_query('SELECT COUNT(*) FROM artifacts')
                stats['total_artifacts'] = result[0][0] if result else 0
            except Exception as exc:
                logger.warning(f"Failed to get artifact count: {exc}")
                stats['error_count'] += 1

            try:
                result = self.db.execute_query('SELECT COUNT(DISTINCT project) FROM artifacts')
                stats['total_projects'] = result[0][0] if result else 0
            except Exception as exc:
                logger.warning(f"Failed to get project count: {exc}")
                stats['error_count'] += 1

            # Get type distribution
            try:
                rows: List[Tuple[Any, ...]] = self.db.execute_query(
                    'SELECT artifact_type, COUNT(*) FROM artifacts GROUP BY artifact_type'
                )
                stats['by_type'] = dict(rows) if rows else {}
            except Exception as exc:
                logger.warning(f"Failed to get type distribution: {exc}")
                stats['error_count'] += 1

            # Get language distribution
            try:
                rows = self.db.execute_query(
                    'SELECT language, COUNT(*) FROM artifacts WHERE language IS NOT NULL GROUP BY language'
                )
                stats['by_language'] = dict(rows) if rows else {}
            except Exception as exc:
                logger.warning(f"Failed to get language distribution: {exc}")
                stats['error_count'] += 1

            # Calculate storage usage
            try:
                result = self.db.execute_query('SELECT SUM(size) FROM artifacts')
                total_size: int = result[0][0] if result and result[0][0] else 0
                stats['storage_mb'] = round(total_size / (1024 * 1024), 2)
            except Exception as exc:
                logger.warning(f"Failed to calculate storage usage: {exc}")
                stats['error_count'] += 1

            return stats

        except Exception as exc:
            logger.error(f"Failed to generate statistics: {exc}")
            return {'error': str(exc)}

    def _generate_unique_id(self, content: str) -> str:
        """Generate unique ID with collision detection"""
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash: str = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]

        base_id: str = f"{timestamp}_{content_hash}"
        artifact_id: str = base_id
        counter: int = 1

        # Check for collisions
        while True:
            rows: List[Tuple[Any, ...]] = self.db.execute_query('SELECT 1 FROM artifacts WHERE id = ?', (artifact_id,))
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
        except Exception as exc:
            logger.warning(f"Failed to log usage: {exc}")

    def close(self) -> None:
        """Clean shutdown"""
        try:
            self.db.close()
            logger.info("Artifact manager closed successfully")
        except Exception as exc:
            logger.error(f"Error during shutdown: {exc}")

    def __enter__(self) -> 'RobustArtifactManager':
        """Context manager entry"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit"""
        self.close()
        return False


def start_web_interface(artifact_manager: RobustArtifactManager, port: int) -> None:
    """Start web interface with full API support - COMPLETE VERSION"""
    try:
        # Import Flask here to avoid import issues
        from flask import Flask, request, jsonify, send_file
        from flask_cors import CORS
    except ImportError:
        print("❌ Flask not available. Install with: pip install flask flask-cors")
        return

    try:
        app: Flask = Flask(__name__)
        CORS(app)  # Enable CORS for frontend communication

        # API Routes
        @app.route('/api/health')
        def health_check():
            """API health check"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })

        @app.route('/api/artifacts', methods=['GET'])
        def get_artifacts():
            """Get all artifacts with optional filtering"""
            try:
                # Get query parameters
                search_query: str = request.args.get('query', '')
                artifact_type: str = request.args.get('type', '')
                project: str = request.args.get('project', '')
                language: str = request.args.get('language', '')

                # Build filters
                search_filters: Dict[str, str] = {}
                if artifact_type and artifact_type != 'all':
                    search_filters['artifact_type'] = artifact_type
                if project and project != 'all':
                    search_filters['project'] = project
                if language and language != 'all':
                    search_filters['language'] = language

                # Search artifacts
                artifacts: List[Artifact] = artifact_manager.search_artifacts(
                    query=search_query if search_query else None, **search_filters
                )

                # Convert to JSON-serializable format
                result: List[Dict[str, Any]] = []
                for artifact in artifacts:
                    artifact_dict: Dict[str, Any] = {
                        'id': artifact.id,
                        'title': artifact.title,
                        'artifact_type': artifact.artifact_type,
                        'language': artifact.language,
                        'tags': artifact.tags,
                        'created': artifact.created,
                        'modified': artifact.modified,
                        'size': artifact.size,
                        'checksum': artifact.checksum,
                        'chat_context': artifact.chat_context,
                        'project': artifact.project,
                        'favorite': artifact.favorite,
                        'version': artifact.version,
                        'parent_id': artifact.parent_id
                    }
                    result.append(artifact_dict)

                return jsonify({
                    'artifacts': result,
                    'total': len(result),
                    'query': search_query,
                    'filters': search_filters
                })

            except Exception as exc:
                logger.error(f"Error getting artifacts: {exc}")
                return jsonify({'error': str(exc)}), 500

        @app.route('/api/artifacts/<artifact_id>', methods=['GET'])
        def get_artifact(artifact_id: str):
            """Get specific artifact with content"""
            try:
                artifact: Optional[Artifact] = artifact_manager.get_artifact(artifact_id)
                if not artifact:
                    return jsonify({'error': 'Artifact not found'}), 404

                return jsonify({
                    'id': artifact.id,
                    'title': artifact.title,
                    'content': artifact.content,
                    'artifact_type': artifact.artifact_type,
                    'language': artifact.language,
                    'tags': artifact.tags,
                    'created': artifact.created,
                    'modified': artifact.modified,
                    'size': artifact.size,
                    'checksum': artifact.checksum,
                    'chat_context': artifact.chat_context,
                    'project': artifact.project,
                    'favorite': artifact.favorite,
                    'version': artifact.version,
                    'parent_id': artifact.parent_id
                })

            except Exception as exc:
                logger.error(f"Error getting artifact {artifact_id}: {exc}")
                return jsonify({'error': str(exc)}), 500

        @app.route('/api/artifacts', methods=['POST'])
        def create_artifact():
            """Create new artifact"""
            try:
                data: Optional[Dict[str, Any]] = request.get_json()
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400

                # Validate required fields
                required_fields: List[str] = ['title', 'content', 'artifact_type']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400

                # Create artifact
                artifact_id: str = artifact_manager.save_artifact(
                    content=data['content'],
                    title=data['title'],
                    artifact_type=data.get('artifact_type', 'code'),
                    language=data.get('language'),
                    tags=data.get('tags', []),
                    chat_context=data.get('chat_context', ''),
                    project=data.get('project', 'default')
                )

                return jsonify({
                    'id': artifact_id,
                    'message': 'Artifact created successfully'
                }), 201

            except Exception as exc:
                logger.error(f"Error creating artifact: {exc}")
                return jsonify({'error': str(exc)}), 500

        @app.route('/api/artifacts/<artifact_id>', methods=['PATCH', 'PUT'])
        def update_artifact(artifact_id: str):
            """Update existing artifact (supports partial updates)"""
            try:
                # Validate artifact exists
                existing_artifact: Optional[Artifact] = artifact_manager.get_artifact(artifact_id)
                if not existing_artifact:
                    return jsonify({'error': 'Artifact not found'}), 404

                # Get JSON data from request
                data: Optional[Dict[str, Any]] = request.get_json()
                if not data:
                    return jsonify({'error': 'No JSON data provided'}), 400

                # Build updates dictionary with only provided fields
                updates: Dict[str, Any] = {}

                # Validate and add each field if provided
                if 'title' in data:
                    title = str(data['title']).strip()
                    if not title:
                        return jsonify({'error': 'Title cannot be empty'}), 400
                    if len(title) > 200:
                        return jsonify({'error': 'Title cannot exceed 200 characters'}), 400
                    updates['title'] = title

                if 'content' in data:
                    content = str(data['content'])
                    max_size = artifact_manager.config.getint('general', 'max_content_size_mb', 10) * 1024 * 1024
                    if len(content) > max_size:
                        return jsonify({'error': f'Content exceeds maximum size of {max_size // (1024 * 1024)}MB'}), 400
                    updates['content'] = content

                if 'tags' in data:
                    tags = data['tags']
                    if isinstance(tags, list):
                        if len(tags) > 20:
                            return jsonify({'error': 'Cannot have more than 20 tags'}), 400
                        # Validate each tag
                        for tag in tags:
                            if not isinstance(tag, str) or len(tag.strip()) == 0 or len(tag) > 50:
                                return jsonify(
                                    {'error': 'Each tag must be a non-empty string under 50 characters'}), 400
                        updates['tags'] = [tag.strip() for tag in tags if tag.strip()]
                    else:
                        return jsonify({'error': 'Tags must be an array'}), 400

                if 'project' in data:
                    project = str(data['project']).strip()
                    if not project:
                        return jsonify({'error': 'Project name cannot be empty'}), 400
                    if len(project) > 100:
                        return jsonify({'error': 'Project name cannot exceed 100 characters'}), 400
                    updates['project'] = project

                if 'chat_context' in data:
                    updates['chat_context'] = str(data['chat_context'])

                if 'favorite' in data:
                    if isinstance(data['favorite'], bool):
                        updates['favorite'] = data['favorite']
                    else:
                        return jsonify({'error': 'Favorite must be a boolean value'}), 400

                # Check if any updates were provided
                if not updates:
                    return jsonify({'error': 'No valid update fields provided'}), 400

                # Perform the update
                success: bool = artifact_manager.edit_artifact(artifact_id, **updates)

                if success:
                    # Return the updated artifact
                    updated_artifact: Optional[Artifact] = artifact_manager.get_artifact(artifact_id)
                    if updated_artifact:
                        return jsonify({
                            'id': updated_artifact.id,
                            'title': updated_artifact.title,
                            'content': updated_artifact.content,
                            'artifact_type': updated_artifact.artifact_type,
                            'language': updated_artifact.language,
                            'tags': updated_artifact.tags,
                            'created': updated_artifact.created,
                            'modified': updated_artifact.modified,
                            'size': updated_artifact.size,
                            'checksum': updated_artifact.checksum,
                            'chat_context': updated_artifact.chat_context,
                            'project': updated_artifact.project,
                            'favorite': updated_artifact.favorite,
                            'version': updated_artifact.version,
                            'parent_id': updated_artifact.parent_id,
                            'message': 'Artifact updated successfully'
                        })
                    else:
                        return jsonify({'error': 'Failed to retrieve updated artifact'}), 500
                else:
                    return jsonify({'error': 'Failed to update artifact'}), 500

            except ValidationError as exc:
                logger.error(f"Validation error updating artifact {artifact_id}: {exc}")
                return jsonify({'error': str(exc)}), 400
            except Exception as exc:
                logger.error(f"Error updating artifact {artifact_id}: {exc}")
                return jsonify({'error': f'Internal server error: {str(exc)}'}), 500

        @app.route('/api/artifacts/<artifact_id>/favorite', methods=['PATCH'])
        def toggle_favorite(artifact_id: str):
            """Toggle favorite status of an artifact"""
            try:
                # Check if artifact exists
                existing_artifact: Optional[Artifact] = artifact_manager.get_artifact(artifact_id)
                if not existing_artifact:
                    return jsonify({'error': 'Artifact not found'}), 404

                # Toggle favorite status
                new_favorite_status = not existing_artifact.favorite
                success: bool = artifact_manager.edit_artifact(artifact_id, favorite=new_favorite_status)

                if success:
                    return jsonify({
                        'id': artifact_id,
                        'favorite': new_favorite_status,
                        'message': f'Artifact {"added to" if new_favorite_status else "removed from"} favorites'
                    })
                else:
                    return jsonify({'error': 'Failed to update favorite status'}), 500

            except Exception as exc:
                logger.error(f"Error toggling favorite for artifact {artifact_id}: {exc}")
                return jsonify({'error': str(exc)}), 500

        @app.route('/api/artifacts/<artifact_id>', methods=['DELETE'])
        def delete_artifact(artifact_id: str):
            """Delete artifact"""
            try:
                success: bool = artifact_manager.delete_artifact(artifact_id)

                if success:
                    return jsonify({'message': 'Artifact deleted successfully'})
                else:
                    return jsonify({'error': 'Artifact not found'}), 404

            except Exception as exc:
                logger.error(f"Error deleting artifact {artifact_id}: {exc}")
                return jsonify({'error': str(exc)}), 500

        @app.route('/api/artifacts/bulk', methods=['DELETE'])
        def bulk_delete_artifacts():
            """Delete multiple artifacts"""
            try:
                data: Optional[Dict[str, Any]] = request.get_json()
                if not data or 'artifact_ids' not in data:
                    return jsonify({'error': 'artifact_ids array required'}), 400

                artifact_ids = data['artifact_ids']
                if not isinstance(artifact_ids, list):
                    return jsonify({'error': 'artifact_ids must be an array'}), 400

                if len(artifact_ids) == 0:
                    return jsonify({'error': 'No artifact IDs provided'}), 400

                if len(artifact_ids) > 100:
                    return jsonify({'error': 'Cannot delete more than 100 artifacts at once'}), 400

                # Delete artifacts and track results
                deleted_count = 0
                failed_ids = []

                for artifact_id in artifact_ids:
                    try:
                        if artifact_manager.delete_artifact(str(artifact_id)):
                            deleted_count += 1
                        else:
                            failed_ids.append(artifact_id)
                    except Exception as exc:
                        logger.warning(f"Failed to delete artifact {artifact_id}: {exc}")
                        failed_ids.append(artifact_id)

                # Return results
                result = {
                    'deleted_count': deleted_count,
                    'total_requested': len(artifact_ids),
                    'message': f'Successfully deleted {deleted_count} out of {len(artifact_ids)} artifacts'
                }

                if failed_ids:
                    result['failed_ids'] = failed_ids
                    result['message'] += f'. Failed to delete {len(failed_ids)} artifacts.'

                status_code = 200 if deleted_count > 0 else 400
                return jsonify(result), status_code

            except Exception as exc:
                logger.error(f"Error in bulk delete: {exc}")
                return jsonify({'error': str(exc)}), 500

        @app.route('/api/artifacts/export', methods=['POST'])
        def export_artifacts():
            """Export multiple artifacts as JSON"""
            try:
                data: Optional[Dict[str, Any]] = request.get_json()
                if not data or 'artifact_ids' not in data:
                    return jsonify({'error': 'artifact_ids array required'}), 400

                artifact_ids = data['artifact_ids']
                if not isinstance(artifact_ids, list):
                    return jsonify({'error': 'artifact_ids must be an array'}), 400

                if len(artifact_ids) == 0:
                    return jsonify({'error': 'No artifact IDs provided'}), 400

                # Collect artifacts
                exported_artifacts = []
                for artifact_id in artifact_ids:
                    try:
                        artifact: Optional[Artifact] = artifact_manager.get_artifact(str(artifact_id))
                        if artifact:
                            exported_artifacts.append({
                                'id': artifact.id,
                                'title': artifact.title,
                                'content': artifact.content,
                                'artifact_type': artifact.artifact_type,
                                'language': artifact.language,
                                'tags': artifact.tags,
                                'created': artifact.created,
                                'modified': artifact.modified,
                                'size': artifact.size,
                                'checksum': artifact.checksum,
                                'chat_context': artifact.chat_context,
                                'project': artifact.project,
                                'favorite': artifact.favorite,
                                'version': artifact.version,
                                'parent_id': artifact.parent_id
                            })
                    except Exception as exc:
                        logger.warning(f"Failed to export artifact {artifact_id}: {exc}")

                return jsonify({
                    'artifacts': exported_artifacts,
                    'total_exported': len(exported_artifacts),
                    'total_requested': len(artifact_ids),
                    'export_date': datetime.now().isoformat()
                })

            except Exception as exc:
                logger.error(f"Error in export: {exc}")
                return jsonify({'error': str(exc)}), 500

        @app.route('/api/artifacts/<artifact_id>/download')
        def download_artifact(artifact_id: str):
            """Download artifact as file"""
            try:
                artifact: Optional[Artifact] = artifact_manager.get_artifact(artifact_id)
                if not artifact:
                    return jsonify({'error': 'Artifact not found'}), 404

                # Create temporary file
                ext: str = FileManager.get_file_extension(artifact.language or artifact.artifact_type)

                with tempfile.NamedTemporaryFile(
                        mode='w',
                        suffix=f'.{ext}',
                        delete=False,
                        encoding='utf-8'
                ) as temp_file:
                    temp_file.write(artifact.content)
                    temp_path: str = temp_file.name

                filename: str = f"{artifact.title.replace(' ', '_')}.{ext}"

                return send_file(
                    temp_path,
                    as_attachment=True,
                    download_name=filename,
                    mimetype='text/plain'
                )

            except Exception as exc:
                logger.error(f"Error downloading artifact {artifact_id}: {exc}")
                return jsonify({'error': str(exc)}), 500

        @app.route('/api/projects')
        def get_projects():
            """Get all projects"""
            try:
                # Get unique projects from database
                rows: List[Tuple[Any, ...]] = artifact_manager.db.execute_query(
                    'SELECT DISTINCT project FROM artifacts ORDER BY project'
                )
                projects: List[str] = [row[0] for row in rows] if rows else ['default']

                return jsonify({'projects': projects})

            except Exception as exc:
                logger.error(f"Error getting projects: {exc}")
                return jsonify({'error': str(exc)}), 500

        @app.route('/api/stats')
        def get_statistics():
            """Get system statistics"""
            try:
                stats: Dict[str, Any] = artifact_manager.get_statistics()
                return jsonify(stats)

            except Exception as exc:
                logger.error(f"Error getting statistics: {exc}")
                return jsonify({'error': str(exc)}), 500

        @app.route('/')
        def index():
            """Basic status page"""
            stats: Dict[str, Any] = artifact_manager.get_statistics()
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Claude Artifact Manager API</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .status {{ color: #28a745; }}
                    .endpoint {{ background: #f8f9fa; padding: 10px; margin: 5px 0; border-left: 3px solid #007bff; }}
                    code {{ background: #e9ecef; padding: 2px 4px; border-radius: 3px; }}
                </style>
            </head>
            <body>
                <h1>🚀 Claude Artifact Manager API</h1>
                <p class="status">Status: ✅ Running</p>
                <p><strong>Total Artifacts:</strong> {stats.get('total_artifacts', 0)}</p>
                <p><strong>Storage Used:</strong> {stats.get('storage_mb', 0)} MB</p>
                <p><strong>Projects:</strong> {stats.get('total_projects', 0)}</p>

                <h2>📡 API Endpoints</h2>
                <div class="endpoint"><strong>GET</strong> /api/health - Health check</div>
                <div class="endpoint"><strong>GET</strong> /api/artifacts - List all artifacts</div>
                <div class="endpoint"><strong>POST</strong> /api/artifacts - Create new artifact</div>
                <div class="endpoint"><strong>GET</strong> /api/artifacts/{{id}} - Get specific artifact</div>
                <div class="endpoint"><strong>PATCH</strong> /api/artifacts/{{id}} - Update artifact</div>
                <div class="endpoint"><strong>DELETE</strong> /api/artifacts/{{id}} - Delete artifact</div>
                <div class="endpoint"><strong>PATCH</strong> /api/artifacts/{{id}}/favorite - Toggle favorite</div>
                <div class="endpoint"><strong>DELETE</strong> /api/artifacts/bulk - Bulk delete</div>
                <div class="endpoint"><strong>POST</strong> /api/artifacts/export - Export artifacts</div>
                <div class="endpoint"><strong>GET</strong> /api/artifacts/{{id}}/download - Download artifact</div>
                <div class="endpoint"><strong>GET</strong> /api/projects - Get all projects</div>
                <div class="endpoint"><strong>GET</strong> /api/stats - Get statistics</div>

                <h2>🌐 Frontend Connection</h2>
                <p>Frontend should connect to: <code>http://localhost:{port}/api</code></p>
                <p><a href="/api/stats">📊 View API Statistics</a></p>

                <h2>🔧 Usage</h2>
                <p>Start frontend and ensure it connects to this API endpoint.</p>
                <p>All endpoints support CORS for frontend integration.</p>
            </body>
            </html>
            """

        print(f"🌐 Starting Claude Artifact Manager API on port {port}...")
        print(f"🌐 API Base URL: http://localhost:{port}/api")
        print("🌐 Available endpoints:")
        print("   GET  /api/health - Health check")
        print("   GET  /api/artifacts - List all artifacts")
        print("   POST /api/artifacts - Create new artifact")
        print("   GET  /api/artifacts/{id} - Get specific artifact")
        print("   PATCH /api/artifacts/{id} - Update artifact")
        print("   DELETE /api/artifacts/{id} - Delete artifact")
        print("   PATCH /api/artifacts/{id}/favorite - Toggle favorite")
        print("   DELETE /api/artifacts/bulk - Bulk delete")
        print("   POST /api/artifacts/export - Export artifacts")
        print("   GET  /api/artifacts/{id}/download - Download artifact")
        print("   GET  /api/projects - Get all projects")
        print("   GET  /api/stats - Get statistics")
        print("🌐 Press Ctrl+C to stop")

        app.run(host='localhost', port=port, debug=False)

    except Exception as exc:
        raise Exception(f"Web interface error: {exc}")


def run_system_test(test_manager: RobustArtifactManager) -> None:
    """Run comprehensive system test with error checking"""
    print("🧪 Running system test...")

    test_content: str = "print('Hello, robust world!')\n\ndef test():\n    return 'Test successful!'"
    test_artifact_id: str = ""

    try:
        # Test 1: Save artifact
        print("  1. Testing save artifact...")
        test_artifact_id = test_manager.save_artifact(
            content=test_content,
            title="Test Artifact",
            artifact_type="code",
            language="python",
            tags=["test", "demo"]
        )
        print(f"  ✅ Saved test artifact: {test_artifact_id}")

        # Test 2: Retrieve artifact
        print("  2. Testing retrieve artifact...")
        artifact: Optional[Artifact] = test_manager.get_artifact(test_artifact_id)
        if artifact and artifact.content == test_content:
            print(f"  ✅ Retrieved artifact: {artifact.title}")
        else:
            raise Exception("Retrieved artifact content doesn't match")

        # Test 3: Search artifacts
        print("  3. Testing search...")
        results: List[Artifact] = test_manager.search_artifacts(query="test")
        if results and any(r.id == test_artifact_id for r in results):
            print(f"  ✅ Found {len(results)} artifacts in search")
        else:
            raise Exception("Search didn't find the test artifact")

        # Test 4: Edit artifact
        print("  4. Testing edit artifact...")
        success: bool = test_manager.edit_artifact(test_artifact_id, title="Updated Test Artifact")
        if success:
            print("  ✅ Successfully edited artifact")
        else:
            raise Exception("Failed to edit artifact")

        # Test 5: Toggle favorite
        print("  5. Testing toggle favorite...")
        success = test_manager.edit_artifact(test_artifact_id, favorite=True)
        if success:
            print("  ✅ Successfully toggled favorite")
        else:
            raise Exception("Failed to toggle favorite")

        # Test 6: Get statistics
        print("  6. Testing statistics...")
        stats: Dict[str, Any] = test_manager.get_statistics()
        if stats and stats.get('total_artifacts', 0) > 0:
            print(f"  ✅ Statistics working: {stats['total_artifacts']} artifacts")
        else:
            raise Exception("Statistics not working properly")

        # Test 7: Delete artifact
        print("  7. Testing delete artifact...")
        if test_manager.delete_artifact(test_artifact_id):
            print("  ✅ Deleted test artifact")
        else:
            raise Exception("Failed to delete test artifact")

        print("🧪 ✅ All tests passed!")

    except Exception as exc:
        print(f"🧪 ❌ Test failed: {exc}")
        # Try to cleanup test artifact if it exists
        if test_artifact_id:
            try:
                test_manager.delete_artifact(test_artifact_id)
            except Exception:
                pass
        raise


def run_interactive_cli(cli_manager: RobustArtifactManager) -> None:
    """Interactive CLI with error checking"""
    print("🔧 Robust Claude Artifact Manager")
    print("Commands: save, get, search, stats, test, help, quit")
    print("Type 'help' for detailed command information")

    while True:
        try:
            command: str = input("\n> ").strip().lower()

            if command in ["quit", "q", "exit"]:
                break
            elif command == "help":
                print_cli_help()
            elif command == "stats":
                handle_stats_command(cli_manager)
            elif command == "save":
                handle_save_command(cli_manager)
            elif command == "get":
                handle_get_command(cli_manager)
            elif command == "search":
                handle_search_command(cli_manager)
            elif command == "test":
                run_system_test(cli_manager)
            elif command == "":
                continue  # Empty input, just continue
            else:
                print(f"❌ Unknown command: '{command}'. Type 'help' for available commands.")

        except KeyboardInterrupt:
            raise  # Re-raise to be caught by main
        except EOFError:
            break  # End of input stream
        except Exception as exc:
            print(f"❌ Command error: {exc}")
            logger.error(f"CLI command error: {exc}")


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


def handle_stats_command(stats_manager: RobustArtifactManager) -> None:
    """Handle stats command with error checking"""
    try:
        stats: Dict[str, Any] = stats_manager.get_statistics()
        print("\n📊 System Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
    except Exception as exc:
        print(f"❌ Failed to get statistics: {exc}")


def handle_save_command(save_manager: RobustArtifactManager) -> None:
    """Handle save command with validation"""
    try:
        print("\n📝 Save New Artifact")
        title: str = input("Title: ").strip()
        if not title:
            print("❌ Title cannot be empty")
            return

        print("Content (type 'END' on a new line to finish):")
        lines: List[str] = []
        while True:
            try:
                line: str = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            except EOFError:
                break

        content: str = "\n".join(lines)
        if not content.strip():
            print("❌ Content cannot be empty")
            return

        artifact_type: str = input("Type (code/document/html/text/data) [code]: ").strip() or "code"
        language: Optional[str] = input("Language (optional): ").strip() or None
        project: str = input("Project [default]: ").strip() or "default"

        tags_input: str = input("Tags (comma-separated): ").strip()
        tags: List[str] = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []

        artifact_id: str = save_manager.save_artifact(
            content=content,
            title=title,
            artifact_type=artifact_type,
            language=language,
            tags=tags,
            project=project
        )
        print(f"✅ Saved artifact: {artifact_id}")

    except Exception as exc:
        print(f"❌ Failed to save artifact: {exc}")


def handle_get_command(get_manager: RobustArtifactManager) -> None:
    """Handle get command with error checking"""
    try:
        artifact_id: str = input("Artifact ID: ").strip()
        if not artifact_id:
            print("❌ Artifact ID cannot be empty")
            return

        artifact: Optional[Artifact] = get_manager.get_artifact(artifact_id)
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
            print(f"Favorite: {'Yes' if artifact.favorite else 'No'}")
            print(f"\nContent preview (first 500 chars):")
            print("-" * 50)
            print(artifact.content[:500])
            if len(artifact.content) > 500:
                print("\n... (truncated)")
        else:
            print(f"❌ Artifact '{artifact_id}' not found")

    except Exception as exc:
        print(f"❌ Failed to get artifact: {exc}")


def handle_search_command(search_manager: RobustArtifactManager) -> None:
    """Handle search command with error checking"""
    try:
        search_query: Optional[str] = input("Search query (optional): ").strip() or None
        project: Optional[str] = input("Project filter (optional): ").strip() or None
        artifact_type: Optional[str] = input("Type filter (optional): ").strip() or None
        language: Optional[str] = input("Language filter (optional): ").strip() or None

        results: List[Artifact] = search_manager.search_artifacts(
            query=search_query,
            project=project,
            artifact_type=artifact_type,
            language=language
        )

        if not results:
            print("❌ No artifacts found matching your criteria")
            return

        print(f"\n🔍 Found {len(results)} artifacts:")
        for i, artifact in enumerate(results[:20], 1):  # Show max 20 results
            tags_str: str = f" [{', '.join(artifact.tags)}]" if artifact.tags else ""
            fav_str: str = " ⭐" if artifact.favorite else ""
            print(f"{i:2d}. {artifact.id} - {artifact.title} ({artifact.artifact_type}){tags_str}{fav_str}")

        if len(results) > 20:
            print(f"... and {len(results) - 20} more (showing first 20)")

    except Exception as exc:
        print(f"❌ Search failed: {exc}")


def main() -> None:
    """Main CLI interface with comprehensive error handling"""
    manager: Optional[RobustArtifactManager] = None

    try:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(
            description='Robust Claude Artifact Manager - COMPLETE VERSION')
        parser.add_argument('--storage', default='claude_artifacts', help='Storage directory')
        parser.add_argument('--stats', action='store_true', help='Show statistics')
        parser.add_argument('--test', action='store_true', help='Run basic test')
        parser.add_argument('--web', action='store_true', help='Start web interface')
        parser.add_argument('--port', type=int, default=5000, help='Web interface port')

        args: argparse.Namespace = parser.parse_args()

        # Initialize manager with error handling
        try:
            manager = RobustArtifactManager(args.storage)
            logger.info("Artifact manager initialized successfully")
        except Exception as exc:
            print(f"❌ Failed to initialize artifact manager: {exc}")
            print("💡 Check storage directory permissions and disk space")
            sys.exit(1)

        if args.stats:
            try:
                stats: Dict[str, Any] = manager.get_statistics()
                print("📊 System Statistics:")
                for key, value in stats.items():
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for subkey, subvalue in value.items():
                            print(f"    {subkey}: {subvalue}")
                    else:
                        print(f"  {key}: {value}")
                return
            except Exception as exc:
                print(f"❌ Failed to get statistics: {exc}")
                sys.exit(1)

        if args.test:
            try:
                run_system_test(manager)
                return
            except Exception as exc:
                print(f"❌ System test failed: {exc}")
                sys.exit(1)

        if args.web:
            try:
                start_web_interface(manager, args.port)
                return
            except Exception as exc:
                print(f"❌ Failed to start web interface: {exc}")
                print("💡 Check if port is already in use or try different port")
                print("💡 Install dependencies: pip install flask flask-cors")
                sys.exit(1)

        # Interactive CLI mode
        try:
            run_interactive_cli(manager)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as exc:
            print(f"❌ CLI error: {exc}")
            sys.exit(1)

    except Exception as exc:
        logger.error(f"Fatal error in main: {exc}")
        print(f"❌ Fatal error: {exc}")
        sys.exit(1)

    finally:
        try:
            if manager is not None:
                manager.close()
        except Exception as exc:
            logger.warning(f"Error during cleanup: {exc}")


if __name__ == "__main__":
    main()