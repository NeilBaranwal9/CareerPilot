import logging
import os
from typing import Any

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base

logger = logging.getLogger("recruiting-platform.db.session")


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    """Enable foreign key constraints in SQLite."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_db_engine(db_path: str = "data/platform.db") -> Engine:
    """
    Creates and returns a SQLAlchemy Engine.
    Ensures the parent directory exists.
    """
    # Resolve absolute path
    abs_path = os.path.abspath(db_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    database_url = f"sqlite:///{abs_path}"
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    return engine


def auto_migrate(engine: Engine) -> list[str]:
    """
    Adds any columns declared on the models but missing from existing SQLite tables.
    New columns are all nullable, so `ALTER TABLE ... ADD COLUMN` is safe and keeps existing rows intact.
    Returns the list of "table.column" entries that were added.
    """
    added: list[str] = []
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing_columns:
                    continue
                column_type = column.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column_type}'))
                added.append(f"{table.name}.{column.name}")
    if added:
        logger.info(f"Auto-migrated database columns: {', '.join(added)}")
    return added


def init_db(db_path: str = "data/platform.db") -> None:
    """Creates all tables in the SQLite database if they don't exist and adds any missing columns."""
    engine = get_db_engine(db_path)
    Base.metadata.create_all(engine)
    try:
        auto_migrate(engine)
    except Exception as e:
        logger.warning(f"Failed to auto-migrate database schema: {e}")


def get_session_factory(db_path: str = "data/platform.db") -> sessionmaker[Session]:
    """Returns a sessionmaker factory configured for the SQLite DB."""
    engine = get_db_engine(db_path)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
