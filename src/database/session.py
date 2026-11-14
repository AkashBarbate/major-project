import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine import Engine

# Database configuration: SQLite persisted under data/
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DB_DIR, exist_ok=True)

SQLITE_PATH = os.getenv("NEWSLENS_DB_PATH", os.path.join(DB_DIR, "newslens.db"))
SQLALCHEMY_DATABASE_URL = f"sqlite:///{SQLITE_PATH}"

# Connect args: check_same_thread=False for FastAPI concurrency
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

# Enable foreign key support for SQLite
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency to yield database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables in SQLite database."""
    Base.metadata.create_all(bind=engine)
    print(f"[NewsLens DB] SQLite initialized at: {SQLITE_PATH}")

