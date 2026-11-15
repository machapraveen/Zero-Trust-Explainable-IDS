"""
Cross-Dialect SQL Date/Time Formatting Helpers
==============================================
Ensures identical query behavior across SQLite (local .venv) and PostgreSQL (Docker).
"""

from sqlalchemy import Integer, cast, extract, func


def sql_format_dt(db, column, mode: str = "hour_full"):
    """
    Format a DateTime column across SQLite and PostgreSQL dialects.
      - 'hour_full':  YYYY-MM-DD HH:00:00
      - 'hour_short': YYYY-MM-DD HH:00
      - 'day':        YYYY-MM-DD
      - 'dow':        integer 0 (Sunday) .. 6 (Saturday)
      - 'hour_num':   integer 0 .. 23
    """
    dialect = db.engine.dialect.name

    if dialect == "postgresql":
        if mode == "hour_full":
            return func.to_char(column, "YYYY-MM-DD HH24:00:00")
        if mode == "hour_short":
            return func.to_char(column, "YYYY-MM-DD HH24:00")
        if mode == "day":
            return func.to_char(column, "YYYY-MM-DD")
        if mode == "dow":
            return cast(extract("dow", column), Integer)
        if mode == "hour_num":
            return cast(extract("hour", column), Integer)
        return func.to_char(column, "YYYY-MM-DD HH24:00:00")

    # SQLite & fallback
    if mode == "hour_full":
        return func.strftime("%Y-%m-%d %H:00:00", column)
    if mode == "hour_short":
        return func.strftime("%Y-%m-%d %H:00", column)
    if mode == "day":
        return func.strftime("%Y-%m-%d", column)
    if mode == "dow":
        return cast(func.strftime("%w", column), Integer)
    if mode == "hour_num":
        return cast(func.strftime("%H", column), Integer)
    return func.strftime("%Y-%m-%d %H:00:00", column)

