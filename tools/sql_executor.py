import sqlite3
from typing import List, Dict, Any


class SQLExecutor:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path

    def execute(self, sql: str, database: str = "default") -> Dict[str, Any]:
        if not self._is_safe_sql(sql):
            return {"error": "Unsafe SQL rejected", "sql": sql}

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            result = {"columns": [desc[0] for desc in cursor.description], "rows": [dict(row) for row in rows]}
            conn.close()
            return result
        except Exception as e:
            return {"error": str(e), "sql": sql}

    def _is_safe_sql(self, sql: str) -> bool:
        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
        return not any(kw in sql.upper() for kw in dangerous)

    def get_schema(self) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            schema_parts = []
            for (table,) in tables:
                cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
                col_desc = ", ".join([f"{c[1]} ({c[2]})" for c in cols])
                schema_parts.append(f"表 {table}: {col_desc}")
            conn.close()
            return "\n".join(schema_parts)
        except Exception as e:
            return f"Error: {e}"
