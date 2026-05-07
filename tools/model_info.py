from typing import Dict, Any, Optional


class ModelInfoTool:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path

    def get_benchmark(self, model_name: str, task: Optional[str] = None) -> Dict[str, Any]:
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            if task:
                cursor = conn.execute("SELECT * FROM ai_models WHERE name LIKE ? AND task LIKE ?", (f"%{model_name}%", f"%{task}%"))
            else:
                cursor = conn.execute("SELECT * FROM ai_models WHERE name LIKE ?", (f"%{model_name}%",))
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return {"models": rows, "count": len(rows)}
        except Exception as e:
            return {"error": str(e)}
