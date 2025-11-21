from sqlalchemy import create_engine, text
from core.config import DATABASE_URL

_engine = create_engine(DATABASE_URL, future=True)

def init_db() -> None:
    with _engine.begin() as c:
        c.execute(text("""
        CREATE TABLE IF NOT EXISTS predictions(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          mode TEXT NOT NULL,
          input_json TEXT NOT NULL,
          prob_default REAL NOT NULL,
          pred_label INTEGER NOT NULL
        );
        """))

def log(ts: str, mode: str, payload_json: str, prob: float, label: int) -> None:
    with _engine.begin() as c:
        c.execute(
            text("INSERT INTO predictions(ts,mode,input_json,prob_default,pred_label) "
                 "VALUES(:ts,:m,:p,:pr,:l)"),
            {"ts": ts, "m": mode, "p": payload_json, "pr": float(prob), "l": int(label)}
        )
