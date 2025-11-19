import json
from pathlib import Path
from typing import Union

def load_schema(path: Union[str, Path]) -> dict:
    p = Path(path)
    with p.open("r") as f:
        return json.load(f)
