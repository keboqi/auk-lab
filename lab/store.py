import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def validate_id(value: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid artifact identifier")
    return value


class Store:
    def __init__(self, root: Path):
        self.root = root
        for name in ("jobs", "sources", "outputs"):
            (root / name).mkdir(parents=True, exist_ok=True)

    def path(self, group: str, identifier: str, suffix: str) -> Path:
        return self.root / group / (validate_id(identifier) + suffix)

    def write(self, group: str, record: dict):
        path = self.path(group, record["id"], ".json")
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(path)

    def read(self, group: str, identifier: str) -> dict:
        return json.loads(self.path(group, identifier, ".json").read_text(encoding="utf-8"))

    def list(self, group: str) -> list[dict]:
        records = []
        for path in (self.root / group).glob("*.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                continue
        return sorted(records, key=lambda record: record.get("created_at", ""), reverse=True)
