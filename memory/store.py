# Databricks notebook source
from pathlib import Path
import json
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EPISODIC_FILE = DATA_DIR / "episodic.json"
SEMANTIC_FILE = DATA_DIR / "semantic.json"
THRESHOLD = 2


def load(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_episodic():
    data = load(EPISODIC_FILE, [])
    return data if isinstance(data, list) else []


def load_semantic():
    data = load(SEMANTIC_FILE, {})
    return data if isinstance(data, dict) else {}


def normalize(text):
    return "".join(str(text or "").lower().split())


def _extract_error_message(entry):
    extra = entry.get("extra") or {}
    if isinstance(extra, dict):
        if extra.get("error_message"):
            return str(extra.get("error_message"))
        errors = extra.get("errors")
        if errors:
            return json.dumps(errors, ensure_ascii=False, sort_keys=True)
        validation_data = extra.get("validation_data")
        if validation_data:
            return json.dumps(validation_data, ensure_ascii=False, sort_keys=True)
    return str(entry.get("correction") or "")


def add_episodic_entry(bank_name, statement_id, correction, confidence=0.0, source="reflection-agent", human_feedback=False):
    episodic = load_episodic()
    entry = {
        "id": f"E{len(episodic) + 1}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bank_name": bank_name or "UNKNOWN",
        "statement_id": statement_id,
        "correction": correction or "",
        "confidence": confidence,
        "source": source,
        "human_feedback": bool(human_feedback),
    }
    #if extra:
    #    entry["extra"] = extra

    episodic.append(entry)
    save(EPISODIC_FILE, episodic)

    # Auto-promote recurring extraction errors/corrections into semantic memory.
    # Threshold is inclusive: when the same error/correction appears 2 or more times
    # for the same bank, save it as a reusable rule for future statements.
    semantic_after_promotion = promote(entry.get("bank_name"), entry, all_episodic=episodic)
    entry["semantic_promotion"] = bool(semantic_after_promotion)
    save(EPISODIC_FILE, episodic)
    return entry


def confirm_episodic_entry(entry_id):
    episodic = load_episodic()
    for entry in episodic:
        if entry.get("id") == entry_id:
            entry["human_feedback"] = True
            save(EPISODIC_FILE, episodic)
            return entry
    return None


def promote(bank_name, new_entry, all_episodic=None):
    all_episodic = all_episodic or load_episodic()
    bank_name = bank_name or "UNKNOWN"
    norm_error = normalize(_extract_error_message(new_entry))
    norm_correction = normalize(new_entry.get("correction"))

    similar_entries = []
    for entry in all_episodic:
        if entry.get("bank_name") != bank_name:
            continue
        same_error = norm_error and normalize(_extract_error_message(entry)) == norm_error
        same_correction = norm_correction and normalize(entry.get("correction")) == norm_correction
        if same_error or same_correction:
            similar_entries.append(entry)

    if len(similar_entries) < THRESHOLD:
        return None

    semantic = load_semantic()
    rules = semantic.setdefault(bank_name, [])
    rule_text = new_entry.get("correction", "")
    error_message = _extract_error_message(new_entry)

    rule_exists = any(
        normalize(rule.get("correction")) == normalize(rule_text)
        or normalize(rule.get("error_message")) == normalize(error_message)
        for rule in rules
    )

    if not rule_exists:
        rules.append({
            "rule_type": "recurring_bank_statement_error_correction",
            "error_message": error_message,
            "correction": rule_text,
            "occurrence_count": len(similar_entries),
            "threshold": THRESHOLD,
            "source_entry_id": new_entry.get("id"),
            "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        save(SEMANTIC_FILE, semantic)
    return semantic


def format_memory_for_context(bank_name):
    semantic = load_semantic()
    episodic = load_episodic()
    bank_name = bank_name or "UNKNOWN"
    bank_rules = semantic.get(bank_name, [])
    bank_episodic = [entry for entry in episodic if entry.get("bank_name") == bank_name]
    parts = []
    if bank_rules:
        parts.append(
            "Known semantic extraction rules:\n"
            + "\n".join(
                f"- Error: {r.get('error_message', '')} | Correction: {r.get('correction')}"
                for r in bank_rules
            )
        )
    if bank_episodic:
        parts.append(
            "Recent episodic corrections:\n"
            + "\n".join(f"- {e.get('correction')}" for e in bank_episodic[-5:])
        )
    return "\n\n".join(parts) if parts else "No prior memory available."
