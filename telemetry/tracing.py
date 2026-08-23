from pathlib import Path
from datetime import datetime, timezone
import csv
import uuid

try:
    import pandas as pd
except Exception:
    pd = None

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

MODEL_PRICING = {
    "gpt-4o": {"input_cost_per_1m": 5.00, "output_cost_per_1m": 15.00},
    "gpt-4.1-mini": {"input_cost_per_1m": 0.40, "output_cost_per_1m": 1.60},
    "gpt-4o-mini": {"input_cost_per_1m": 0.15, "output_cost_per_1m": 0.60},
}


def estimate_tokens(text) -> int:
    text = str(text or "")
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def calculate_token_cost(model_name, prompt_tokens, completion_tokens):
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["gpt-4o-mini"])

    input_cost = (int(prompt_tokens) / 1_000_000) * pricing["input_cost_per_1m"]
    output_cost = (int(completion_tokens) / 1_000_000) * pricing["output_cost_per_1m"]

    return {
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(input_cost + output_cost, 8),
    }


def log_run_usage(
    model_name,
    prompt_tokens,
    completion_tokens,
    run_id=None,
    document_name=None,
    stage="run",
    output_file=None,
):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_file = Path(output_file) if output_file else DATA_DIR / "token_usage_log.csv"
    run_id = run_id or str(uuid.uuid4())

    cost = calculate_token_cost(
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    row = {
        "run_id": run_id,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "document_name": document_name or "",
        "stage": stage,
        "model_name": model_name,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        **cost,
    }

    write_header = not output_file.exists()

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return row


def get_cost_history(output_file=None):
    """
    Return complete run-cost history from token_usage_log.csv.

    If pandas is available, returns a pandas DataFrame.
    If pandas is not available, returns a list of dictionaries.
    """
    output_file = Path(output_file) if output_file else DATA_DIR / "token_usage_log.csv"

    if not output_file.exists():
        if pd is not None:
            return pd.DataFrame()
        return []

    if pd is not None:
        return pd.read_csv(output_file)

    rows = []
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows


def get_run_ids(output_file=None):
    """
    Return unique run IDs ordered newest first.
    """
    history = get_cost_history(output_file)

    if pd is not None:
        if history.empty or "run_id" not in history.columns:
            return []

        sort_col = "run_timestamp_utc" if "run_timestamp_utc" in history.columns else None
        if sort_col:
            history = history.sort_values(sort_col, ascending=False)

        return history["run_id"].dropna().drop_duplicates().astype(str).tolist()

    if not history:
        return []

    sorted_rows = sorted(
        history,
        key=lambda row: row.get("run_timestamp_utc", ""),
        reverse=True,
    )
    seen = set()
    run_ids = []
    for row in sorted_rows:
        run_id = str(row.get("run_id", "")).strip()
        if run_id and run_id not in seen:
            seen.add(run_id)
            run_ids.append(run_id)
    return run_ids


def get_run_cost(run_id, output_file=None):
    """
    Return cost metrics for a specific run_id.
    """
    if not run_id:
        return None

    history = get_cost_history(output_file)

    if pd is not None:
        if history.empty or "run_id" not in history.columns:
            return None

        match = history[history["run_id"].astype(str) == str(run_id)]
        if match.empty:
            return None

        if "run_timestamp_utc" in match.columns:
            match = match.sort_values("run_timestamp_utc", ascending=True)

        return match.iloc[-1].to_dict()

    matches = [row for row in history if str(row.get("run_id")) == str(run_id)]
    if not matches:
        return None

    matches = sorted(
        matches,
        key=lambda row: row.get("run_timestamp_utc", ""),
    )
    return matches[-1]


def get_latest_run(output_file=None):
    """
    Return the most recent run-cost record.
    """
    history = get_cost_history(output_file)

    if pd is not None:
        if history.empty:
            return None

        if "run_timestamp_utc" in history.columns:
            latest = history.sort_values("run_timestamp_utc", ascending=False).iloc[0]
        else:
            latest = history.iloc[-1]

        return latest.to_dict()

    if not history:
        return None

    sorted_rows = sorted(
        history,
        key=lambda row: row.get("run_timestamp_utc", ""),
        reverse=True,
    )
    return sorted_rows[0]
