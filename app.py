# Databricks notebook source
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from process_pipeline import BankStatementProcessor
from memory.store import add_episodic_entry
from telemetry.tracing import estimate_tokens, log_run_usage, get_run_cost, get_run_ids


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
BANK_STATEMENTS_DIR = DATA_DIR / "bank_statements"
APPROVED_DIR = DATA_DIR / "approved"
TRACE_DIR = ROOT_DIR / "trace"

for directory in (DATA_DIR, BANK_STATEMENTS_DIR, APPROVED_DIR, TRACE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


STAGE_MESSAGES = {
    "loaded": "Bank statement loaded...",
    "parallel_extraction": "Extracting bank statement sections...",
    "parallel_extraction_done": "Bank statement extraction completed...",
    "merge_starting": "Merging extracted bank statement data...",
    "merge_done": "Bank statement data merged...",
    "validation_starting": "Validating bank statement data...",
    "validation_done": "Bank statement validation completed...",
    "reflection_starting": "Reviewing validation errors and preparing correction...",
    "reflection_done": "Correction review completed...",
    "pipeline_completed": "Bank statement processing completed...",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json(value):
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        return json.loads(str(value))
    except Exception:
        return {"raw": str(value)}


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def stage_message(stage_name: str) -> str:

    if stage_name == "loaded":
        return STAGE_MESSAGES.get(stage_name)

    elif stage_name == "parallel_extraction":
        return STAGE_MESSAGES.get(stage_name)

    elif stage_name == "parallel_extraction_done":
        return STAGE_MESSAGES.get(stage_name)

    elif stage_name == "merge_starting":
        return STAGE_MESSAGES.get(stage_name)

    elif stage_name == "merge_done":
        return STAGE_MESSAGES.get(stage_name)

    elif stage_name == "validation_starting":
        return STAGE_MESSAGES.get(stage_name)

    elif stage_name == "validation_done":
        return STAGE_MESSAGES.get(stage_name)

    elif stage_name == "reflection_starting":
        return STAGE_MESSAGES.get(stage_name)

    elif stage_name == "reflection_done":
        return STAGE_MESSAGES.get(stage_name)

    elif stage_name == "pipeline_completed":
        return STAGE_MESSAGES.get(stage_name)

    return None


def build_run_cost(run_id: str, document_name: str, model_name: str, result: dict):
    document_text = result.get("document_text", "")
    merged_result = result.get("merged_result", "")
    parallel_results = result.get("parallel_results", {})
    validation_data = result.get("validation_data", {})
    reflection_data = result.get("reflection_data", {})

    prompt_text = document_text + json.dumps(parallel_results, ensure_ascii=False)
    output_text = json.dumps(
        {
            "merged_result": merged_result,
            "validation_data": validation_data,
            "reflection_data": reflection_data,
        },
        ensure_ascii=False,
    )

    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(output_text)

    usage_row = log_run_usage(
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        run_id=run_id,
        document_name=document_name,
        stage="streamlit_pipeline_run",
    )
    return usage_row


def process_pdf(pdf_path: Path):
    run_id = str(uuid.uuid4())
    processor = BankStatementProcessor()
    model_name = getattr(
        processor,
        "model_name",
        os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    )

    events = []
    final_result = None
    progress = st.progress(0)
    #status = st.empty()
    status_container = st.container()
    stages_seen = 0
    stage_history = []

    for event in processor.run_pipeline_stages(str(pdf_path)):
        events.append(event)
        stage_name = event.get("stages", "unknown")
        stages_seen += 1
        progress.progress(min(stages_seen / 8, 1.0))
        message = stage_message(stage_name)

        if 'stage_log' not in locals():
            stage_log = []
        stage_log.append(message)

        status_container.markdown(
             message,
             unsafe_allow_html=True
        )
        #stage_history.append((stage_name, ": " + message))

        #status.text(stage_message(stage_name, event))
        #with status_container:
        #    for msg in stage_history:
        #        st.write(msg)
        

        

        if stage_name == "pipeline_completed":
            final_result = event

    if final_result is None and events:
        final_result = events[-1]

    if final_result is None:
        raise RuntimeError("Pipeline did not return a final result.")

    cost_row = build_run_cost(run_id, pdf_path.name, model_name, final_result)
    final_result["run_id"] = run_id
    final_result["model_name"] = model_name
    final_result["cost_tracking"] = cost_row
    final_result["events"] = events
    return final_result


def approve_current_result(result: dict):
    statement_id = result.get("statement_id") or f"statement_{uuid.uuid4().hex[:8]}"
    approved_file = APPROVED_DIR / f"{statement_id}.json"

    payload = {
        "approval_status": "approved",
        "approved_at_utc": now_utc(),
        "run_id": result.get("run_id"),
        "statement_id": statement_id,
        "bank_name": result.get("bank_name"),
        "model_name": result.get("model_name"),
        "cost_tracking": result.get("cost_tracking"),
        "merged_data": result.get("merged_data") or safe_json(result.get("merged_result")),
        "validation_data": result.get("validation_data"),
        "reflection_data": result.get("reflection_data"),
    }

    write_json(approved_file, payload)
    return approved_file


def reject_and_reextract(result: dict, human_note: str, source_pdf_path: Path):
    if not human_note.strip():
        raise ValueError("Human note is required before re-extraction.")

    statement_id = result.get("statement_id") or source_pdf_path.stem
    bank_name = result.get("bank_name") or "UNKNOWN"
    merged_result = result.get("merged_result") or json.dumps(
        result.get("merged_data", {}),
        ensure_ascii=False,
    )
    document_text = result.get("document_text", "")

    # Do not write rejected JSON into any rejected subdirectory.
    # Store correction in episodic memory, then re-run extraction.
    add_episodic_entry(
        bank_name=bank_name,
        statement_id=statement_id,
        correction=human_note.strip(),
        confidence=1.0,
        source="human-review-reject",
        human_feedback=True
        # extra={
        #     "action": "reject_to_reextract",
        #     "run_id": result.get("run_id"),
        #     "validation_data": result.get("validation_data"),
        #     "error_message": human_note.strip(),
        # },
    )

    processor = BankStatementProcessor()
    try:
        reflection = processor.force_reflection(
            document_text=document_text,
            bank_name=bank_name,
            statement_id=statement_id,
            merge_final_result=merged_result,
            human_review=human_note.strip(),
        )
        st.session_state["last_human_reflection"] = reflection
    except Exception as exc:
        st.warning(f"Human note saved to episodic memory, but reflection agent failed: {exc}")

    return process_pdf(source_pdf_path)


def list_existing_pdfs():
    return sorted(BANK_STATEMENTS_DIR.glob("*.pdf"))


def list_data_subdirectories():
    return sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()]) if DATA_DIR.exists() else []


def build_cost_metrics_row(cost_tracking: dict | None):
    if not cost_tracking:
        return None

    input_tokens = int(cost_tracking.get("input_tokens", cost_tracking.get("prompt_tokens", 0)) or 0)
    output_tokens = int(cost_tracking.get("output_tokens", cost_tracking.get("completion_tokens", 0)) or 0)
    total_tokens = int(cost_tracking.get("total_tokens", input_tokens + output_tokens) or 0)
    total_cost = float(cost_tracking.get("total_cost_usd", 0.0) or 0.0)

    return {
        "model": cost_tracking.get("model", cost_tracking.get("model_name", "")),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "total_cost USD": round(total_cost, 8),
    }


def render_cost_metrics(cost_tracking: dict | None):
    table_row = build_cost_metrics_row(cost_tracking)
    if not table_row:
        st.info("Cost metrics unavailable.")
        return

    st.dataframe([table_row], use_container_width=True, hide_index=True)


st.set_page_config(page_title="Bank Statement Process", layout="wide")
st.title("Bank Statement Process")

if "current_pdf_path" not in st.session_state:
    st.session_state["current_pdf_path"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_action" not in st.session_state:
    st.session_state["last_action"] = None
if "show_cost_metrics" not in st.session_state:
    st.session_state["show_cost_metrics"] = False

with st.sidebar:
    st.header("data")
    st.write("Subdirectories")
    subdirectories = list_data_subdirectories()
    if subdirectories:
        for subdirectory in subdirectories:
            st.write(f"- {subdirectory}")
    else:
        st.info("No subdirectories found under data.")

    st.divider()
    st.subheader("bank_statements")
    existing_pdfs = list_existing_pdfs()

    selected_pdf_name = None
    if existing_pdfs:
        selected_pdf_name = st.selectbox(
            "Choose PDF document",
            [p.name for p in existing_pdfs],
            index=0,
        )
        st.session_state["current_pdf_path"] = str(BANK_STATEMENTS_DIR / selected_pdf_name)
    else:
        st.warning("No PDF documents found in data/bank_statements.")

    process_clicked = st.button(
        "process_document",
        type="primary",
        use_container_width=True,
        disabled=not bool(selected_pdf_name),
    )

    st.divider()
    st.subheader("Run History")

    try:
        run_ids = get_run_ids()
    except Exception as exc:
        run_ids = []
        st.warning(f"Unable to load run history: {exc}")

    if run_ids:
        selected_run_id = st.selectbox(
            "Select Run ID",
            run_ids,
            index=None,
            placeholder="Choose a run ID",
            key="run_history",
        )

        if selected_run_id:
            historical_run_cost = get_run_cost(selected_run_id)
            if historical_run_cost:
                st.markdown("### Selected Run Cost")
                render_cost_metrics(historical_run_cost)
            else:
                st.info("Cost metrics unavailable for selected run.")
    else:
        st.info("No historical runs available.")

if process_clicked and st.session_state.get("current_pdf_path"):
    with st.spinner("Running process_pipeline.py stages..."):
        result = process_pdf(Path(st.session_state["current_pdf_path"]))
        st.session_state["last_result"] = result
        st.session_state["last_action"] = "processed"
        st.session_state["show_cost_metrics"] = True
    st.success("Document processed and merged. Review the output below.")

result = st.session_state.get("last_result")
current_pdf = st.session_state.get("current_pdf_path")

if not result:
    st.info("Select a PDF from data/bank_statements in the sidebar, then click process_document.")
else:
    st.subheader("Processed document preview")

    

    st.write("Merged output")
    st.json(
    result.get("merged_data")
    or safe_json(result.get("merged_result"))
    )


    st.divider()
    st.subheader("Final action")
    human_note = st.text_area(
        "Human note for re-extraction",
        placeholder="Example: Customer address is on page 2 header. Do not use merchant address as mailing_address.",
        height=120,
    )

    approve_col, reject_col = st.columns(2)
    with approve_col:
        approve_clicked = st.button(
            "Approve",
            type="primary",
            use_container_width=True,
        )
    with reject_col:
        reject_clicked = st.button(
            "Reject with human note",
            use_container_width=True,
        )

    if approve_clicked:
        with st.spinner("Writing approved JSON file..."):
            approved_file = approve_current_result(result)
        approved_file = approve_current_result(result)
        st.session_state["last_action"] = "approved"
        st.session_state["show_cost_metrics"] = True
        st.success(f"Approved JSON written to {approved_file}")
        st.success("✅ Process Completed Successfully")

    if reject_clicked:
        try:
            with st.spinner("Saving human note to episodic memory and re-running extraction..."):
                new_result = reject_and_reextract(result, human_note, Path(current_pdf))
                st.session_state["last_result"] = new_result
                st.session_state["last_action"] = "rejected_reextracted"
                st.session_state["show_cost_metrics"] = True
            st.success("Re-extraction completed. No rejected JSON folder was used.")
            st.text("Bank statement re-extracted successfully. Review the updated outputs above.")
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.caption("Run locally with: streamlit run app.py")