# Databricks notebook source
from dotenv import load_dotenv
from crewai import Crew, Agent, Task, Process
from crewai.llm import LLM
from pathlib import Path
import json
import os

from utils import validate_path, extract_text_from_pdf, parse_json_file
from memory.store import add_episodic_entry, format_memory_for_context
from config.config_registry import ConfigRegistry

load_dotenv()

# for key in os.environ:
#     if key.startswith("OPENAI_"):
#         print(f"{key}: {os.environ[key]}")

MAX_TOKENS = 16000


def safe_json_loads(value, default=None):
    if default is None:
        default = {}
    if isinstance(value, dict):
        return value
    text = parse_json_file(value)
    if isinstance(text, dict):
        return text
    try:
        return json.loads(str(text))
    except Exception:
        return default



class BankStatementProcessor:
    ALLOWED_DIR = Path(".").resolve() / "data" / "bank_statements"

    def __init__(self):
        load_dotenv()
        print("model:", os.getenv("OPENAI_MODEL_NAME"))
        print("api_key:", os.getenv("OPENAI_API_KEY"))
        print("endpoint:", os.getenv("OPENAI_API_BASE"))
        print("api_version:", os.getenv("OPENAI_API_VERSION"))

        
        self.llm = LLM(
            model=os.getenv("OPENAI_MODEL_NAME"),
            api_key=os.getenv("OPENAI_API_KEY"),
            endpoint=os.getenv("OPENAI_API_BASE"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            temperature=0.2,
        )
        self.registry = ConfigRegistry()
        self.agents = self._create_agents()

    def _create_agents(self):
        agents = {}
        for agent_name, cfg in self.registry.get_agents().items():
            agents[agent_name] = Agent(
                role=cfg["role"],
                goal=cfg["goal"],
                backstory=cfg["backstory"],
                llm=self.llm,
                verbose=True,
            )
    

        return agents

    def _build_task(self, task_key, async_execution=False, extra_description=None):
        tasks = self.registry.get_tasks()
        cfg = tasks[task_key]
        description = cfg["description"]
        if extra_description:
            description = description + "\n\n" + extra_description
        return Task(
            description=description,
            agent=self.agents[cfg["agent"]],
            async_execution=async_execution,
            expected_output=cfg["expected_output"],
        )

    def extract_parallel(self, document_text, memory_context=""):
        task_account = self._build_task("account_task", async_execution=False)
        task_identity = self._build_task("identity_task", async_execution=False)
        task_transactions = self._build_task("transactions_task", async_execution=False)


        # print("\n=== TASKS ===")
        # for t in [task_account, task_identity, task_transactions]:
        #     print(t)

        # print("\n=== MEMORY CONTEXT ===")
        # print(type(memory_context))
        # print(memory_context)

        
        try:
            print("Starting parallel extraction...")
            crew = Crew(
                agents=[task_account.agent, task_identity.agent, task_transactions.agent],
                tasks=[task_account, task_identity, task_transactions],
                process=Process.sequential,
                verbose=True,
            )
        except Exception as e:
            import traceback
            print(f"Error occurred while creating Crew: {e}")
            traceback.print_exc()
            
        crew.kickoff(inputs={"document_text": document_text, "memory_context": memory_context or ""})

        for task in (task_account, task_identity, task_transactions):
            thread = getattr(task, "thread", None)
            if thread:
                thread.join()

        return {
            "account_result": str(task_account),
            "identity_result": str(task_identity),
            "transactions_result": str(task_transactions),
        }

    def merge_results(self, parallel_results):
        merge_task = self._build_task("merge_task")
        crew = Crew(
            agents=[merge_task.agent],
            tasks=[merge_task],
            process=Process.sequential,
            verbose=True,
        )
        merge_result = crew.kickoff(inputs={
            "account_result": parallel_results.get("account_result"),
            "identity_result": parallel_results.get("identity_result"),
            "transactions_result": parallel_results.get("transactions_result"),
        })
        
        merged_data = safe_json_loads(merge_result, default={})
        return merged_data, json.dumps(merged_data, indent=2, ensure_ascii=False)

    def validate_merged_data(self, merge_final_result):
        validation_task = self._build_task(
            "validation_task",
            extra_description="MERGED_DATA_TO_VALIDATE:\n{merged_data}",
        )
        crew = Crew(
            agents=[validation_task.agent],
            tasks=[validation_task],
            process=Process.sequential,
            verbose=True,
        )
        validation_result = crew.kickoff(inputs={"merged_data": merge_final_result})
        validation_data = safe_json_loads(validation_result, default={"is_valid": "no", "validation_errors": []})
        
        return validation_data

    def reflect_on_validation(self, document_text, bank_name, statement_id, merge_final_result, errors, source="reflection"):
        reflection_task = self._build_task("reflection_task")
        crew = Crew(
            agents=[reflection_task.agent],
            tasks=[reflection_task],
            process=Process.sequential,
            verbose=True,
        )
        reflection_result = crew.kickoff(inputs={
            "bank_name": bank_name or "UNKNOWN",
            "statement_id": statement_id,
            "document_text": document_text,
            "merged_final_result": merge_final_result,
            "errors": json.dumps(errors, indent=2, ensure_ascii=False),
        })
        
        reflection_data = safe_json_loads(reflection_result, default={})
        correction = reflection_data.get("corrected_extraction") or reflection_data.get("correction") or reflection_data.get("mistake_description") or str(errors)
        confidence = reflection_data.get("confidence_level") or reflection_data.get("confidence") or 0.0

        
        entry = add_episodic_entry(
            bank_name=bank_name,
            statement_id=statement_id,
            correction=correction,
            confidence=confidence,
            source=source,
            human_feedback=(source == "human")
            
        )
        reflection_data["episodic_entry_id"] = entry["id"]
        return reflection_data

    def run_pipeline_stages(self, pdf_path):
        path = validate_path(pdf_path, self.ALLOWED_DIR)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        statement_id = path.stem
        document_text = extract_text_from_pdf(str(path), self.ALLOWED_DIR)
        yield {"stages": "loaded", "statement_id": statement_id, "document_text": document_text}
               

        memory_context = format_memory_for_context("UNKNOWN")
        yield {"stages": "parallel_extraction"}
               
        parallel_results = self.extract_parallel(document_text, memory_context=memory_context)
        yield {"stages": "parallel_extraction_done", "results": parallel_results}

        yield {"stages": "merge_starting" }
               
        merge_data, merge_final_result = self.merge_results(parallel_results)
        yield {"stages": "merge_done", "merged_data": merge_data, "merged_result": merge_final_result}
               

        yield {"stages": "validation_starting"}
                
        validation_data = self.validate_merged_data(merge_final_result)
        yield {"stages": "validation_done", "validation_data": validation_data}
               

        bank_name = merge_data.get("bank_name") or "UNKNOWN"
        reflection_data = None
        is_valid_value = str(validation_data.get("is_valid", "no")).strip().lower()
        is_valid = is_valid_value in {"yes", "true", "valid", "1"}
        errors = validation_data.get("validation_errors") or validation_data.get("errors") or []

        if not is_valid:
            yield {"stages": "reflection_starting"}
                   
            reflection_data = self.reflect_on_validation(
                document_text=document_text,
                bank_name=bank_name,
                statement_id=statement_id,
                merge_final_result=merge_final_result,
                errors=errors,
                source="reflection",
            )
            yield {"stages": "reflection_done", "reflection_data": reflection_data}

        yield {
            "stages": "pipeline_completed",
            "statement_id": statement_id,
            "bank_name": bank_name,
            "document_text": document_text,
            "parallel_results": parallel_results,
            "merged_data": merge_data,
            "merged_result": merge_final_result,
            "validation_data": validation_data,
            "reflection_data": reflection_data,
        }

    def process_document(self, pdf_path):
        result = None
        for stage in self.run_pipeline_stages(pdf_path):
            result = stage
        return result

    def force_reflection(self, document_text, bank_name, statement_id, merge_final_result, human_review):
        return self.reflect_on_validation(
            document_text=document_text,
            bank_name=bank_name,
            statement_id=statement_id,
            merge_final_result=merge_final_result,
            errors=[{"field": "human_review", "message": human_review}],
            source="human",
        )


if __name__ == "__main__":
    pdf_path = "data/bank_statements/bank_statement.pdf"
    processor = BankStatementProcessor()
    for event in processor.run_pipeline_stages(pdf_path):
        print(event.get("stages"), event.keys())