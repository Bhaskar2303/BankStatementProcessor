from pathlib import Path
import yaml


class ConfigRegistry:
    """Loads CrewAI agent and task YAML files from the config directory."""

    def __init__(self, config_dir: str | Path | None = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.agents_config = self._load_yaml(self.config_dir / "agents.yaml")
        self.tasks_config = self._load_yaml(self.config_dir / "tasks.yaml")

    @staticmethod
    def _load_yaml(file_path: Path) -> dict:
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_agents(self) -> dict:
        return self.agents_config

    def get_tasks(self) -> dict:
        return self.tasks_config

    def get_agent(self, name: str) -> dict:
        return self.agents_config.get(name, {})

    def get_task(self, name: str) -> dict:
        return self.tasks_config.get(name, {})
