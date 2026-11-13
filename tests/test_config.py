import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from car_rag.config import DEFAULT_MODEL, Settings, load_local_env


class ConfigurationTests(unittest.TestCase):
    def test_loads_local_env_values_without_overriding_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "# local settings\nOPENROUTER_MODEL='custom/model'\n"
                "OPENROUTER_API_KEY=file-key\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "process-key"}, clear=True):
                load_local_env(env_file)
                settings = Settings.from_env()

        self.assertEqual("custom/model", settings.model)
        self.assertEqual("process-key", settings.api_key)

    def test_missing_env_file_keeps_default_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            load_local_env(Path("missing-config.env"))
            settings = Settings.from_env()

        self.assertEqual(DEFAULT_MODEL, settings.model)

    def test_malformed_env_line_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("not-an-assignment\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1"):
                load_local_env(env_file)


if __name__ == "__main__":
    unittest.main()