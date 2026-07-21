from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cultural_mood_tracker.cli.cloud_start import cloud_data_is_ready, streamlit_command


class CloudStartTests(unittest.TestCase):
    def test_cloud_data_requires_chroma(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            chroma_path = root / "chroma_db" / "chroma.sqlite3"
            self.assertFalse(cloud_data_is_ready(root))
            chroma_path.parent.mkdir(parents=True)
            chroma_path.touch()
            self.assertTrue(cloud_data_is_ready(root))

    def test_streamlit_command_uses_provider_port(self) -> None:
        root = Path("/app")
        command = streamlit_command(root, "10000")
        self.assertIn("--server.port=10000", command)
        self.assertIn("--server.address=0.0.0.0", command)


if __name__ == "__main__":
    unittest.main()
