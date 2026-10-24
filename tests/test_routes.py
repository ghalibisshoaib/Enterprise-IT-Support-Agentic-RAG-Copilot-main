import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes


class IngestRouteTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(routes.router)
        self.client = TestClient(app)

    def test_upload_is_disabled_without_admin_key(self):
        with patch.object(routes.settings, "admin_api_key", ""):
            response = self.client.post(
                "/api/ingest",
                files={"file": ("note.txt", b"content", "text/plain")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Document ingestion is not configured")

    def test_upload_rejects_files_over_size_limit(self):
        with tempfile.TemporaryDirectory() as upload_dir:
            with (
                patch.object(routes.settings, "admin_api_key", "test-secret"),
                patch.object(routes.settings, "max_upload_bytes", 3),
                patch.object(routes.settings, "upload_dir", upload_dir),
            ):
                response = self.client.post(
                    "/api/ingest",
                    headers={"X-Admin-Key": "test-secret"},
                    files={"file": ("note.txt", b"four", "text/plain")},
                )

            self.assertEqual(response.status_code, 413)
            self.assertEqual(list(Path(upload_dir).iterdir()), [])

    def test_upload_indexes_file_and_preserves_original_source_name(self):
        document = SimpleNamespace(metadata={})
        with tempfile.TemporaryDirectory() as upload_dir:
            with (
                patch.object(routes.settings, "admin_api_key", "test-secret"),
                patch.object(routes.settings, "upload_dir", upload_dir),
                patch.object(routes, "load_file", return_value=[document]),
                patch.object(routes, "chunk_documents", return_value=["chunk"]),
                patch.object(routes, "add_documents", return_value=["vector-id"]),
            ):
                response = self.client.post(
                    "/api/ingest",
                    headers={"X-Admin-Key": "test-secret"},
                    files={"file": ("policy.txt", b"policy", "text/plain")},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["file"], "policy.txt")
        self.assertEqual(response.json()["chunks"], 1)
        self.assertEqual(document.metadata["source"], "policy.txt")


if __name__ == "__main__":
    unittest.main()
