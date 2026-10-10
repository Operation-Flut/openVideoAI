import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.models.downloader import ModelDownloader, ModelInfo


class ModelDownloadTests(unittest.TestCase):
    @patch("src.models.downloader.requests.get")
    def test_html_page_is_not_saved_as_a_model(self, get):
        response = Mock()
        response.headers = {"content-type": "text/html; charset=utf-8"}
        get.return_value = response
        model = ModelInfo(
            name="Test",
            description="",
            size="",
            weight="",
            url="https://example.com",
            download_url="https://example.com/releases",
            description_short="",
            features=[],
        )

        with tempfile.TemporaryDirectory() as directory:
            downloader = ModelDownloader(models_dir=directory)
            self.assertFalse(downloader.download_model(model))
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(response.close.call_count, 1)


if __name__ == "__main__":
    unittest.main()
