import json
import unittest
import urllib.parse
from pathlib import Path


class DownloadLinksValidationTests(unittest.TestCase):
    def setUp(self):
        models_file = Path(__file__).resolve().parent.parent / "models.json"
        with open(models_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def test_all_configured_download_urls_are_valid_and_non_empty(self):
        models_with_links = [m for m in self.data["models"] if m.get("downloadUrl")]
        self.assertGreater(len(models_with_links), 0)

        for m in models_with_links:
            url = m["downloadUrl"]
            parsed = urllib.parse.urlparse(url)
            self.assertIn(parsed.scheme, ("http", "https"), f"Invalid scheme for {m['name']}: {url}")
            self.assertTrue(parsed.netloc, f"Missing domain in URL for {m['name']}: {url}")
            self.assertTrue(parsed.path, f"Missing path in URL for {m['name']}: {url}")

    def test_models_have_valid_metadata_and_features(self):
        for m in self.data["models"]:
            self.assertTrue(m["name"], "Model name cannot be empty")
            self.assertTrue(m["size"], f"Model {m['name']} missing size")
            self.assertTrue(m["weight"], f"Model {m['name']} missing weight")
            self.assertIsInstance(m["features"], list, f"Features for {m['name']} must be a list")


if __name__ == "__main__":
    unittest.main()
