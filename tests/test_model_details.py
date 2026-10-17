"""Model details metadata tests.

The model details popup is catalog-driven: the media a model is good for, the
wording of that media value and the `bestFor` use-case note all come from
`models.json` through the registry and the API response. These tests cover that
chain - catalog validity, the shared label helper, and both route payloads - so
the popup cannot quietly lose a field.
"""
import asyncio
import unittest

from src.api.dependencies import PaginationParams
from src.api.routes import models as models_routes
from src.models.registry import get_registry, media_label
from src.services.model_service import ModelService

# Facts about the catalog, not about the machine: these entries are marked as
# video-only because selecting them for a still image makes no sense.
VIDEO_ONLY = ("RIFE", "FILM", "ffmpeg", "handbrake", "RealBasicVSR")


class MediaLabelTests(unittest.TestCase):
    def test_known_values_are_phrased_for_the_details_popup(self):
        self.assertEqual(media_label("image"), "Images only")
        self.assertEqual(media_label("video"), "Video only")
        self.assertEqual(media_label("both"), "Images and video")

    def test_unknown_and_missing_values_read_as_both(self):
        for value in (None, "", "gif", "video/images"):
            self.assertEqual(media_label(value), "Images and video", value)

    def test_values_are_normalised_before_being_labelled(self):
        self.assertEqual(media_label(" VIDEO "), "Video only")
        self.assertEqual(media_label("Image"), "Images only")


class CatalogBestForTests(unittest.TestCase):
    """The `bestFor` note the popup shows under "Best used for"."""

    def test_every_entry_states_where_it_is_best_used(self):
        registry = get_registry()
        models = registry.get_all_models()
        self.assertTrue(models)
        for model in models:
            note = (model.best_for or "").strip()
            self.assertTrue(note, f"{model.name} has no bestFor note")
            # A real sentence rather than a placeholder or a stray label.
            self.assertGreater(len(note.split()), 8, f"{model.name}: {note!r}")
            self.assertTrue(note.endswith("."), f"{model.name} is not a sentence: {note!r}")

    def test_notes_are_plain_text(self):
        # Rendered inside a paragraph, so markup or quotes would show up verbatim.
        for model in get_registry().get_all_models():
            self.assertNotIn('"', model.best_for, model.name)
            self.assertNotIn("<", model.best_for, model.name)


class ModelDetailsPayloadTests(unittest.TestCase):
    """Both model routes carry the media and bestFor fields."""

    def setUp(self):
        self.registry = get_registry()
        self.service = ModelService()

    def list_models(self, **kwargs):
        params = {
            "pagination": PaginationParams(page=1, page_size=100),
            "feature": None,
            "downloaded_only": False,
            "compatible_only": False,
            "is_ai": None,
            "model_service": self.service,
            "registry": self.registry,
        }
        params.update(kwargs)
        return asyncio.run(models_routes.list_models(**params))

    def test_list_payload_carries_media_and_best_for(self):
        response = self.list_models()
        self.assertEqual(len(response), len(self.registry.get_all_models()))

        by_name = {model.name: model for model in response}
        real_esrgan = by_name["Real-ESRGAN"]
        self.assertEqual(real_esrgan.media, "both")
        self.assertEqual(real_esrgan.media_label, "Images and video")
        self.assertTrue((real_esrgan.best_for or "").strip())

        for name in VIDEO_ONLY:
            self.assertEqual(by_name[name].media, "video", name)
            self.assertEqual(by_name[name].media_label, "Video only", name)

    def test_every_list_entry_labels_its_media_consistently(self):
        # The label is built server-side from the same value the UI receives, so
        # the two can never disagree.
        for model in self.list_models():
            self.assertEqual(model.media_label, media_label(model.media), model.name)
            self.assertTrue((model.best_for or "").strip(), model.name)

    def test_single_model_payload_carries_the_same_fields(self):
        detail = asyncio.run(models_routes.get_model(model_name="FILM", registry=self.registry))
        self.assertEqual(detail.name, "FILM")
        self.assertEqual(detail.media, "video")
        self.assertEqual(detail.media_label, "Video only")
        self.assertTrue((detail.best_for or "").strip())


if __name__ == "__main__":
    unittest.main()
