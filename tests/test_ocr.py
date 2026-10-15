import io
import shutil
import unittest
from unittest import mock

from PIL import Image, ImageDraw, ImageFont

from docmeta.models import ImagePart, LoadedDocument
from docmeta.pipeline import MetadataExtractor
from test_pipeline import FakeBackend


def text_png(text: str) -> bytes:
    img = Image.new("RGB", (900, 120), "white")
    ImageDraw.Draw(img).text((20, 30), text, fill="black", font=ImageFont.load_default(size=44))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


class OcrPipelineTests(unittest.TestCase):
    def test_ocr_replaces_images_with_text(self):
        backend = FakeBackend()
        doc = LoadedDocument("scan", images=[ImagePart("image/png", b"x")])
        with mock.patch("docmeta.pipeline.ocr_images", return_value="RENT 6500"):
            MetadataExtractor(backend, ocr=True).extract_document(doc)
        _, text, images, _ = backend.calls[0]
        self.assertIn("RENT 6500", text)
        self.assertEqual(images, [])

    def test_without_ocr_images_pass_through(self):
        backend = FakeBackend()
        doc = LoadedDocument("scan", images=[ImagePart("image/png", b"x")])
        MetadataExtractor(backend).extract_document(doc)
        self.assertEqual(len(backend.calls[0][2]), 1)

    @unittest.skipUnless(shutil.which("tesseract"), "tesseract not installed")
    def test_real_tesseract(self):
        from docmeta.readers import ocr_images
        out = ocr_images([ImagePart("image/png", text_png("Monthly rent 6500"))])
        self.assertIn("rent", out.lower())
        self.assertIn("6500", out)


if __name__ == "__main__":
    unittest.main()
