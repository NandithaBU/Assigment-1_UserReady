import unittest
from types import SimpleNamespace

from docmeta.backends import ExtractionError, OpenAICompatBackend, parse_json_object
from docmeta.config import Settings, build_backend
from docmeta.models import ImagePart
from docmeta.schema import json_schema


class FakeClient:
    """Mimics openai.OpenAI().chat.completions.create."""

    def __init__(self, reply='{"party_one": "A"}', reject_schema=False):
        self.reply, self.reject_schema, self.calls = reply, reject_schema, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_schema and "response_format" in kwargs:
            raise RuntimeError("response_format not supported")
        msg = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class ParseJsonTests(unittest.TestCase):
    def test_plain_fenced_and_chatty(self):
        self.assertEqual(parse_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(parse_json_object('Sure! Here you go: {"a": {"b": 2}} Hope it helps'),
                         {"a": {"b": 2}})

    def test_failures(self):
        for bad in ["no json here", "{not json}", "", "} {"]:
            with self.assertRaises(ExtractionError):
                parse_json_object(bad)


class OpenAICompatBackendTests(unittest.TestCase):
    def test_text_request_uses_text_model_and_schema(self):
        client = FakeClient()
        backend = OpenAICompatBackend(model="text-m", vision_model="vision-m", client=client)
        out = backend.extract("SYS", "hello", [], json_schema())
        self.assertEqual(out, {"party_one": "A"})
        call = client.calls[0]
        self.assertEqual(call["model"], "text-m")
        self.assertEqual(call["temperature"], 0)
        self.assertEqual(call["response_format"]["type"], "json_schema")
        self.assertIn("agreement_value", call["messages"][0]["content"])
        self.assertEqual(call["messages"][1]["content"], "hello")

    def test_images_use_vision_model_and_data_uri(self):
        client = FakeClient()
        backend = OpenAICompatBackend(model="text-m", vision_model="vision-m", client=client)
        backend.extract("SYS", "read", [ImagePart("image/png", b"\x89PNG")], json_schema())
        call = client.calls[0]
        self.assertEqual(call["model"], "vision-m")
        parts = call["messages"][1]["content"]
        self.assertTrue(parts[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(parts[-1], {"type": "text", "text": "read"})

    def test_falls_back_when_server_rejects_schema(self):
        client = FakeClient(reply='Here: {"party_one": "B"}', reject_schema=True)
        backend = OpenAICompatBackend(model="m", client=client)
        self.assertEqual(backend.extract("S", "t", [], json_schema()), {"party_one": "B"})
        self.assertEqual(len(client.calls), 2)
        self.assertNotIn("response_format", client.calls[1])

    def test_model_default_applies_to_vision_when_set(self):
        b = OpenAICompatBackend(model="only-model", client=FakeClient())
        self.assertEqual(b.vision_model, "only-model")
        d = OpenAICompatBackend(client=FakeClient())
        self.assertNotEqual(d.model, d.vision_model)  # separate text / vision defaults

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            build_backend(Settings(provider="nope"))


if __name__ == "__main__":
    unittest.main()


class FakePipeline:
    def __init__(self, reply, chat_style=True):
        self.reply, self.chat_style, self.calls = reply, chat_style, []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.chat_style:
            return [{"generated_text": messages + [{"role": "assistant", "content": self.reply}]}]
        return [{"generated_text": self.reply}]


class TransformersBackendTests(unittest.TestCase):
    def test_chat_and_plain_output_formats(self):
        from docmeta.backends import TransformersBackend
        for chat_style in (True, False):
            pipe = FakePipeline('Sure: {"party_one": "C"}', chat_style)
            out = TransformersBackend(pipeline=pipe).extract("SYS", "doc", [], json_schema())
            self.assertEqual(out, {"party_one": "C"})
            messages, kwargs = pipe.calls[0]
            self.assertFalse(kwargs["do_sample"])
            self.assertIn("agreement_value", messages[0]["content"])

    def test_images_rejected_with_helpful_message(self):
        from docmeta.backends import TransformersBackend
        backend = TransformersBackend(pipeline=FakePipeline("{}"))
        with self.assertRaises(ExtractionError) as ctx:
            backend.extract("S", "t", [ImagePart("image/png", b"x")], json_schema())
        self.assertIn("--ocr", str(ctx.exception))
