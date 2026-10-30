import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from _load import load

models = load("ollama-orchestrator", "models")


def tags_response(names_and_sizes):
    return {"models": [
        {"name": name, "details": {"parameter_size": size}}
        for name, size in names_and_sizes
    ]}


class ParseParamSize(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(models.parse_param_size("7B"), 7.0)
        self.assertEqual(models.parse_param_size("70B"), 70.0)

    def test_moe(self):
        self.assertEqual(models.parse_param_size("8x7B"), 56.0)

    def test_empty(self):
        self.assertEqual(models.parse_param_size(""), 0.0)
        self.assertEqual(models.parse_param_size(None), 0.0)


class Classify(unittest.TestCase):
    def test_code_model(self):
        tags, size = models.classify("qwen2.5-coder:7b", "7B")
        self.assertIn("code", tags)
        self.assertEqual(size, 7.0)

    def test_vision_model(self):
        tags, _ = models.classify("llava:13b", "13B")
        self.assertIn("vision", tags)

    def test_reasoning_model(self):
        tags, _ = models.classify("deepseek-r1:14b", "14B")
        self.assertIn("reasoning", tags)

    def test_embedding_model_is_tagged_and_excluded_downstream(self):
        tags, _ = models.classify("nomic-embed-text", "")
        self.assertIn("embedding", tags)

    def test_plain_model_is_general(self):
        tags, _ = models.classify("llama3.1:8b", "8B")
        self.assertEqual(tags, {"general"})

    def test_size_from_tag_when_details_missing(self):
        _, size = models.classify("llama3.1:70b", "")
        self.assertEqual(size, 70.0)


class Normalize(unittest.TestCase):
    def test_drops_embeddings_and_sorts_by_size(self):
        raw = tags_response([
            ("nomic-embed-text", ""),
            ("llama3.1:8b", "8B"),
            ("llama3.1:70b", "70B"),
        ])["models"]
        out = models.normalize(raw)
        self.assertEqual([m["name"] for m in out], ["llama3.1:70b", "llama3.1:8b"])

    def test_skips_entries_without_a_name(self):
        out = models.normalize([{"details": {}}])
        self.assertEqual(out, [])


class Assign(unittest.TestCase):
    def test_single_model_covers_every_role(self):
        raw = tags_response([("llama3.1:8b", "8B")])["models"]
        installed = models.normalize(raw)
        assignment = models.assign(installed)
        self.assertTrue(all(v == "llama3.1:8b" for v in assignment.values()))
        self.assertEqual(set(assignment), set(models.ROLE_SPECS))

    def test_code_model_wins_build_and_backend(self):
        raw = tags_response([
            ("llama3.1:8b", "8B"),
            ("qwen2.5-coder:7b", "7B"),
        ])["models"]
        installed = models.normalize(raw)
        assignment = models.assign(installed)
        self.assertEqual(assignment["build"], "qwen2.5-coder:7b")
        self.assertEqual(assignment["backend"], "qwen2.5-coder:7b")

    def test_vision_model_wins_recon_and_diff(self):
        raw = tags_response([
            ("llama3.1:8b", "8B"),
            ("llava:7b", "7B"),
        ])["models"]
        installed = models.normalize(raw)
        assignment = models.assign(installed)
        self.assertEqual(assignment["recon"], "llava:7b")
        self.assertEqual(assignment["diff"], "llava:7b")

    def test_bigger_general_model_wins_architect(self):
        raw = tags_response([
            ("llama3.1:8b", "8B"),
            ("llama3.1:70b", "70B"),
        ])["models"]
        installed = models.normalize(raw)
        assignment = models.assign(installed)
        self.assertEqual(assignment["architect"], "llama3.1:70b")

    def test_no_models_gives_none_for_every_role(self):
        assignment = models.assign([])
        self.assertTrue(all(v is None for v in assignment.values()))


class Detect(unittest.TestCase):
    def test_http_success(self):
        payload = json.dumps(tags_response([("llama3.1:8b", "8B")])).encode("utf-8")

        class FakeResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(url, timeout=5):
            return FakeResp(payload)

        found = models.detect(opener=opener)
        self.assertEqual(found[0]["name"], "llama3.1:8b")

    def test_http_failure_raises_helpful_error(self):
        import urllib.error

        def opener(url, timeout=5):
            raise urllib.error.URLError("refused")

        with self.assertRaises(models.ModelDetectionError):
            models.detect(opener=opener)


class Cli(unittest.TestCase):
    def test_main_writes_report_from_input_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tags_path = os.path.join(tmp, "tags.json")
            with open(tags_path, "w", encoding="utf-8") as fh:
                json.dump(tags_response([("llama3.1:8b", "8B")]), fh)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = models.main([tmp, "--input", tags_path])
            self.assertEqual(rc, 0)
            out_path = os.path.join(tmp, "replica", "models.json")
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, encoding="utf-8") as fh:
                report = json.load(fh)
            self.assertEqual(report["assignment"]["build"], "llama3.1:8b")

    def test_main_exits_1_with_no_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            tags_path = os.path.join(tmp, "tags.json")
            with open(tags_path, "w", encoding="utf-8") as fh:
                json.dump({"models": []}, fh)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = models.main([tmp, "--input", tags_path, "--no-write"])
            self.assertEqual(rc, 1)
            self.assertIn("ollama pull", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
