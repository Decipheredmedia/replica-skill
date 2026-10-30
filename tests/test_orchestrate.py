import json
import os
import tempfile
import unittest

from _load import ROOT, load

orchestrate = load("ollama-orchestrator", "orchestrate")


class BuildPrompt(unittest.TestCase):
    def test_includes_brief_and_prior_outputs(self):
        system, user = orchestrate.build_prompt("# recon skill", "a scheduling app",
                                                  {"recon": "18 screens found"})
        self.assertIn("recon skill", system)
        self.assertIn("a scheduling app", user)
        self.assertIn("18 screens found", user)
        self.assertIn("## Output of the recon step", user)

    def test_no_brief_still_produces_a_user_message(self):
        _, user = orchestrate.build_prompt("# skill", "", {})
        self.assertIn("infer from context", user)


class ReadSkill(unittest.TestCase):
    def test_reads_real_skill_md(self):
        text = orchestrate.read_skill("replica-recon", skills_root=ROOT)
        self.assertIn("name: replica-recon", text)


class RunPipeline(unittest.TestCase):
    def fake_chat_factory(self, calls):
        def chat(model, system, user):
            calls.append((model, system[:20]))
            return "output for %s" % model
        return chat

    def test_dry_run_writes_every_step_without_calling_chat(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            manifest = orchestrate.run_pipeline(
                tmp, app_brief="a note app", chat=self.fake_chat_factory(calls),
                dry_run=True, skills_root=ROOT)
            self.assertEqual(calls, [])
            self.assertEqual(len(manifest["tasks"]), len(orchestrate.STEPS))
            for role, _, out_name in orchestrate.STEPS:
                self.assertTrue(os.path.exists(os.path.join(tmp, "replica", out_name)))

    def test_resume_skips_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "replica"))
            with open(os.path.join(tmp, "replica", "recon.md"), "w") as fh:
                fh.write("already done\n")
            calls = []
            manifest = orchestrate.run_pipeline(
                tmp, chat=self.fake_chat_factory(calls), only=["recon"],
                skills_root=ROOT)
            recon_task = next(t for t in manifest["tasks"] if t["id"] == "recon")
            self.assertEqual(recon_task["status"], "skipped")
            self.assertEqual(calls, [])

    def test_only_restricts_which_steps_run(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            manifest = orchestrate.run_pipeline(
                tmp, chat=self.fake_chat_factory(calls), only=["recon"],
                skills_root=ROOT)
            statuses = {t["id"]: t["status"] for t in manifest["tasks"]}
            self.assertEqual(statuses["recon"], "generated")
            self.assertEqual(statuses["architect"], "not_selected")
            self.assertEqual(len(calls), 1)

    def test_models_assignment_is_used_per_role(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            orchestrate.run_pipeline(
                tmp, models={"recon": "llava:7b"}, chat=self.fake_chat_factory(calls),
                only=["recon"], skills_root=ROOT)
            self.assertEqual(calls[0][0], "llava:7b")

    def test_manifest_written_to_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            orchestrate.run_pipeline(tmp, dry_run=True, skills_root=ROOT)
            manifest_path = os.path.join(tmp, "replica", "orchestrate-manifest.json")
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path) as fh:
                data = json.load(fh)
            self.assertEqual(len(data["tasks"]), len(orchestrate.STEPS))


if __name__ == "__main__":
    unittest.main()
