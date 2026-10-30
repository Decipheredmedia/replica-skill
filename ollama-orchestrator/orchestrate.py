#!/usr/bin/env python3
"""Run the replica-skill pipeline end to end against local Ollama models
instead of Claude. Standard library only.

Reads each skill's `SKILL.md` as its system prompt, in the recon -> architect
-> design -> build -> backend -> test -> diff -> entrepreneur -> brand ->
launch -> deploy order, with the app brief and every prior step's output as
context. Writes one file per step into `replica/`, and a manifest
(`replica/orchestrate-manifest.json`) that a task runner such as cairn or
hermes-agents can read to see what ran, on what model, and where the output
went.

    python3 orchestrate.py --app "https://example.com, a scheduling app" .
    python3 orchestrate.py --only recon,architect .
    python3 orchestrate.py --dry-run .             # wiring check, no model calls

Models come from `replica/models.json` (written by `models.py`) unless
`--models` points elsewhere; missing roles fall back to `--default-model`.
"""

import argparse
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)

# (role, skill folder, output file). Order is the pipeline order.
STEPS = [
    ("recon", "replica-recon", "recon.md"),
    ("architect", "replica-architect", "architecture.md"),
    ("design", "replica-design", "design.md"),
    ("build", "replica-build", "build.md"),
    ("backend", "replica-backend", "backend.md"),
    ("test", "replica-test", "test-plan.md"),
    ("diff", "replica-diff", "diff.md"),
    ("entrepreneur", "replica-entrepreneur", "entrepreneur.md"),
    ("brand", "replica-brand", "brand.md"),
    ("launch", "replica-launch", "launch.md"),
    ("deploy", "replica-deploy", "deploy.md"),
]

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "llama3.1:8b"


def read_skill(skill_folder, skills_root=REPO_ROOT):
    path = os.path.join(skills_root, skill_folder, "SKILL.md")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def load_models(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("assignment", {})


def build_prompt(skill_md, app_brief, prior_outputs):
    """system, user for one step. `prior_outputs` is {role: text}, in
    pipeline order, so each step sees everything written before it."""
    system = (
        skill_md + "\n\n---\n\nYou are running non-interactively, with no human to "
        "ask follow-up questions. Make the most reasonable assumption, state it, "
        "and continue. Write only the file this skill produces, as Markdown."
    )
    parts = ["App to clone: %s" % (app_brief or "(no brief given, infer from context)")]
    for role, text in prior_outputs.items():
        parts.append("\n## Output of the %s step\n\n%s" % (role, text))
    return system, "\n".join(parts)


def ollama_chat(model, system, user, host=DEFAULT_HOST, timeout=600, opener=None):
    """Call Ollama's /api/chat (non-streaming) and return the reply text."""
    body = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    request = urllib.request.urlopen if opener is None else opener
    with request(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("content", "")


def run_pipeline(root, app_brief="", models=None, steps=None, chat=None,
                  resume=True, only=None, skills_root=REPO_ROOT, dry_run=False,
                  default_model=DEFAULT_MODEL):
    steps = steps or STEPS
    models = models or {}
    chat = chat or ollama_chat
    only = set(only) if only else None

    out_dir = os.path.join(root, "replica")
    os.makedirs(out_dir, exist_ok=True)

    prior_outputs = {}
    results = []
    for role, skill_folder, out_name in steps:
        out_path = os.path.join(out_dir, out_name)
        if only and role not in only:
            if os.path.exists(out_path):
                with open(out_path, encoding="utf-8") as fh:
                    prior_outputs[role] = fh.read()
            results.append({"role": role, "status": "not_selected", "output": out_path})
            continue

        model = models.get(role) or default_model

        if resume and os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as fh:
                content = fh.read()
            prior_outputs[role] = content
            results.append({"role": role, "status": "skipped", "model": model,
                             "output": out_path})
            continue

        if dry_run:
            content = "(dry run: %s would run here on %s)" % (role, model)
        else:
            skill_md = read_skill(skill_folder, skills_root=skills_root)
            system, user = build_prompt(skill_md, app_brief, prior_outputs)
            content = chat(model, system, user)

        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(content if content.endswith("\n") else content + "\n")
        prior_outputs[role] = content
        results.append({"role": role, "status": "generated", "model": model,
                         "output": out_path})

    manifest = {
        "app": app_brief,
        "steps": results,
        "tasks": [
            {
                "id": r["role"],
                "title": "replica-%s" % r["role"],
                "status": r["status"],
                "model": r.get("model"),
                "output": os.path.relpath(r["output"], root),
            }
            for r in results
        ],
    }
    manifest_path = os.path.join(out_dir, "orchestrate-manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=".",
                         help="project root; reads and writes its replica/ folder")
    parser.add_argument("--app", default="", help="what to clone: a URL and/or brief")
    parser.add_argument("--models", help="path to models.json (default: <root>/replica/models.json)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--default-model", default=DEFAULT_MODEL)
    parser.add_argument("--only", help="comma-separated roles to (re)run, e.g. recon,architect")
    parser.add_argument("--no-resume", action="store_true",
                         help="regenerate steps even if their output file exists")
    parser.add_argument("--dry-run", action="store_true",
                         help="check the wiring without calling a model")
    args = parser.parse_args(argv)

    models_path = args.models or os.path.join(args.root, "replica", "models.json")
    models = load_models(models_path)
    only = [r.strip() for r in args.only.split(",")] if args.only else None

    def chat(model, system, user):
        return ollama_chat(model, system, user, host=args.host)

    manifest = run_pipeline(
        args.root, app_brief=args.app, models=models, chat=chat,
        resume=not args.no_resume, only=only, dry_run=args.dry_run,
        default_model=args.default_model)

    for task in manifest["tasks"]:
        print("%-14s %-10s %s" % (task["id"], task["status"], task["output"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
