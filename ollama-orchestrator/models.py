#!/usr/bin/env python3
"""Detect Ollama models already installed and assign the best one to each
replica-skill role. Standard library only.

Replaces guessing a model name: it asks a running Ollama for its own
`/api/tags` list, classifies every model by what it is good at (code,
vision, reasoning, general) and by size, then picks the strongest fit for
each of the eleven replica-skill steps. Models can and will be reused across
roles on a VPS that only has one or two pulled.

    python3 models.py                          # detect, assign, print, write replica/models.json
    python3 models.py --host http://10.0.0.5:11434
    python3 models.py --input tags.json         # offline: a saved /api/tags response
    python3 models.py --no-write
    python3 models.py --json                    # machine-readable, no table

Exits 1 (with the exact `ollama pull ...` commands to run) if no usable
models are installed, so a setup script can act on it.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

ROLES = ["recon", "architect", "design", "build", "backend", "test", "diff",
          "entrepreneur", "brand", "launch", "deploy"]

# What each step in the replica-skill pipeline leans on most. "prefer" is
# checked in order; the first tag a model has wins its bonus. "min_b" is the
# softest of floors, a model under it is still used if nothing bigger exists.
ROLE_SPECS = {
    "recon":        {"prefer": ("vision", "general"), "min_b": 7},
    "architect":    {"prefer": ("reasoning", "general"), "min_b": 7},
    "design":       {"prefer": ("general",), "min_b": 3},
    "build":        {"prefer": ("code",), "min_b": 7},
    "backend":      {"prefer": ("code",), "min_b": 7},
    "test":         {"prefer": ("code", "general"), "min_b": 3},
    "diff":         {"prefer": ("vision", "general"), "min_b": 3},
    "entrepreneur": {"prefer": ("general",), "min_b": 3},
    "brand":        {"prefer": ("general",), "min_b": 3},
    "launch":       {"prefer": ("general",), "min_b": 3},
    "deploy":       {"prefer": ("code", "general"), "min_b": 3},
}

CODE_HINTS = ("coder", "code", "codellama", "starcoder", "codestral", "granite-code",
              "devstral", "codegemma")
VISION_HINTS = ("vision", "llava", "-vl", ":vl", "bakllava", "moondream", "pixtral")
REASONING_HINTS = ("r1", "qwq", "reasoner", "o1", "think")
EMBED_HINTS = ("embed", "bge-", "nomic-embed", "minilm", "e5-")

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
RECOMMENDED_PULLS = ["llama3.1:8b", "qwen2.5-coder:7b", "llava:7b"]

SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[xX]?\s*(\d+(?:\.\d+)?)?\s*[bB]\b")


class ModelDetectionError(RuntimeError):
    pass


def parse_param_size(text):
    """"7B" -> 7.0, "8x7B" (a mixture of experts) -> 56.0, "" -> 0.0."""
    if not text:
        return 0.0
    m = SIZE_RE.search(str(text))
    if not m:
        return 0.0
    a, b = m.groups()
    a = float(a)
    return a * float(b) if b else a


def classify(name, parameter_size=""):
    """Return (tags: set, size_b: float) for a model name like 'qwen2.5-coder:7b'."""
    low = name.lower()
    size_b = parse_param_size(parameter_size)
    if not size_b:
        tag_m = re.search(r":(\d+(?:\.\d+)?)\s*x?\s*(\d+(?:\.\d+)?)?b", low)
        if tag_m:
            a, b = tag_m.groups()
            size_b = float(a) * float(b) if b else float(a)
    tags = set()
    if any(h in low for h in EMBED_HINTS):
        tags.add("embedding")
        return tags, size_b
    if any(h in low for h in CODE_HINTS):
        tags.add("code")
    if any(h in low for h in VISION_HINTS):
        tags.add("vision")
    if any(h in low for h in REASONING_HINTS):
        tags.add("reasoning")
    tags.add("general")
    return tags, size_b


def normalize(raw_models):
    """Turn a raw /api/tags (or `ollama list`) response into plain dicts,
    dropping embedding-only models (they cannot follow the skills' prompts)."""
    out = []
    for m in raw_models:
        name = m.get("name") or m.get("model") or ""
        if not name:
            continue
        details = m.get("details") or {}
        param_size = details.get("parameter_size", "")
        tags, size_b = classify(name, param_size)
        if "embedding" in tags:
            continue
        out.append({
            "name": name,
            "family": details.get("family", ""),
            "parameter_size": param_size or ("%sB" % size_b if size_b else ""),
            "size_b": size_b,
            "tags": sorted(tags),
        })
    out.sort(key=lambda m: (-m["size_b"], m["name"]))
    return out


def fetch_tags_http(host, timeout=5, opener=None):
    url = host.rstrip("/") + "/api/tags"
    request = urllib.request.urlopen if opener is None else opener
    with request(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_tags_cli():
    """Fall back to `ollama list` when the HTTP API is not reachable (same
    host, no network loopback, running inside a minimal container, etc.)."""
    proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
    if proc.returncode != 0:
        raise ModelDetectionError(proc.stderr.strip() or "ollama list failed")
    models = []
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append({"name": parts[0]})
    return {"models": models}


def detect(host=DEFAULT_HOST, timeout=5, opener=None):
    """Return the normalized model list, trying the HTTP API first."""
    try:
        data = fetch_tags_http(host, timeout=timeout, opener=opener)
    except (urllib.error.URLError, OSError, ValueError):
        try:
            data = fetch_tags_cli()
        except (OSError, subprocess.SubprocessError, ModelDetectionError) as exc:
            raise ModelDetectionError(
                "Could not reach Ollama at %s and `ollama list` failed (%s). "
                "Install it with https://ollama.com/download or start it with "
                "`ollama serve`." % (host, exc)) from exc
    return normalize(data.get("models", []))


def score(model, spec):
    s = model["size_b"]
    for i, tag in enumerate(spec["prefer"]):
        if tag in model["tags"]:
            s += (len(spec["prefer"]) - i) * 50
            break
    if model["size_b"] and model["size_b"] < spec["min_b"]:
        s -= 10
    return s


def assign(models, specs=None):
    """Pick the best installed model for every role. Models are reused across
    roles; nothing is held back for a role it is not needed for."""
    specs = specs or ROLE_SPECS
    if not models:
        return {role: None for role in specs}
    return {role: max(models, key=lambda m: score(m, spec))["name"]
            for role, spec in specs.items()}


def to_report(models, assignment):
    return {
        "host": DEFAULT_HOST,
        "installed": models,
        "assignment": assignment,
    }


def print_table(models, assignment):
    if not models:
        print("No usable Ollama models found.")
        print("Pull a starter set that covers every role:")
        for name in RECOMMENDED_PULLS:
            print("  ollama pull %s" % name)
        return
    print("Installed models:")
    for m in models:
        tags = ",".join(m["tags"])
        print("  %-28s %-8s %s" % (m["name"], m["parameter_size"] or "?", tags))
    print("\nRole assignment:")
    for role in ROLES:
        print("  %-14s -> %s" % (role, assignment.get(role) or "(none installed)"))


def write_report(root, report):
    out_dir = os.path.join(root, "replica")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "models.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=".",
                         help="project root to write replica/models.json into")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--input", help="offline: a saved /api/tags JSON response")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    if args.input:
        with open(args.input, encoding="utf-8") as fh:
            models = normalize(json.load(fh).get("models", []))
    else:
        try:
            models = detect(host=args.host)
        except ModelDetectionError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    assignment = assign(models)
    report = to_report(models, assignment)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_table(models, assignment)

    if not args.no_write:
        path = write_report(args.root, report)
        if not args.json:
            print("\nWrote %s" % path)

    return 0 if models else 1


if __name__ == "__main__":
    sys.exit(main())
