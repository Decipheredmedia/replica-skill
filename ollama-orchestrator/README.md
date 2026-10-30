# ollama-orchestrator

Runs the eleven replica-skill prompts end to end against models you already
have pulled in [Ollama](https://ollama.com), on your own Ubuntu VPS, with no
Claude, no API key and no per-token bill. It detects what is installed,
scores each model on what it is good at (code, vision, reasoning, general
instruction-following) and size, and assigns the strongest fit to every step
of the pipeline, reusing models across roles when the box only has one or
two pulled.

```
recon -> architect -> design -> build -> backend -> test -> diff -> entrepreneur -> brand -> launch -> deploy
```

This folder is tooling, not a twelfth skill: the SKILL.md files it reads are
still `replica-recon/SKILL.md` and so on, unchanged.

## Quick start

```bash
# One-time, on a fresh Ubuntu VPS: installs Ollama if missing, starts it,
# detects what is installed (or pulls a small starter set with --auto-pull).
./ollama-orchestrator/setup-vps.sh --auto-pull

# Detect installed models, score them, assign one to every role, write
# replica/models.json:
python3 ollama-orchestrator/models.py /path/to/your/clone-repo

# Run the full pipeline (resumable: steps with an existing output file are
# skipped, so a killed run or a model swap partway through just continues):
python3 ollama-orchestrator/orchestrate.py --app "https://example.com, a scheduling app" \
    /path/to/your/clone-repo
```

Output lands in the target repo's `replica/` folder exactly like the Claude
version: `recon.md`, `architecture.md`, `design.md`, and so on, plus
`replica/models.json` (what was detected and assigned) and
`replica/orchestrate-manifest.json` (what ran, on which model, and where it
wrote to).

## Detecting and assigning models

`models.py` calls Ollama's own `/api/tags` (falling back to `ollama list` if
the HTTP API is not reachable), classifies each model by name:

| tag | matched by names containing |
| --- | --- |
| `code` | `coder`, `codellama`, `starcoder`, `codestral`, `granite-code`, `devstral`, `codegemma` |
| `vision` | `vision`, `llava`, `-vl`, `bakllava`, `moondream`, `pixtral` |
| `reasoning` | `r1`, `qwq`, `reasoner`, `o1`, `think` |
| `general` | everything else that is not an embedding model |

Embedding-only models (`embed`, `bge-`, `nomic-embed`, `minilm`, `e5-`) are
dropped; they cannot follow a skill's prompt. Each role then scores every
remaining model on size plus a bonus for matching its preferred tag (`build`
and `backend` prefer `code`, `recon` and `diff` prefer `vision`, `architect`
prefers `reasoning`, the rest prefer `general`) and the highest score wins.
Nothing is reserved: a single installed model covers every role if that is
all the box has.

If nothing usable is installed, `models.py` exits 1 and prints the pull
commands for a small starter set that covers every role:

```
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
ollama pull llava:7b
```

To override an assignment by hand, copy `config.example.json` to
`replica/models.json` and edit the `assignment` map; `orchestrate.py` reads
it as-is.

## Working with cairn

[cairn](https://github.com/cairn-dev/cairn) is a background task runner: it
turns a task into a branch or PR using its SWE/PM/Fullstack Planner agents.
As of this writing cairn's model list
(`cairn_utils/supported_models.py`) is Anthropic, OpenAI and Gemini only, it
has no local-model or custom-base-URL provider, so it cannot call Ollama
directly. The two still combine cleanly:

1. Run `orchestrate.py` locally (or on the VPS) to produce the `replica/`
   plan, build and tests fully offline against your Ollama models.
2. Commit the result to a branch and open a cairn task against that branch
   asking its SWE/PM agents to review, wire up secrets that need a human
   (Stripe keys, OAuth apps, the domain), or fix anything the local run got
   wrong. cairn's hosted-model agents are good at exactly that kind of
   judgement call; the bulk generation is what you want off the API bill.
3. cairn's `.cairn/settings.json` repo rules are a good place to pin "read
   `replica/*.md` before changing anything", so its agents stay consistent
   with what the orchestrator already decided.

## Working with hermes-agents

[hermes-agents](https://github.com/Decipheredmedia/hermes-agents) is a
general-purpose agent that is explicitly model-agnostic (`hermes model` to
switch) and compatible with the [agentskills.io](https://agentskills.io)
`SKILL.md` standard that this repo already uses. That makes it the more
direct fit:

```bash
# Point hermes at your local Ollama instead of a cloud provider, then let it
# load this repo's skills directly, no copying needed if it is run from here:
hermes model set ollama/llama3.1:8b   # or your own endpoint, per hermes-agents' docs
hermes skills add ./replica-recon ./replica-architect ./replica-design \
    ./replica-build ./replica-backend ./replica-test ./replica-diff \
    ./replica-entrepreneur ./replica-brand ./replica-launch ./replica-deploy
```

Because hermes-agents persists memory and can run unattended on a VPS, it is
a reasonable way to drive the pipeline conversationally ("run replica-recon
on https://example.com, then replica-architect") while still falling back to
`orchestrate.py` for a fully scripted, cron-able run. Check hermes-agents'
own docs for the exact model-provider flag; it changes across releases.

## Notes

- Everything here is standard-library Python 3 and bash, no extra installs
  beyond Ollama itself.
- Vision models only help `recon` and `diff` if you actually feed them
  screenshots; text-only recon and diff work fine on a `general` model too.
- `orchestrate.py --dry-run` checks the whole pipeline's wiring (file paths,
  resume logic, manifest) without calling a model, useful before a long run
  on a fresh VPS.
