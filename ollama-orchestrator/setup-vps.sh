#!/usr/bin/env bash
# Bootstrap an Ubuntu VPS to run the replica-skill pipeline fully locally
# with Ollama, alongside cairn and hermes-agents. Idempotent: safe to re-run.
#
#   curl -fsSL .../setup-vps.sh | bash
#   ./setup-vps.sh --auto-pull        # also pulls a starter model set if none exist
#
set -euo pipefail

AUTO_PULL=0
for arg in "$@"; do
  case "$arg" in
    --auto-pull) AUTO_PULL=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

STARTER_MODELS=("llama3.1:8b" "qwen2.5-coder:7b" "llava:7b")

if [ "$(uname -s)" != "Linux" ]; then
  echo "This script targets Ubuntu. Detected: $(uname -s)." >&2
fi

echo "== Python 3 =="
if ! command -v python3 >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y python3
fi
python3 --version

echo "== Ollama =="
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "already installed: $(ollama --version 2>/dev/null || true)"
fi

if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl enable --now ollama || true
else
  pgrep -x ollama >/dev/null 2>&1 || nohup ollama serve >/tmp/ollama.log 2>&1 &
fi

echo "== Detecting installed models =="
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sleep 2
if ! python3 "$HERE/models.py" --no-write; then
  if [ "$AUTO_PULL" -eq 1 ]; then
    echo "No usable models found. Pulling the starter set..."
    for m in "${STARTER_MODELS[@]}"; do
      ollama pull "$m"
    done
    python3 "$HERE/models.py" --no-write
  else
    echo "Re-run with --auto-pull to pull a starter set automatically, or run:"
    for m in "${STARTER_MODELS[@]}"; do
      echo "  ollama pull $m"
    done
  fi
fi

python3 "$HERE/models.py"

cat <<'EOF'

== Next steps ==

Run the pipeline against the models you have installed:

  python3 ollama-orchestrator/orchestrate.py --app "https://example.com, what it does" /path/to/your/clone-repo

See ollama-orchestrator/README.md for how to wire this up with:
  - cairn      (https://github.com/cairn-dev/cairn)       -- background task runner / PR flow
  - hermes-agents (https://github.com/Decipheredmedia/hermes-agents) -- loads SKILL.md folders directly
EOF
