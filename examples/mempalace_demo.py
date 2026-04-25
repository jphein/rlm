"""RLM + MemPalace demo: verbatim memory archive as an RLM REPL tool.

Demonstrates the composition pattern proposed in the fork README:
RLM's recursive REPL gets a `mempalace_search` tool, so the model can
*decide when* to retrieve from a verbatim memory archive instead of a
hand-coded retrieve-then-generate pipeline.

Backend: Azure AI Foundry (gpt-5.3-chat). Other RLM-supported backends
(openai, anthropic, gemini, vertex via openai-compat) work with no
changes other than `backend=...` and `backend_kwargs`.

Memory backend: palace-daemon HTTP gateway (see
https://github.com/rboarescu/palace-daemon and
https://github.com/jphein/palace-daemon).

To run:
    pip install rlms openai
    PALACE_DAEMON_URL=http://your-daemon:8085 \\
    PALACE_API_KEY=... \\
    AZURE_API_KEY=... \\
    AZURE_ENDPOINT=https://your-foundry.services.ai.azure.com \\
    AZURE_DEPLOYMENT=gpt-5.3-chat \\
        python examples/mempalace_demo.py "what did I decide about X?"
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request

from rlm import RLM
from rlm.logger import RLMLogger


# ──────────────── tool: mempalace_search ─────────────────────────────
DAEMON_URL = os.environ.get("PALACE_DAEMON_URL", "http://localhost:8085")
DAEMON_KEY = os.environ.get("PALACE_API_KEY", "")


def mempalace_search(
    query: str, limit: int = 5, wing: str | None = None
) -> list[dict]:
    """Search the verbatim memory archive.

    Calls palace-daemon's /search endpoint. Returns up to `limit` hits as a
    list of dicts: {text, wing, room, similarity, created_at}. The
    similarity score is in [0,1] (1 = identical); created_at is ISO-8601.

    Args:
        query: natural-language search query
        limit: max results (1-10 sensible)
        wing: optional wing-name filter (e.g. "wing_projects")
    """
    params = f"?q={urllib.parse.quote(query)}&limit={limit}"
    if wing:
        params += f"&wing={urllib.parse.quote(wing)}"
    req = urllib.request.Request(
        f"{DAEMON_URL}/search{params}",
        headers={"x-api-key": DAEMON_KEY} if DAEMON_KEY else {},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    return [
        {
            "text": (h.get("text") or "")[:600],
            "wing": h.get("wing"),
            "room": h.get("room"),
            "similarity": h.get("similarity"),
            "created_at": h.get("created_at"),
        }
        for h in d.get("results", [])
    ]


# ──────────────── set up RLM ────────────────────────────────────────
def make_rlm() -> RLM:
    api_key = os.environ.get("AZURE_API_KEY") or os.environ["OPENAI_API_KEY"]
    endpoint = os.environ.get("AZURE_ENDPOINT")
    deployment = os.environ.get("AZURE_DEPLOYMENT", "gpt-5.3-chat")

    if endpoint:
        backend = "azure_openai"
        backend_kwargs = {
            "api_key": api_key,
            "azure_endpoint": endpoint,
            "api_version": "2024-12-01-preview",
            "azure_deployment": deployment,
            "model_name": deployment,
        }
    else:
        backend = "openai"
        backend_kwargs = {"api_key": api_key, "model_name": deployment}

    return RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        verbose=True,
        logger=RLMLogger(log_dir="./logs"),
        # Inject the search tool into the REPL globals so the model can
        # call it directly inside the recursive REPL.
        custom_tools={"mempalace_search": mempalace_search},
    )


# ──────────────── entrypoint ────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} \"<question>\"", file=sys.stderr)
        sys.exit(2)
    question = sys.argv[1]

    prompt = (
        "You have a tool `mempalace_search(query, limit=5, wing=None)` available "
        "in your REPL — it returns a list of {text, wing, room, similarity, "
        "created_at} dicts from a verbatim memory archive. Use it to ground your "
        "answer to the following question. Cite drawer wings/rooms/dates when "
        "you do.\n\n"
        f"Question: {question}"
    )
    result = make_rlm().completion(prompt)
    print()
    print("=" * 78)
    print(result.response)


if __name__ == "__main__":
    main()
