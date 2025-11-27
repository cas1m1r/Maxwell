#!/usr/bin/env python3
"""
Maxwell Observatory - Capsule Graph Viewer
Run:
    export CAPSULE_DIR=/path/to/your/capsule/folder   # optional
    python app.py

Then open: http://localhost:1234
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from flask import Flask, jsonify, send_from_directory, abort, request

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

# Instead of CAPSULE_DIR as a single folder, we use a ROOT and runs inside it
CAPSULE_ROOT = Path(os.getenv("CAPSULE_ROOT", ".")).resolve()

import threading

search_lock = threading.Lock()

app = Flask(__name__, static_folder="static", static_url_path="/static")


# -------------------------------------------------------------------
# Capsule/Run Loading
# -------------------------------------------------------------------


def list_runs() -> List[Dict[str, str]]:
    """
    Look at CAPSULE_ROOT and treat each subdirectory that has at least
    one .json file as a 'run'.

    Also allow the root itself as a pseudo-run called "__all__".
    """
    runs = []

    # Root as special run (all capsules)
    runs.append({"id": "__all__", "path": str(CAPSULE_ROOT), "label": "All capsules"})

    for entry in CAPSULE_ROOT.iterdir():
        if entry.is_dir():
            has_json = any(entry.glob("*.json"))
            if has_json:
                runs.append({"id": entry.name, "path": str(entry.resolve()), "label": entry.name})

    return runs


def load_capsules(run_id: str = "__all__") -> Dict[str, Dict[str, Any]]:
    """
    Load capsules for a given run.

    - run_id == "__all__": scan CAPSULE_ROOT for all *.json (non-recursive)
    - otherwise: scan CAPSULE_ROOT / run_id for *.json
    """
    capsules: Dict[str, Dict[str, Any]] = {}

    if run_id == "__all__":
        search_dir = CAPSULE_ROOT
    else:
        search_dir = CAPSULE_ROOT / run_id

    if not search_dir.exists() or not search_dir.is_dir():
        return capsules

    for path in search_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue

        capsule_id = path.stem
        data["_capsule_id"] = capsule_id
        data["_filename"] = str(path.name)
        data["_run_id"] = run_id
        capsules[capsule_id] = data

    return capsules


# -------------------------------------------------------------------
# Graph construction
# -------------------------------------------------------------------

def build_global_graph(capsules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a merged graph across all capsules.

    Node types:
      - "capsule"   (one per file)
      - "concept"   (merged across capsules by label)
      - "issue"     (merged across capsules by label)

    Edge types:
      - "capsule_concept" (capsule -> concept membership)
      - "concept_concept" (co-occurs / conceptual edges inside capsules)
      - "concept_issue"   (concept -> issue relations)
    """
    # node_id -> node dict
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def add_node(node_id: str, label: str, ntype: str) -> Dict[str, Any]:
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "type": ntype,
                "capsules": set(),  # will convert to list later
            }
        return nodes[node_id]

    # Process each capsule
    for cid, cap in capsules.items():
        seed = cap.get("seed", cid)
        concepts = cap.get("concepts", {}) or {}
        issues = cap.get("issues", {}) or {}
        edges_local = cap.get("edges", []) or {}

        # Capsule node
        capsule_node_id = f"capsule:{cid}"
        capsule_node = add_node(capsule_node_id, seed, "capsule")
        capsule_node["capsules"].add(cid)

        # Concept nodes
        concept_id_map: Dict[str, str] = {}  # local concept-id -> global node-id
        for local_id, c in concepts.items():
            label = c.get("label", local_id)
            # Global concept key by label (normalized)
            key = f"concept:{label.strip()}"
            n = add_node(key, label, "concept")
            n["capsules"].add(cid)
            concept_id_map[local_id] = key

            # Capsule -> concept edge
            edges.append({
                "source": capsule_node_id,
                "target": key,
                "type": "capsule_concept",
            })

        # Issue nodes
        issue_id_map: Dict[str, str] = {}
        for local_id, i in issues.items():
            label = i.get("label", local_id)
            key = f"issue:{label.strip()}"
            n = add_node(key, label, "issue")
            n["capsules"].add(cid)
            issue_id_map[local_id] = key

        # Concept <-> Issue edges from issues' concept lists
        for local_id, i in issues.items():
            key_issue = issue_id_map[local_id]
            for c_local in i.get("concepts", []) or []:
                g_concept = concept_id_map.get(c_local)
                if not g_concept:
                    continue
                edges.append({
                    "source": g_concept,
                    "target": key_issue,
                    "type": "concept_issue",
                })

        # Intra-capsule edges from cap["edges"]
        # Expect format: {"type": "co_occurs" | "relates_to_issue" | "evidence", "source": ..., "target": ...}
        for e in cap.get("edges", []) or []:
            etype = e.get("type")
            src = e.get("source")
            tgt = e.get("target")

            if etype is None or src is None or tgt is None:
                continue

            # Map local ids to global node ids
            gsrc = None
            gtgt = None

            if src in concept_id_map:
                gsrc = concept_id_map[src]
            elif src in issue_id_map:
                gsrc = issue_id_map[src]

            if tgt in concept_id_map:
                gtgt = concept_id_map[tgt]
            elif tgt in issue_id_map:
                gtgt = issue_id_map[tgt]

            if not gsrc or not gtgt:
                continue

            # We'll simplify edge type buckets
            if etype == "co_occurs":
                etype_global = "concept_concept"
            elif etype in ("relates_to_issue", "evidence"):
                etype_global = etype
            else:
                etype_global = "other"

            edges.append({
                "source": gsrc,
                "target": gtgt,
                "type": etype_global,
            })

    # Convert capsules sets to lists, and compute simple degree
    for node in nodes.values():
        node["capsules"] = sorted(node["capsules"])
        node["degree"] = 0

    for e in edges:
        s = e["source"]
        t = e["target"]
        if s in nodes:
            nodes[s]["degree"] += 1
        if t in nodes:
            nodes[t]["degree"] += 1

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
    }


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/capsules")
def api_capsules():
    run_id = request.args.get("run", "__all__")
    capsules = load_capsules(run_id)
    out = []
    for cid, cap in capsules.items():
        out.append({
            "id": cid,
            "filename": cap.get("_filename", ""),
            "seed": cap.get("seed", ""),
            "run_id": cap.get("_run_id", run_id),
            "num_concepts": len(cap.get("concepts", {}) or {}),
            "num_issues": len(cap.get("issues", {}) or {}),
            "num_edges": len(cap.get("edges", []) or []),
            "num_questions": len(cap.get("questions", []) or []),
        })
    return jsonify(out)


@app.route("/api/graph")
def api_graph():
    run_id = request.args.get("run", "__all__")
    capsules = load_capsules(run_id)
    graph = build_global_graph(capsules)
    graph["run_id"] = run_id
    return jsonify(graph)


@app.route("/api/runs")
def api_runs():
    """
    List available runs (subdirectories) under CAPSULE_ROOT.
    """
    return jsonify(list_runs())


# app.py (or wherever your Flask app is defined)
from maxwell import open_ended_wonder, sanitized_filename



from flask import Flask, request, jsonify
from maxwell import open_ended_wonder, sanitized_filename

# ... app = Flask(__name__), other routes, etc ...

@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json(force=True) or {}
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "topic is required"}), 400

    model = data.get("model", "gemma3:12b")
    run_id = sanitized_filename(topic)

    # 🔒 try to take the lock without blocking
    if not search_lock.acquire(blocking=False):
        # someone else is already exploring
        return jsonify({"error": "exploration already in progress"}), 409

    try:
        # This is your open-ended exploration loop
        open_ended_wonder(topic, model)
    finally:
        # always release, even if something blows up
        search_lock.release()

    return jsonify({
        "id": run_id,
        "label": topic
    })


@app.route("/api/capsule/<capsule_id>")
def api_capsule(capsule_id: str):
    run_id = request.args.get("run", "__all__")
    capsules = load_capsules(run_id)
    cap = capsules.get(capsule_id)
    if not cap:
        abort(404)
    cap = dict(cap)
    cap.pop("_filename", None)
    cap.pop("_capsule_id", None)
    cap.pop("_run_id", None)
    return jsonify(cap)


if __name__ == "__main__":
    print(f"[MAXWELL OBSERVATORY] Root: {CAPSULE_ROOT}")
    app.run(host="0.0.0.0", port=1234, debug=True)
