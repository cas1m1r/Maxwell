# reflector.py
"""
Reflector: Maxwell-2.0 Third Agent (Graph-Aware)
-------------------------------------------------
Given a "knowledge capsule" (output from Mapper), Reflector:

- Reviews what Maxwell has learned about a topic
- Summarizes key answers / insights
- Highlights surprising or unexpected aspects
- Surfaces contradictions, complications, edge cases, or limits
- Generates new questions for deeper exploration

Now incorporates graph awareness by using:
- concept frequency (count)
- graph edges (co_occurs, relates_to_issue, evidence)
- issue connections (which concepts have many unresolved issues)
- document attachments (which concepts are well- or poorly-supported)
"""

from __future__ import annotations

import json
from typing import Dict, Any, List, Optional, Set, Tuple

from llama_utils import setup_client, ask_model


class Reflector:
    def __init__(
        self,
        endpoint: str,
        model: str,
        capsule: Optional[Dict[str, Any]] = None,
        capsule_path: Optional[str] = None,
    ):
        """
        You can either pass:
        - capsule: a dict returned by Mapper.to_capsule_dict(), OR
        - capsule_path: path to a saved capsule JSON file.

        endpoint: Ollama / LLM host
        model: model name (e.g. "llama3.1")
        """
        self.client = setup_client(endpoint)
        self.model = model

        if capsule is not None:
            self.capsule = capsule
        elif capsule_path is not None:
            with open(capsule_path, "r", encoding="utf-8") as f:
                self.capsule = json.load(f)
        else:
            raise ValueError("Reflector requires either 'capsule' or 'capsule_path'.")

        self.report: Optional[Dict[str, Any]] = None

    # -----------------------------------------------------
    # Internal helpers for graph analysis
    # -----------------------------------------------------
    def _build_graph_analysis(
        self,
    ) -> Dict[str, Any]:
        """
        Build basic graph metrics from the capsule:
        - concept degrees
        - issue connectivity
        - doc support counts
        - connected components (approx)
        """
        concepts: Dict[str, Dict[str, Any]] = self.capsule.get("concepts", {}) or {}
        issues: Dict[str, Dict[str, Any]] = self.capsule.get("issues", {}) or {}
        edges: List[Dict[str, Any]] = self.capsule.get("edges", []) or []
        documents: Dict[str, Dict[str, Any]] = self.capsule.get("documents", {}) or {}

        concept_ids: Set[str] = set(concepts.keys())

        # Degree (graph neighborhood size) per concept
        degree: Dict[str, int] = {cid: 0 for cid in concept_ids}
        # How many issues touch each concept
        issue_links: Dict[str, int] = {cid: 0 for cid in concept_ids}
        # How many docs attached to each concept
        doc_links: Dict[str, int] = {cid: 0 for cid in concept_ids}

        # Build adjacency list for components
        adjacency: Dict[str, Set[str]] = {cid: set() for cid in concept_ids}

        # Count document support
        for d in documents.values():
            cid = d.get("attached_to")
            if cid in doc_links:
                doc_links[cid] += 1

        # Count issue links from issues dict
        for iid, issue in issues.items():
            cids = issue.get("concepts", []) or []
            for cid in cids:
                if cid in issue_links:
                    issue_links[cid] += 1

        # Walk edges to compute degree and adjacency
        for e in edges:
            etype = e.get("type")
            src = e.get("source")
            tgt = e.get("target")

            if etype in ("co_occurs", "relates_to_issue", "evidence"):
                # For concepts only; ignore non-concept nodes for degree, except for adjacency bookkeeping
                if src in concept_ids and tgt in concept_ids:
                    # concept <-> concept edge
                    degree[src] += 1
                    degree[tgt] += 1
                    adjacency[src].add(tgt)
                    adjacency[tgt].add(src)
                elif src in concept_ids:
                    degree[src] += 1
                elif tgt in concept_ids:
                    degree[tgt] += 1

        # Find connected components among concepts using simple DFS
        visited: Set[str] = set()
        components: List[List[str]] = []

        def dfs(start: str, comp: List[str]) -> None:
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                for neigh in adjacency.get(node, []):
                    if neigh not in visited:
                        stack.append(neigh)

        for cid in concept_ids:
            if cid not in visited:
                comp_nodes: List[str] = []
                dfs(cid, comp_nodes)
                if comp_nodes:
                    components.append(comp_nodes)

        # Build a convenience structure that we can serialize
        return {
            "degree": degree,
            "issue_links": issue_links,
            "doc_links": doc_links,
            "components": components,
        }

    def _select_top_concepts(
        self,
        graph_info: Dict[str, Any],
        max_central: int = 10,
        max_controversial: int = 10,
        max_unsupported: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        From the graph metrics, select:
        - most central concepts (by degree + count)
        - most controversial (many issue_links)
        - potentially under-supported (high count / degree but low doc_links)
        """
        concepts: Dict[str, Dict[str, Any]] = self.capsule.get("concepts", {}) or {}
        degree: Dict[str, int] = graph_info["degree"]
        issue_links: Dict[str, int] = graph_info["issue_links"]
        doc_links: Dict[str, int] = graph_info["doc_links"]

        records: List[Dict[str, Any]] = []
        for cid, c in concepts.items():
            records.append(
                {
                    "id": cid,
                    "label": c.get("label", ""),
                    "count": c.get("count", 0),
                    "degree": degree.get(cid, 0),
                    "issues": issue_links.get(cid, 0),
                    "docs": doc_links.get(cid, 0),
                }
            )

        # Most central: sort by degree, then count
        central = sorted(
            records,
            key=lambda r: (r["degree"], r["count"]),
            reverse=True,
        )[:max_central]

        # Most controversial: many issues
        controversial = sorted(
            [r for r in records if r["issues"] > 0],
            key=lambda r: r["issues"],
            reverse=True,
        )[:max_controversial]

        # Undersupported: high degree or count but few docs
        undersupported = [
            r
            for r in records
            if (r["degree"] + r["count"] >= 2) and r["docs"] == 0
        ]
        undersupported = sorted(
            undersupported,
            key=lambda r: (r["degree"] + r["count"]),
            reverse=True,
        )[:max_unsupported]

        return {
            "central": central,
            "controversial": controversial,
            "undersupported": undersupported,
        }

    def _summarize_components(
        self,
        graph_info: Dict[str, Any],
        max_components: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Turn connected components into a small list of
        {size, concept_labels[]} for the biggest ones.
        """
        components: List[List[str]] = graph_info["components"]
        concepts: Dict[str, Dict[str, Any]] = self.capsule.get("concepts", {}) or {}

        # Sort components by size descending
        components_sorted = sorted(components, key=len, reverse=True)[:max_components]

        out: List[Dict[str, Any]] = []
        for comp in components_sorted:
            labels = [concepts[cid]["label"] for cid in comp if cid in concepts]
            out.append(
                {
                    "size": len(comp),
                    "concept_ids": comp,
                    "concept_labels": labels,
                }
            )
        return out

    # -----------------------------------------------------
    # Context construction (now graph-aware)
    # -----------------------------------------------------
    def _build_context(
        self,
        max_concepts: int = 15,
        max_issues: int = 10,
        max_questions: int = 10,
    ) -> str:
        seed = self.capsule.get("seed", "")
        concepts = self.capsule.get("concepts", {}) or {}
        issues = self.capsule.get("issues", {}) or {}
        questions = self.capsule.get("questions", []) or []
        documents = self.capsule.get("documents", {}) or {}

        # Graph analysis
        graph_info = self._build_graph_analysis()
        top_sets = self._select_top_concepts(graph_info)
        components_summary = self._summarize_components(graph_info)

        # Top concepts by count (raw frequency)
        concept_items: List[Dict[str, Any]] = []
        for cid, c in concepts.items():
            concept_items.append(
                {
                    "id": cid,
                    "label": c.get("label", ""),
                    "count": c.get("count", 0),
                    "questions": c.get("questions", []),
                }
            )
        concept_items.sort(key=lambda x: x["count"], reverse=True)
        concept_items = concept_items[:max_concepts]

        # Issues
        issue_items: List[Dict[str, Any]] = []
        for iid, i in issues.items():
            issue_items.append(
                {
                    "id": iid,
                    "label": i.get("label", ""),
                    "questions": i.get("questions", []),
                    "concepts": i.get("concepts", []),
                }
            )
        issue_items = issue_items[:max_issues]

        # Truncate questions
        questions = questions[:max_questions]

        # Sample documents (titles only)
        doc_items: List[Dict[str, Any]] = []
        for did, d in list(documents.items())[:10]:
            doc_items.append(
                {
                    "id": did,
                    "title": d.get("title", ""),
                    "source": d.get("source", ""),
                    "attached_to": d.get("attached_to", ""),
                }
            )

        lines: List[str] = []
        lines.append(f"SEED TOPIC: {seed}\n")

        # Basic concepts
        lines.append("KEY CONCEPTS (top by raw frequency):")
        if concept_items:
            for c in concept_items:
                lines.append(
                    f"- {c['label']} (id={c['id']}, count={c['count']})"
                )
        else:
            lines.append("- (none)")

        # Issues
        lines.append("\nISSUES / UNKNOWNS:")
        if issue_items:
            for i in issue_items:
                lines.append(
                    f"- {i['label']} (id={i['id']}, concepts={len(i['concepts'])})"
                )
        else:
            lines.append("- (none)")

        # Example questions
        lines.append("\nEXAMPLE QUESTIONS EXPLORED:")
        if questions:
            for q in questions:
                lines.append(f"- {q}")
        else:
            lines.append("- (none)")

        # Documents
        lines.append("\nATTACHED DOCUMENTS (if any):")
        if doc_items:
            for d in doc_items:
                lines.append(
                    f"- {d['title']} [source={d['source']}, concept_id={d['attached_to']}]"
                )
        else:
            lines.append("- (none)")

        # Graph-derived sections
        lines.append("\nGRAPH ANALYSIS:")

        lines.append("\n- MOST CENTRAL CONCEPTS (by degree + count):")
        if top_sets["central"]:
            for c in top_sets["central"]:
                lines.append(
                    f"  * {c['label']} (id={c['id']}, degree={c['degree']}, count={c['count']}, issues={c['issues']}, docs={c['docs']})"
                )
        else:
            lines.append("  * (none)")

        lines.append("\n- MOST CONTROVERSIAL CONCEPTS (linked to many issues):")
        if top_sets["controversial"]:
            for c in top_sets["controversial"]:
                lines.append(
                    f"  * {c['label']} (id={c['id']}, issues={c['issues']}, degree={c['degree']}, docs={c['docs']})"
                )
        else:
            lines.append("  * (none)")

        lines.append("\n- POTENTIALLY UNDER-SUPPORTED CONCEPTS (frequent/central but few docs):")
        if top_sets["undersupported"]:
            for c in top_sets["undersupported"]:
                lines.append(
                    f"  * {c['label']} (id={c['id']}, degree={c['degree']}, count={c['count']}, docs={c['docs']}, issues={c['issues']})"
                )
        else:
            lines.append("  * (none)")

        lines.append("\n- CONNECTED COMPONENTS (clusters of concepts):")
        if components_summary:
            for idx, comp in enumerate(components_summary):
                size = comp["size"]
                labels = ", ".join(comp["concept_labels"][:6])
                more_flag = "..." if size > 6 else ""
                lines.append(
                    f"  * Cluster {idx+1}: size={size}, sample_concepts=[{labels}{more_flag}]"
                )
        else:
            lines.append("  * (no concept clusters found)")

        return "\n".join(lines)

    # -----------------------------------------------------
    # Reflection call
    # -----------------------------------------------------
    def reflect(self) -> Dict[str, Any]:
        """
        Ask the model to:
        - summarize what we learned / answers we have
        - highlight surprises / unexpected aspects (including graph-based patterns)
        - describe contradictions, complications, edge cases, limits
        - generate new follow-up questions tied to concepts / clusters / issues

        Stores result in self.report and returns it.
        """
        context = self._build_context()

        # JSON template kept outside of f-string to avoid brace issues
        json_template = """
{
  "topic": "<seed topic>",
  "learned": {
    "overview": "<narrative summary of what we now 'know'>",
    "key_points": ["...", "..."]
  },
  "surprises": {
    "overview": "<what is unexpected or non-obvious, including any graph-derived patterns>",
    "items": ["...", "..."]
  },
  "contradictions": {
    "overview": "<where things are messy, conflicting, contextual, or limited>",
    "items": ["...", "..."]
  },
  "new_questions": {
    "overview": "<short framing of the next frontier>",
    "questions": ["...", "..."]
  },
  "graph_insights": {
    "central_concepts": ["<labels of key hubs>", "..."],
    "controversial_concepts": ["<labels with many linked issues>", "..."],
    "undersupported_concepts": ["<labels that need more evidence>", "..."],
    "interesting_clusters": ["<natural subdomains or themes>", "..."]
  }
}
"""

        prompt = (
            "You are Maxwell-2's Reflector agent.\n"
            "You are given a compressed description of what another agent has learned about a topic, "
            "including:\n"
            "- key concepts and issues/unknowns\n"
            "- questions explored\n"
            "- any attached documents\n"
            "- graph analysis of which concepts are central, controversial, undersupported, "
            "  and how they cluster.\n\n"
            "Your job is to:\n"
            "1) Explain what we have actually learned — what answers we now have, in clear human terms.\n"
            "2) Point out anything surprising, non-obvious, or counterintuitive in the learned structure, "
            "   especially using the graph patterns (central hubs, controversial concepts, sparse support, clusters).\n"
            "3) Identify contradictions, complications, edge cases, or important limitations/context where "
            "   the 'answer' is not clean or simple.\n"
            "4) Generate new, high-value questions that emerge from this structure — questions that would be "
            "   excellent next steps for deeper exploration, including follow-ups on specific concepts, issues, "
            "   or clusters.\n"
            "5) Summarize any particularly interesting 'graph insights' — for example, concepts that act as hubs "
            "   between clusters, contested concepts with many issues, or clusters that might represent "
            "   subdomains of the topic.\n\n"
            "Here is the compressed context:\n"
            "----------------------------------------\n"
            f"{context}\n"
            "----------------------------------------\n\n"
            "Now produce a reflection in STRICT JSON using this format:\n"
            f"{json_template}\n"
            "Do NOT add any commentary outside the JSON. Do not change key names."
        )

        response = ask_model(self.client, self.model, prompt)
        raw = response.message.content

        # Extract JSON from model output
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON object delimiters found in model output.")
            data = json.loads(raw[start : end + 1])
        except Exception:
            # fallback to very simple structure
            data = {
                "topic": self.capsule.get("seed", ""),
                "learned": {
                    "overview": raw.strip(),
                    "key_points": [],
                },
                "surprises": {
                    "overview": "",
                    "items": [],
                },
                "contradictions": {
                    "overview": "",
                    "items": [],
                },
                "new_questions": {
                    "overview": "",
                    "questions": [],
                },
                "graph_insights": {
                    "central_concepts": [],
                    "controversial_concepts": [],
                    "undersupported_concepts": [],
                    "interesting_clusters": [],
                },
            }

        # Ensure minimum structure
        topic = self.capsule.get("seed", "")
        data.setdefault("topic", topic)

        data.setdefault("learned", {})
        data["learned"].setdefault("overview", "")
        data["learned"].setdefault("key_points", [])

        data.setdefault("surprises", {})
        data["surprises"].setdefault("overview", "")
        data["surprises"].setdefault("items", [])

        data.setdefault("contradictions", {})
        data["contradictions"].setdefault("overview", "")
        data["contradictions"].setdefault("items", [])

        data.setdefault("new_questions", {})
        data["new_questions"].setdefault("overview", "")
        data["new_questions"].setdefault("questions", [])

        data.setdefault("graph_insights", {})
        gi = data["graph_insights"]
        gi.setdefault("central_concepts", [])
        gi.setdefault("controversial_concepts", [])
        gi.setdefault("undersupported_concepts", [])
        gi.setdefault("interesting_clusters", [])

        self.report = data
        return data

    # -----------------------------------------------------
    # Convenience: save reflection report
    # -----------------------------------------------------
    def save_report(self, path: str) -> None:
        """
        Save the reflection report as JSON.
        Only valid after calling reflect().
        """
        if self.report is None:
            raise RuntimeError("No report available. Call .reflect() first.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
