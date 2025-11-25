# mapper.py
"""
Mapper: Maxwell-2.0 Second Agent
--------------------------------
Responsible for taking Explorer output and organizing it into:

- A concept index (normalized concepts, variants, question usage)
- An issue index (unknowns / controversies)
- A lightweight graph (concept <-> concept, concept <-> issue, concept <-> doc)
- A serializable "knowledge capsule" that can be saved and reloaded.

Expected input (from Explorer.to_json()):
{
  "seed": "...",
  "generated_questions": [...],
  "nodes": [
    {
      "question": "...",
      "sources": [...],
      "summary": "...",
      "key_concepts": ["...", "..."],
      "unknowns": ["...", "..."],
      "timestamp": ...,
      "retrieval_plan": {...}   # optional; not required by Mapper
    },
    ...
  ]
}
"""

from __future__ import annotations

import json
import time
from typing import Dict, Any, List, Set, Optional


class Mapper:
    def __init__(self, explorer_output: Dict[str, Any]):
        """
        explorer_output: dict returned by Explorer.to_json()
        """
        self.seed: str = explorer_output.get("seed", "")
        self.questions: List[str] = explorer_output.get(
            "generated_questions", []
        )
        self.raw_nodes: List[Dict[str, Any]] = explorer_output.get("nodes", [])

        # Core structures
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.issues: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []   # graph edges
        self.documents: Dict[str, Dict[str, Any]] = {}  # external evidence docs

        # Internal counters
        self._doc_counter: int = 0

    # ------------------------
    # Normalization helpers
    # ------------------------
    @staticmethod
    def _norm(name: str) -> str:
        """
        Normalize a concept/issue label into a stable ID.
        Lowercase, strip, replace spaces with underscores.
        """
        return "_".join(name.strip().lower().split())

    # ------------------------
    # Core build steps
    # ------------------------
    def build_concept_index(self) -> None:
        """
        Populate self.concepts and self.issues from raw_nodes.
        Also attach which questions each concept/issue appears in.
        """
        for node_idx, node in enumerate(self.raw_nodes):
            q_label = node.get("question") or (
                self.questions[node_idx]
                if node_idx < len(self.questions)
                else f"Q{node_idx}"
            )

            # --- Concepts ---
            for c in node.get("key_concepts", []):
                if not c:
                    continue
                c_str = str(c).strip()
                cid = self._norm(c_str)

                entry = self.concepts.setdefault(
                    cid,
                    {
                        "id": cid,
                        "label": c_str,       # first-seen label
                        "variants": set(),    # different surface forms
                        "questions": set(),   # question texts
                        "count": 0,           # how many times seen
                    },
                )
                entry["variants"].add(c_str)
                entry["questions"].add(q_label)
                entry["count"] += 1

            # --- Issues / unknowns ---
            for u in node.get("unknowns", []):
                if not u:
                    continue
                u_str = str(u).strip()
                uid = f"issue_{self._norm(u_str)}"

                ientry = self.issues.setdefault(
                    uid,
                    {
                        "id": uid,
                        "label": u_str,
                        "questions": set(),
                        "concepts": set(),   # which concepts it co-occurs with
                    },
                )
                ientry["questions"].add(q_label)

                # We will fill concepts in _build_graph()

    def build_graph(self) -> None:
        """
        Build graph edges:
        - concept <-> concept (co-occurs in same node)
        - concept <-> issue  (issue appears with that concept)
        """
        self.edges = []

        for node_idx, node in enumerate(self.raw_nodes):
            q_label = node.get("question") or (
                self.questions[node_idx]
                if node_idx < len(self.questions)
                else f"Q{node_idx}"
            )

            concept_labels = [str(c).strip() for c in node.get("key_concepts", []) if c]
            issue_labels = [str(u).strip() for u in node.get("unknowns", []) if u]

            concept_ids = [self._norm(c) for c in concept_labels]
            issue_ids = [f"issue_{self._norm(u)}" for u in issue_labels]

            # --- Concept <-> Concept co-occurrence edges ---
            for i in range(len(concept_ids)):
                for j in range(i + 1, len(concept_ids)):
                    src = concept_ids[i]
                    tgt = concept_ids[j]
                    if src == tgt:
                        continue
                    self.edges.append(
                        {
                            "type": "co_occurs",
                            "source": src,
                            "target": tgt,
                            "question": q_label,
                        }
                    )

            # --- Concept <-> Issue edges ---
            for issue_id in issue_ids:
                issue_entry = self.issues.get(issue_id)
                if issue_entry is None:
                    # if issue not seen yet (unlikely), create minimal entry
                    idx = issue_labels[issue_ids.index(issue_id)]
                    self.issues[issue_id] = {
                        "id": issue_id,
                        "label": idx,
                        "questions": {q_label},
                        "concepts": set(),
                    }
                    issue_entry = self.issues[issue_id]

                for cid in concept_ids:
                    self.edges.append(
                        {
                            "type": "relates_to_issue",
                            "source": cid,
                            "target": issue_id,
                            "question": q_label,
                        }
                    )
                    issue_entry["concepts"].add(cid)

    # ------------------------
    # Document / evidence ingestion
    # ------------------------
    def ingest_document(
        self,
        concept_label: str,
        title: str,
        url: str,
        summary: str,
        source: str = "wikipedia",
    ) -> str:
        """
        Optional: call this from your retrieval pipeline to attach
        external documents (e.g., wiki summaries) as evidence.

        - creates/updates the concept entry if needed
        - creates a document node
        - adds an edge concept -> document (type='evidence')

        Returns the document ID.
        """
        cid = self._norm(concept_label)
        c_entry = self.concepts.setdefault(
            cid,
            {
                "id": cid,
                "label": concept_label,
                "variants": set([concept_label]),
                "questions": set(),
                "count": 0,
            },
        )

        self._doc_counter += 1
        doc_id = f"doc_{self._doc_counter}"

        self.documents[doc_id] = {
            "id": doc_id,
            "title": title,
            "url": url,
            "summary": summary,
            "source": source,
            "attached_to": cid,
            "timestamp": time.time(),
        }

        self.edges.append(
            {
                "type": "evidence",
                "source": cid,     # concept
                "target": doc_id,  # document
            }
        )

        return doc_id

    # ------------------------
    # Serialization
    # ------------------------
    def to_capsule_dict(self) -> Dict[str, Any]:
        """
        Convert internal structures into a JSON-serializable dict.
        (Sets become lists.)
        """
        concepts_out: Dict[str, Any] = {}
        for cid, entry in self.concepts.items():
            concepts_out[cid] = {
                "id": entry["id"],
                "label": entry["label"],
                "variants": sorted(list(entry.get("variants", set()))),
                "questions": sorted(list(entry.get("questions", set()))),
                "count": entry.get("count", 0),
            }

        issues_out: Dict[str, Any] = {}
        for iid, entry in self.issues.items():
            issues_out[iid] = {
                "id": entry["id"],
                "label": entry["label"],
                "questions": sorted(list(entry.get("questions", set()))),
                "concepts": sorted(list(entry.get("concepts", set()))),
            }

        capsule = {
            "seed": self.seed,
            "created_at": time.time(),
            "questions": self.questions,
            "raw_nodes": self.raw_nodes,   # original Explorer nodes
            "concepts": concepts_out,
            "issues": issues_out,
            "edges": self.edges,
            "documents": self.documents,
        }
        return capsule

    def save_capsule(self, path: str) -> None:
        """
        Save the capsule to disk as JSON.
        """
        capsule = self.to_capsule_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(capsule, f, indent=2, ensure_ascii=False)

    # ------------------------
    # Reloading memory
    # ------------------------
    @classmethod
    def from_capsule_dict(cls, capsule: Dict[str, Any]) -> "Mapper":
        """
        Rebuild a Mapper instance from a previously saved capsule dict.
        This allows "re-instantiating" prior learning.
        """
        explorer_like = {
            "seed": capsule.get("seed", ""),
            "generated_questions": capsule.get("questions", []),
            "nodes": capsule.get("raw_nodes", []),
        }
        mapper = cls(explorer_like)

        # Replace auto-built structures with stored ones
        mapper.concepts = {}
        for cid, entry in capsule.get("concepts", {}).items():
            mapper.concepts[cid] = {
                "id": entry["id"],
                "label": entry["label"],
                "variants": set(entry.get("variants", [])),
                "questions": set(entry.get("questions", [])),
                "count": entry.get("count", 0),
            }

        mapper.issues = {}
        for iid, entry in capsule.get("issues", {}).items():
            mapper.issues[iid] = {
                "id": entry["id"],
                "label": entry["label"],
                "questions": set(entry.get("questions", [])),
                "concepts": set(entry.get("concepts", [])),
            }

        mapper.edges = capsule.get("edges", [])
        mapper.documents = capsule.get("documents", {})
        mapper._doc_counter = len(mapper.documents)

        return mapper

    @classmethod
    def load_capsule(cls, path: str) -> "Mapper":
        """
        Load a capsule JSON from disk and instantiate a Mapper.
        """
        with open(path, "r", encoding="utf-8") as f:
            capsule = json.load(f)
        return cls.from_capsule_dict(capsule)
