# graph_merge.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import json
import re


# ---------- Data classes ----------

@dataclass
class GlobalConcept:
    id: str                         # canonical ID (normalized label)
    label: str                      # “nice” label
    variants: Set[str] = field(default_factory=set)
    runs: Set[str] = field(default_factory=set)
    questions: Set[str] = field(default_factory=set)
    total_count: int = 0


@dataclass
class MergedGraph:
    concepts: Dict[str, GlobalConcept] = field(default_factory=dict)
    runs: Dict[str, dict] = field(default_factory=dict)  # run_id -> {seed, path}
    run_concept_edges: List[Tuple[str, str]] = field(default_factory=list)
    concept_cooccurrence: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    def to_json(self) -> dict:
        return {
            "concepts": {
                cid: {
                    **asdict(c),
                    "runs": sorted(c.runs),
                    "questions": sorted(c.questions),
                    "variants": sorted(c.variants),
                }
                for cid, c in self.concepts.items()
            },
            "runs": self.runs,
            "run_concept_edges": [
                {"type": "run-concept", "run": r, "concept": c}
                for (r, c) in self.run_concept_edges
            ],
            "concept_concept_edges": [
                {"type": "concept-concept", "source": a, "target": b, "weight": w}
                for (a, b), w in self.concept_cooccurrence.items()
            ],
        }


# ---------- Helpers ----------

def normalize_label(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[\(\)\[\]:,]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


# ---------- Core merge logic ----------

def merge_capsules(paths: List[Path]) -> MergedGraph:
    merged = MergedGraph()

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            capsule = json.load(f)

        run_id = path.stem
        seed = capsule.get("seed", "")
        merged.runs[run_id] = {"seed": seed, "path": str(path)}

        # 1) Build local mapping: question_text -> set(local_concept_ids)
        q_to_local_concepts: Dict[str, Set[str]] = defaultdict(set)

        concepts = capsule.get("concepts", {})
        for local_cid, c in concepts.items():
            label = c.get("label") or c.get("id") or local_cid
            variants = set(c.get("variants") or [])
            questions = set(c.get("questions") or [])
            count = c.get("count", 0)

            norm_label = normalize_label(label)
            norm_variants = {normalize_label(v) for v in variants}

            # Find or create global concept
            global_id = _find_or_create_global_concept(
                merged,
                norm_label=norm_label,
                label=label,
                norm_variants=norm_variants,
            )

            gc = merged.concepts[global_id]
            gc.runs.add(run_id)
            gc.questions.update(questions)
            gc.variants.update(variants)
            gc.total_count += count

            # For co-occurrence later
            for q in questions:
                q_to_local_concepts[q].add(global_id)

            # Edge: run -> concept
            merged.run_concept_edges.append((run_id, global_id))

        # 2) Build co-occurrence edges for this run
        for _, concept_ids in q_to_local_concepts.items():
            ids = sorted(concept_ids)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pair = (ids[i], ids[j])
                    merged.concept_cooccurrence[pair] += 1

    return merged


def _find_or_create_global_concept(
    merged: MergedGraph,
    norm_label: str,
    label: str,
    norm_variants: Set[str],
) -> str:
    # Direct match on normalized label
    if norm_label in merged.concepts:
        return norm_label

    # Variant match against existing concepts
    for cid, gc in merged.concepts.items():
        existing_keys = {normalize_label(gc.label)} | {normalize_label(v) for v in gc.variants}
        if norm_label in existing_keys or (existing_keys & norm_variants):
            return cid

    # Fallback: new concept
    merged.concepts[norm_label] = GlobalConcept(id=norm_label, label=label, variants=set())
    return norm_label


# ---------- CLI / entrypoint ----------

def merge_capsules_in_dir(input_dir: str, output_path: str) -> None:
    base = Path(input_dir)
    paths = sorted(base.glob("*.json"))
    merged = merge_capsules(paths)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged.to_json(), f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge Maxwell capsules into a global concept graph.")
    parser.add_argument("input_dir", help="Directory containing capsule JSON files")
    parser.add_argument("output", help="Path for merged graph JSON")
    args = parser.parse_args()

    merge_capsules_in_dir(args.input_dir, args.output)
