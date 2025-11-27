# meta_explorer.py
"""
Meta Explorer – second-order exploration on top of Maxwell capsules.

Goal
----
Given a pile of capsules (one run or all runs), MetaExplorer:

1. Merges them into a global concept graph (via merge_capsules.py).
2. Computes simple graph + usage statistics:
   - degree per concept
   - connected components
   - run/question/usage stats
3. Detects "interesting" second-order structures:
   - Bridge concepts: concepts that appear in many runs and are well-connected.
   - Small cross-run islands: tight little components that span multiple runs.
   - Fringe concepts: rare concepts that hang off dense cores.

4. Emits "meta seeds": natural-language prompts you can feed back into
   `open_ended_wonder` OR just inspect as suggested questions.

This module is intentionally pure-Python (no networkx dependency) and
can be used either as a library or a CLI.

Typical CLI usage
-----------------
    export CAPSULE_ROOT=/path/to/capsules
    # explore over all capsules under CAPSULE_ROOT
    python meta_explorer.py --root "$CAPSULE_ROOT"

    # or restrict to a single run folder
    python meta_explorer.py --root "$CAPSULE_ROOT" --run some_run_id

You can also import it:

    from meta_explorer import MetaExplorer

    mex = MetaExplorer(capsule_root=".", run_id="__all__")
    mex.load_and_merge()
    seeds = mex.generate_meta_seeds()
    for s in seeds:
        print(s.category, s.question)

If you want to *automatically* spin new Maxwell explorations from these
meta-seeds, see `spawn_meta_explorations()` at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import json
import os

from merge_capsules import MergedGraph, merge_capsules


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------


@dataclass
class MetaSeed:
    """
    A second-order question the system thinks is worth exploring.

    id          : stable slug (e.g., "bridge:what_is_malware")
    category    : "bridge" | "island" | "fringe"
    question    : natural-language prompt you can feed to open_ended_wonder
    rationale   : short explanation of why this was selected
    concepts    : list of concept IDs involved
    concept_labels : same, but pretty labels
    runs        : run_ids that motivated this seed
    component_id: optional connected-component index (for islands)
    extra       : arbitrary metadata (scores, counts, etc.)
    """

    id: str
    category: str
    question: str
    rationale: str
    concepts: List[str]
    concept_labels: List[str]
    runs: List[str]
    component_id: Optional[int] = None
    extra: Dict[str, Any] = None


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------


def _slug(text: str) -> str:
    """Crude slugifier for IDs."""
    return (
        text.lower()
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("?", "")
        .replace(":", "")
    )[:80]


def _connected_components(adjacency: Dict[str, Set[str]]) -> List[Set[str]]:
    """
    Simple DFS connected components over an undirected adjacency map.
    """
    remaining: Set[str] = set(adjacency.keys())
    components: List[Set[str]] = []

    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = {start}
        while stack:
            node = stack.pop()
            for nb in adjacency.get(node, ()):
                if nb in remaining:
                    remaining.remove(nb)
                    comp.add(nb)
                    stack.append(nb)
        components.append(comp)

    return components


# -------------------------------------------------------------------
# Core class
# -------------------------------------------------------------------


class MetaExplorer:
    def __init__(self, capsule_root: str, run_id: str = "__all__"):
        """
        capsule_root: directory that contains your run subfolders / capsules.
                      Should match CAPSULE_ROOT used by app.py.
        run_id      : "__all__" to scan all *.json under capsule_root,
                      or a specific subdirectory name for a single run.
        """
        self.capsule_root = Path(capsule_root).resolve()
        self.run_id = run_id

        self.merged: Optional[MergedGraph] = None
        self.adjacency: Dict[str, Set[str]] = {}
        self.degree: Dict[str, int] = {}
        self.components: List[Set[str]] = []

    # -----------------------------
    # Loading / merging
    # -----------------------------

    def _collect_capsule_paths(self) -> List[Path]:
        if self.run_id == "__all__":
            search_dir = self.capsule_root
        else:
            search_dir = self.capsule_root / self.run_id

        if not search_dir.exists() or not search_dir.is_dir():
            raise FileNotFoundError(f"Capsule directory not found: {search_dir}")

        paths = sorted(search_dir.glob("*.json"))
        return paths

    def load_and_merge(self) -> None:
        """
        Load all capsules for this run and build a MergedGraph in memory.
        """
        paths = self._collect_capsule_paths()
        if not paths:
            raise RuntimeError(f"No capsule JSON files found for run '{self.run_id}'")

        self.merged = merge_capsules(paths)
        self._build_graph_view()

    # -----------------------------
    # Internal: build concept graph
    # -----------------------------

    def _build_graph_view(self) -> None:
        """
        Build concept-only adjacency & degree + connected components
        from self.merged.concept_cooccurrence.
        """
        if self.merged is None:
            raise RuntimeError("MetaExplorer needs merged graph; call load_and_merge() first.")

        # Initialize adjacency with empty sets
        self.adjacency = {cid: set() for cid in self.merged.concepts.keys()}

        # Fill adjacency from co-occurrence (undirected)
        for (a, b), weight in self.merged.concept_cooccurrence.items():
            if a == b:
                continue
            if a not in self.adjacency or b not in self.adjacency:
                continue
            self.adjacency[a].add(b)
            self.adjacency[b].add(a)

        # Degree is just neighbor set size here
        self.degree = {cid: len(nbrs) for cid, nbrs in self.adjacency.items()}

        # Connected components
        self.components = _connected_components(self.adjacency)

    # -------------------------------------------------------------------
    # Meta-seed generation
    # -------------------------------------------------------------------

    def generate_meta_seeds(
        self,
        max_bridge: int = 6,
        max_islands: int = 6,
        max_fringe: int = 6,
        min_runs_for_bridge: int = 2,
    ) -> List[MetaSeed]:
        """
        High-level orchestration: call this after load_and_merge().

        Returns a list of MetaSeed objects ordered (roughly) by importance.
        """
        if self.merged is None or not self.adjacency:
            self.load_and_merge()

        seeds: List[MetaSeed] = []

        seeds.extend(
            self._detect_bridge_concepts(
                max_bridge=max_bridge, min_runs=min_runs_for_bridge
            )
        )

        seeds.extend(self._detect_island_components(max_islands=max_islands))
        seeds.extend(self._detect_fringe_concepts(max_fringe=max_fringe))

        # Stable ordering: bridges first, then islands, then fringe
        category_order = {"bridge": 0, "island": 1, "fringe": 2}
        seeds.sort(key=lambda s: (category_order.get(s.category, 99), s.id))

        return seeds

    # ----------------------------- Bridge concepts -----------------------------

    def _detect_bridge_concepts(
        self, max_bridge: int = 6, min_runs: int = 2
    ) -> List[MetaSeed]:
        """
        Bridge = concept that:
          - appears in multiple runs
          - has reasonably high degree in the global graph
        """
        merged = self.merged
        assert merged is not None

        candidates: List[Tuple[float, str]] = []

        for cid, gc in merged.concepts.items():
            runs = len(gc.runs)
            if runs < min_runs:
                continue

            deg = self.degree.get(cid, 0)
            if deg < 2:
                continue

            # simple score: runs * degree * log(1 + total_count)
            score = runs * deg * (1.0 + (gc.total_count ** 0.5))
            candidates.append((score, cid))

        candidates.sort(reverse=True)
        chosen = candidates[:max_bridge]

        seeds: List[MetaSeed] = []

        for score, cid in chosen:
            gc = merged.concepts[cid]
            run_ids = sorted(gc.runs)
            run_labels = [
                merged.runs[r].get("seed", r) for r in run_ids if r in merged.runs
            ]

            # keep it readable
            preview_runs = ", ".join(run_labels[:4])
            if len(run_labels) > 4:
                preview_runs += ", …"

            question = (
                f"How does the concept '{gc.label}' connect and differentiate its role "
                f"across topics like {preview_runs}? What deeper pattern or theory might "
                f"explain why it keeps reappearing?"
            )

            rationale = (
                f"'{gc.label}' appears in {len(run_ids)} runs and connects to "
                f"{self.degree.get(cid, 0)} other concepts (score={score:.1f}). "
                "It looks like a bridge across multiple domains."
            )

            seeds.append(
                MetaSeed(
                    id=f"bridge:{cid}",
                    category="bridge",
                    question=question,
                    rationale=rationale,
                    concepts=[cid],
                    concept_labels=[gc.label],
                    runs=run_ids,
                    component_id=None,
                    extra={
                        "score": score,
                        "degree": self.degree.get(cid, 0),
                        "run_count": len(run_ids),
                        "total_count": gc.total_count,
                    },
                )
            )

        return seeds

    # ----------------------------- Island components ---------------------------

    def _detect_island_components(self, max_islands: int = 6) -> List[MetaSeed]:
        """
        Island = relatively small connected component that spans multiple runs.

        These are often "micro-constellations" of concepts that are internally
        coherent but not strongly tied into the rest of the graph.
        """
        merged = self.merged
        assert merged is not None

        seeds: List[MetaSeed] = []

        for idx, comp in enumerate(self.components):
            size = len(comp)
            if size < 3 or size > 40:
                # too small to be interesting or too big to be a "pocket"
                continue

            # which runs are represented in this component?
            runs_here: Set[str] = set()
            for cid in comp:
                gc = merged.concepts[cid]
                runs_here.update(gc.runs)

            if len(runs_here) < 2:
                # pure single-run cluster – more like internal structure than a meta-gap
                continue

            # Central concept = highest degree within the component
            center_id = max(comp, key=lambda c: self.degree.get(c, 0))
            center = merged.concepts[center_id]

            labels = [merged.concepts[c].label for c in comp]
            labels_sorted = sorted(labels, key=len)  # short names first
            preview_labels = ", ".join(labels_sorted[:5])
            if len(labels_sorted) > 5:
                preview_labels += ", …"

            run_labels = [
                merged.runs[r].get("seed", r) for r in sorted(runs_here) if r in merged.runs
            ]
            preview_runs = ", ".join(run_labels[:4])
            if len(run_labels) > 4:
                preview_runs += ", …"

            question = (
                f"Is there a deeper organizing story that links concepts like "
                f"{preview_labels} across topics such as {preview_runs}? "
                f"What 'hidden chapter' of the domain might this small island represent?"
            )

            rationale = (
                f"Component {idx} is a small island of {size} concepts touching "
                f"{len(runs_here)} different runs, centered roughly on '{center.label}'. "
                "It looks like a cross-run pocket that isn't fully integrated yet."
            )

            seeds.append(
                MetaSeed(
                    id=f"island:{idx}:{_slug(center.label)}",
                    category="island",
                    question=question,
                    rationale=rationale,
                    concepts=sorted(list(comp)),
                    concept_labels=labels_sorted,
                    runs=sorted(list(runs_here)),
                    component_id=idx,
                    extra={
                        "component_size": size,
                        "run_count": len(runs_here),
                        "center_concept": center_id,
                        "center_degree": self.degree.get(center_id, 0),
                    },
                )
            )

        # choose the "most interesting" islands:
        # sort by (run_count * component_size) descending
        seeds.sort(
            key=lambda s: (
                -(s.extra.get("run_count", 1) * s.extra.get("component_size", 1)),
                s.id,
            )
        )
        return seeds[:max_islands]

    # ----------------------------- Fringe concepts -----------------------------

    def _detect_fringe_concepts(self, max_fringe: int = 6) -> List[MetaSeed]:
        """
        Fringe = low-frequency concepts that hang off higher-degree cores.

        Intuition: these are often emerging sub-topics or weird edge-cases.
        """
        merged = self.merged
        assert merged is not None

        candidates: List[Tuple[float, str]] = []

        for cid, gc in merged.concepts.items():
            count = gc.total_count
            deg = self.degree.get(cid, 0)
            if count > 3:
                continue  # not that rare
            if deg < 2:
                continue  # too isolated to be a "fringe of something"

            # neighbors
            neighbors = list(self.adjacency.get(cid, []))
            if not neighbors:
                continue

            avg_neighbor_deg = sum(self.degree.get(n, 0) for n in neighbors) / len(
                neighbors
            )

            # We like cases where this node is rare but its neighbors are busy.
            score = (avg_neighbor_deg + 1.0) / (count + 0.5)
            candidates.append((score, cid))

        candidates.sort(reverse=True)
        chosen = candidates[:max_fringe]

        seeds: List[MetaSeed] = []

        for score, cid in chosen:
            gc = merged.concepts[cid]
            neighbors = list(self.adjacency.get(cid, []))
            neighbor_labels = [
                merged.concepts[n].label for n in neighbors if n in merged.concepts
            ]
            neighbor_preview = ", ".join(sorted(neighbor_labels, key=len)[:6])
            if len(neighbor_labels) > 6:
                neighbor_preview += ", …"

            question = (
                f"'{gc.label}' only shows up rarely, but it is connected to busy ideas "
                f"like {neighbor_preview}. What subtle edge-case or emerging sub-topic "
                f"does this fringe concept reveal?"
            )

            rationale = (
                f"'{gc.label}' has total_count={gc.total_count} but degree={self.degree.get(cid, 0)} "
                f"into denser regions (fringe_score={score:.2f}). It may be pointing at "
                "an under-explored corner of the domain."
            )

            # collect runs via the concept + neighbors
            runs_here: Set[str] = set(gc.runs)
            for n in neighbors:
                runs_here.update(merged.concepts[n].runs)

            seeds.append(
                MetaSeed(
                    id=f"fringe:{cid}",
                    category="fringe",
                    question=question,
                    rationale=rationale,
                    concepts=[cid] + neighbors,
                    concept_labels=[gc.label] + neighbor_labels,
                    runs=sorted(list(runs_here)),
                    component_id=None,
                    extra={
                        "score": score,
                        "degree": self.degree.get(cid, 0),
                        "total_count": gc.total_count,
                        "avg_neighbor_degree": (
                            sum(self.degree.get(n, 0) for n in neighbors)
                            / max(len(neighbors), 1)
                        ),
                    },
                )
            )

        return seeds

    # -------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------

    def to_json(self, seeds: List[MetaSeed]) -> Dict[str, Any]:
        merged = self.merged
        return {
            "run_id": self.run_id,
            "root": str(self.capsule_root),
            "num_concepts": len(merged.concepts) if merged else 0,
            "num_components": len(self.components),
            "meta_seeds": [asdict(s) for s in seeds],
        }

    # Optional: automatically spawn new explorations from seeds.
    # Only used if you want a full feedback loop.

    def spawn_meta_explorations(
        self,
        seeds: List[MetaSeed],
        model: str = "gemma3:12b",
        max_to_run: Optional[int] = None,
    ) -> None:
        """
        For each meta-seed, invoke open_ended_wonder(question, model).

        This assumes maxwell.open_ended_wonder writes capsules somewhere
        under CAPSULE_ROOT just like manual runs.
        """
        from maxwell import open_ended_wonder  # local import to avoid cycles

        limit = max_to_run if max_to_run is not None else len(seeds)
        for s in seeds[:limit]:
            print(f"[META] Spawning exploration for: {s.id}")
            print(f"       Q: {s.question}\n")
            open_ended_wonder(s.question, model=model)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def _main():
    import argparse

    default_root = os.getenv("CAPSULE_ROOT", ".")

    parser = argparse.ArgumentParser(
        description="Meta Explorer: generate second-order questions from merged capsules."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=default_root,
        help="Capsule root directory (default: $CAPSULE_ROOT or current dir)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="__all__",
        help="Run id to analyze (subdirectory name) or '__all__' for all capsules in root.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write meta-seeds JSON. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        help="If set, automatically spawn Maxwell explorations for the generated seeds.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:12b",
        help="Model name for spawned explorations (if --spawn).",
    )
    args = parser.parse_args()

    mex = MetaExplorer(capsule_root=args.root, run_id=args.run)
    mex.load_and_merge()
    seeds = mex.generate_meta_seeds()

    payload = mex.to_json(seeds)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[meta_explorer] Wrote {len(seeds)} meta-seeds to {args.json_out}")
    else:
        # pretty-print to stdout
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.spawn and seeds:
        mex.spawn_meta_explorations(seeds, model=args.model)


if __name__ == "__main__":
    _main()
