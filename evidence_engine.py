# evidence_engine.py
import json
from typing import Dict, Any, List, Optional

from llama_utils import ask_model, setup_client
from retrievers import WikipediaRetriever, WebResult


class EvidenceEngine:
    def __init__(self, endpoint: str, model: str):
        self.client = setup_client(endpoint)
        self.model = model
        self.retriever = WikipediaRetriever()

    def _run_queries(self, queries: str, max_docs: int = 3) -> List[WebResult]:
        """Run a batch of queries and dedupe by URL."""
        results: List[WebResult] = []
        seen = set()

        h = self.retriever.search(queries.split(':')[0], max_results=max_docs)
        if h not in seen:
            seen.add(h)
            results.append(h)
        return results

    def _summarize_evidence(
        self,
        node_question: str,
        target: str,
        docs: List[WebResult],
    ) -> Optional[Dict[str, Any]]:
        if not docs:
            return None

        context_lines = []
        for d in docs:
            context_lines.append(
                f"{d}\n"
            )
        context_text = "\n---\n".join(context_lines)

        json_template = """
{
  "stance": "supports"|"challenges"|"extends",
  "summary": "<2-4 sentence synthesis>",
  "points": ["...", "..."],
  "references": [{"title": "...", "url": "..."}]
}
"""

        prompt = (
            "You are Maxwell-2's evidence mapper.\n\n"
            f"Central question: {node_question}\n"
            f"Focus: {target}\n\n"
            "The following web search results are relevant:\n"
            f"{context_text}\n\n"
            "Based on these results, assess how they relate to the current description of this concept/issue.\n\n"
            "Return STRICT JSON in the following format:\n"
            f"{json_template}\n"
            "Do NOT add any text outside the JSON."
        )

        resp = ask_model(self.client, self.model, prompt)
        raw = resp.message.content

        try:
            start = raw.find("{")
            end = raw.rfind("}")
            data = json.loads(raw[start : end + 1])
        except Exception:
            return None

        return {
            "target": target,
            "summary": data.get("summary", ""),
            "points": data.get("points", []),
            "references": data.get("references", []),
        }

    def enrich_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes a node from Explorer (with 'retrieval_plan') and
        attaches an 'evidence' list with stance-aware summaries.
        """
        plan = node.get("retrieval_plan", {})
        concept_plans = node.get("key_concepts", [])
        issue_plans = plan.get("issue_plans", [])

        evidence_entries: List[Dict[str, Any]] = []

        question = node.get("question", "")

        # Concepts
        for queries in concept_plans:
            docs = self._run_queries(queries)
            ev = self._summarize_evidence(
                node_question=question,
                target=queries,
                docs=docs,
            )
            if ev:
                evidence_entries.append(ev)

        # Issues / unknowns
        for iplan in issue_plans:
            issue = iplan.get("issue", "")
            queries = iplan.get("queries", [])
            docs = self._run_queries(queries)
            ev = self._summarize_evidence(
                node_question=question,
                target=issue,
                kind="unknown",
                stance_hint="",
                docs=docs,
            )
            if ev:
                evidence_entries.append(ev)

        node["evidence"] = evidence_entries
        return node
