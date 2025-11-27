# explorer.py
import json
import time
from typing import List, Dict, Any
from llama_utils import setup_client, ask_model



class Explorer:
    """
    Maxwell-2.0 Explorer
    --------------------
    Stage 1 of the epistemic pipeline.
    Responsible for:
    - Turning a seed topic into sub-questions
    - Suggesting external sources to query
    - Ingesting preliminary knowledge into JSON nodes
    - Building retrieval plans for deeper external exploration
    """

    def __init__(self, seed_topic: str, endpoint: str, model: str):
        self.seed = seed_topic
        self.model = model
        self.client = setup_client(endpoint)
        self.sub_questions: List[str] = []
        self.knowledge_nodes: List[Dict[str, Any]] = []

        #  derive sub-questions from the topic and do preliminary research
        self.pre_process_topic()

    # -----------------------------------------------------
    # STAGE 1: Generate exploratory sub-questions
    # -----------------------------------------------------
    def pre_process_topic(self) -> None:
        prompt = (
            "You will be given a topic. Produce a concise list of 5–10 high-value exploratory questions that "
            "would help someone deeply map the conceptual landscape of the topic.\n"
            f"Topic: {self.seed}\n\n"
            "Return ONLY a bullet list of questions."
        )
        print(f'[EXPLORER]::PRE_PROCESSING -> {self.seed}')
        response = ask_model(self.client, self.model, prompt)
        raw = response.message.content

        # Extract each line as a question
        for line in raw.splitlines():
            if "?" in line:
                q = line.strip("-• ").strip()
                if q:
                    self.sub_questions.append(q)

    # -----------------------------------------------------
    # STAGE 2: Determine where to explore (high-level)
    # -----------------------------------------------------
    def _suggest_sources(self, question: str) -> List[str]:
        prompt = (
            f'Given the question:\n"{question}"\n'
            "List the 5 most relevant external sources or domains to query to understand it.\n"
            "Examples: Wikipedia, academic journals, survey data, historical records, etc.\n"
            "Return a bullet list of source names only."
        )

        response = ask_model(self.client, self.model, prompt)
        raw = response.message.content
        print(f'[EXPLORER]::Mapping Topic')
        sources: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            src = line.strip("-• ").strip()
            if src:
                sources.append(src)
        return sources

    # -----------------------------------------------------
    # STAGE 3: Build knowledge nodes (structured ingestion)
    # -----------------------------------------------------
    def _ingest(self, question: str, sources: List[str]) -> Dict[str, Any]:
        """
        Ingest a question using the model's internal knowledge.

        Returns a structured node with:
        - summary (3-sentence synthesis)
        - key_concepts (list of strings)
        - unknowns (list of strings / controversies)
        """
        # Keep JSON template outside f-string to avoid brace issues
        json_template = """
{
  "QUERY": "<3-sentence synthesis>",
  "key_concepts": ["...", "..."],
  "unknowns": ["...", "..."]
}
"""

        prompt = (
            "You are Maxwell-2's Explorer.\n"
            "You will be given a question. Using your own knowledge (no web browsing), "
            "produce a structured description of what is currently known.\n\n"
            f"Question: {question}\n\n"
            "Return STRICT JSON in the following format:\n"
            f"{json_template}\n"
            "Do NOT add any extra keys or text outside the JSON."
        )
        print(f'[EXPLORER]::Determining Research Topics')
        response = ask_model(self.client, self.model, prompt)
        raw = response.message.content

        # Try to parse a JSON block out of the response
        parsed: Dict[str, Any]
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON object delimiters found in model output.")
            parsed = json.loads(raw[start : end + 1])
        except Exception:
            # Fallback: treat entire content as summary
            parsed = {
                "summary": raw.strip(),
                "key_concepts": [],
                "unknowns": [],
            }

        node: Dict[str, Any] = {
            "question": question,
            "sources": sources,
            "summary": parsed.get("summary", ""),
            "key_concepts": parsed.get("key_concepts", []),
            "unknowns": parsed.get("unknowns", []),
            "timestamp": time.time(),
        }
        return node

    # -----------------------------------------------------
    # STAGE 3.5: Build retrieval plans for a node
    # -----------------------------------------------------
    def _build_retrieval_plan_for_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given a node with summary, key_concepts, and unknowns,
        ask the model to generate structured search plans for
        each concept and each unknown/controversy.

        NOTE: This does NOT perform web retrieval.
        It only designs the search queries to be used later.
        """

        summary = node.get("summary", "")
        concepts = node.get("key_concepts", [])
        unknowns = node.get("unknowns", [])

        # JSON template kept out of the f-string
        plan_template = """
    {
      "concept": "<TOPIC>"      
    }
"""

        prompt = (
            "You are designing search queries to validate and enrich a conceptual map about a topic.\n\n"
            "SUMMARY OF CURRENT UNDERSTANDING:\n"
            f"{summary}\n\n"
            "KEY CONCEPTS (one per list item):\n"
            f"{json.dumps(concepts, indent=2)}\n\n"
            "UNKNOWN OR CONTROVERSIAL POINTS (one per list item):\n"
            f"{json.dumps(unknowns, indent=2)}\n\n"
            f"For each unknown/controversy, produce one topic to query for further research\n\n"
            "Return STRICT JSON in the following format:\n"
            f"{plan_template}\n"
            "Do NOT add extra commentary outside the JSON."
        )
        block = "\n\t-".join(concepts)
        print(f'[EXPLORER]::Refining Data Retrival for Concepts\n{block}')
        response = ask_model(self.client, self.model, prompt)
        raw = response.message.content

        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON object delimiters found in model output.")
            plan = json.loads(raw[start : end + 1])
        except Exception:
            plan = {"concept_plans": [], "issue_plans": []}
        return plan

    # -----------------------------------------------------
    # STAGE 4: Run full exploration
    # -----------------------------------------------------
    def explore(self) -> List[Dict[str, Any]]:
        """
        For each sub-question:
        - Suggest high-level source domains
        - Ingest structured knowledge (summary, concepts, unknowns)
        - Build a retrieval plan (search queries for future web use)
        """
        self.knowledge_nodes = []

        for q in self.sub_questions:
            sources = self._suggest_sources(q)
            node = self._ingest(q, sources)
            retrieval_plan = self._build_retrieval_plan_for_node(node)
            node["retrieval_plan"] = retrieval_plan
            self.knowledge_nodes.append(node)

        return self.knowledge_nodes

    # -----------------------------------------------------
    # JSON Export
    # -----------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "generated_questions": self.sub_questions,
                "nodes": self.knowledge_nodes,
            },
            indent=2
        )
