# maxwell.py
from evidence_engine import EvidenceEngine
from reflector import Reflector
from dotenv import load_dotenv
from explorer import Explorer
from datetime import datetime
from mapper import Mapper
import string
import time
import json
import os

load_dotenv()
URL = os.getenv("URL")


class Maxwell:
    def __init__(self, endpoint, model="gemma3:12b"):
        self.endpoint = endpoint
        self.model = model

    def explore_topic(self, topic: str):
        explorer = Explorer(seed_topic=topic, endpoint=self.endpoint, model=self.model)
        nodes = explorer.explore()
        # 2) External retrieval + evidence linking
        ev_engine = EvidenceEngine(endpoint=self.endpoint, model=self.model)
        enriched_nodes = [ev_engine.enrich_node(n) for n in nodes]
        return enriched_nodes

    def run(self, topic: str, out_file: str = None):
        data = self.explore_topic(topic)
        if out_file:
            with open(out_file, "w") as f:
                json.dump(data, f, indent=2)
        return data




def one_shot_exploration(subject: str, model: str)->"Reflector":
    # 1) Run Explorer
    exp = Explorer(seed_topic=subject, endpoint=URL, model=model)
    nodes = exp.explore()

    # 2) Package Explorer output in the standardized dict
    explorer_output = {
        "seed": exp.seed,
        "generated_questions": exp.sub_questions,
        "nodes": exp.knowledge_nodes,
    }

    # 3) Build the conceptual map
    mapper = Mapper(explorer_output)
    mapper.build_concept_index()
    mapper.build_graph()

    capsule = mapper.to_capsule_dict()
    reflector = Reflector(endpoint=URL, model=model, capsule=capsule)
    report = reflector.reflect()
    # reflector.save_report(f'{sanitized_filename(subject)}.json')
    with open(f'{sanitized_filename(subject)}.json', 'w') as f:
        f.write(json.dumps(capsule, indent=2))
    print(json.dumps(report, indent=2))
    return reflector


def open_ended_wonder(topic:str, model: str, limit:int=12)->"Reflector":
    exploring = True
    t0 = time.time()
    depth = 1
    out_folder = os.path.join(os.getcwd(), sanitized_filename(topic))
    if not os.path.isdir(out_folder): os.mkdir(out_folder)
    reflector = None
    considered = set()
    try:
        print(f'\t\t\t[EXPLORATION STARTED - {time.ctime(t0)}]\n')
        idea_pool = [topic]
        while exploring and depth < limit:
            if not idea_pool:
                print(f'\n** IDEA SPACE EXHAUSTED. EXITING **\n')
                exploring = False
                break
            print(f'='*80)
            print(f'\t\tDEPTH: {depth:02d}')
            print(f'=' * 80)
            # Begin exploration of topic
            seed = idea_pool.pop()
            print(f"[MAXWELL] Exploring: {seed}")
            exp = Explorer(seed_topic=seed, endpoint=URL, model=model)
            exp.explore()
            considered.add(seed)
            # 2) Package Explorer output in the standardized dict
            explorer_output = {
                "seed": exp.seed,
                "generated_questions": exp.sub_questions,
                "nodes": exp.knowledge_nodes,
            }

            # 3) Build the conceptual map
            mapper = Mapper(explorer_output)
            mapper.build_concept_index()
            mapper.build_graph()

            # 4) Reflect and distill what was learned, and what new questions were uncovered
            capsule = mapper.to_capsule_dict()
            reflector = Reflector(endpoint=URL, model=model, capsule=capsule)
            report = reflector.reflect()
            # save result
            out_file = f'{os.path.join(out_folder, sanitized_filename(seed[0:10]))}.json'
            # reflector.save_report(out_file)
            with open(out_file, "w") as f:
                f.write(json.dumps(capsule, indent=2))
            f.close()
            # now overwrite the current question with the new ones creates (or add then to a queue
            # prioritize the topics we understand least
            for challenging_idea in report['graph_insights']['undersupported_concepts']:
                if isinstance(challenging_idea, dict):
                    challenging_idea = challenging_idea.get("label", "")
                if challenging_idea and challenging_idea not in considered and not "":
                    idea_pool.append(f"What is {challenging_idea}?")
                    considered.add(challenging_idea.lower())
            # also add new questions
            for new_question in reflector.report['new_questions']['questions']:
                if new_question not in considered:
                    idea_pool.append(new_question)
                    # considered.add(new_question.lower())
            # increment depth counter
            depth += 1


    except KeyboardInterrupt:
        exploring = False
        dt = (time.time() - t0)/60.
        print(f'[X] TERMINATING MAXWELL EXPLORATION [{dt:.2f} minutes Elapsed]')
        pass
    return reflector


def make_timestamp():
    now = datetime.now()
    return f'{now.day:02d}{now.month:02d}{now.year:02d}'


def sanitized_filename(subject):
    safe_name = subject.replace(" ", "_")
    safe_chars = string.ascii_letters + '_'
    sanitized = ''
    for let in safe_name:
        if let in safe_chars:
            sanitized += let
    return f'{sanitized}_{make_timestamp()}' # it doesn't necessarily need to be timestamped I suppose


def main():
    open_ended = True
    model = 'gemma3:12b'
    seed = 'How does money hold value?'
    # seed = 'Over the course of human history has the slope of moral progress been positive or negative?'
    import sys
    if len(sys.argv) > 1:
        seed = ' '.join(sys.argv[1:])
    if not open_ended:
        one_shot_exploration(seed, model)
    else:
        if os.path.isfile('questions.txt'):
            questions = open('questions.txt').read().splitlines()
            for seed in questions:
                open_ended_wonder(seed, model,limit=18)


if __name__ == '__main__':
    main()
