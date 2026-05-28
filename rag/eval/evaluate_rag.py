from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from statistics import mean
from typing import Callable

from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from pydantic import BaseModel, Field

from rag.service.rag_agent import MangaRagAgent


class JudgeScores(BaseModel):
    RelevanceScore: int = Field(ge=0, le=10)
    HallucinationScore: int = Field(ge=0, le=10)


@dataclass(frozen=True)
class TitleSeed:
    title: str
    descriptor: str
    keyword: str


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    medium: str
    method: str
    query: str
    target: str


ANIME_SEEDS: list[TitleSeed] = [
    TitleSeed("Death Note", "a student finds a notebook that kills people", "Light Yagami"),
    TitleSeed("Attack on Titan", "humans hide behind walls from giant humanoids", "Eren Yeager"),
    TitleSeed("One Piece", "a pirate crew searches for legendary treasure", "Monkey D. Luffy"),
    TitleSeed("Naruto", "an outcast ninja wants to become the hokage", "Naruto Uzumaki"),
    TitleSeed("Fullmetal Alchemist: Brotherhood", "two brothers seek the philosopher stone", "Edward Elric"),
    TitleSeed("Demon Slayer", "a swordsman fights demons after his family is killed", "Tanjiro Kamado"),
    TitleSeed("Jujutsu Kaisen", "students exorcise curses at a sorcery school", "Satoru Gojo"),
    TitleSeed("Hunter x Hunter", "a boy becomes a hunter to find his father", "Gon Freecss"),
    TitleSeed("Steins;Gate", "friends discover time travel through a microwave", "Rintaro Okabe"),
    TitleSeed("Code Geass", "an exiled prince gains the power of absolute command", "Lelouch Lamperouge"),
    TitleSeed("Cowboy Bebop", "space bounty hunters chase targets across planets", "Spike Spiegel"),
    TitleSeed("Neon Genesis Evangelion", "teens pilot biomechanical mechs against angels", "Shinji Ikari"),
    TitleSeed("My Hero Academia", "a quirkless boy enters superhero school", "Izuku Midoriya"),
    TitleSeed("Bleach", "a teenager becomes a soul reaper", "Ichigo Kurosaki"),
    TitleSeed("Chainsaw Man", "a devil hunter fuses with a chainsaw devil", "Denji"),
    TitleSeed("Tokyo Ghoul", "a college student becomes half ghoul", "Ken Kaneki"),
    TitleSeed("Re:ZERO -Starting Life in Another World-", "a transported boy resets by dying", "Subaru Natsuki"),
    TitleSeed("Vinland Saga", "a young viking seeks revenge in wartime", "Thorfinn"),
    TitleSeed("Frieren: Beyond Journey's End", "an elf mage reflects after the hero's death", "Frieren"),
    TitleSeed("SPY x FAMILY", "a spy builds a fake family with an assassin", "Loid Forger"),
]

MANGA_SEEDS: list[TitleSeed] = [
    TitleSeed("Berserk", "a black swordsman survives in a brutal dark fantasy world", "Guts"),
    TitleSeed("Vagabond", "a wandering swordsman chases invincibility", "Miyamoto Musashi"),
    TitleSeed("Kingdom", "an orphan wants to become a great general in ancient china", "Xin"),
    TitleSeed("Monster", "a surgeon hunts a serial killer he once saved", "Kenzo Tenma"),
    TitleSeed("Goodnight Punpun", "a surreal coming of age story with heavy depression themes", "Punpun"),
    TitleSeed("20th Century Boys", "old friends face a cult tied to their childhood game", "Kenji Endo"),
    TitleSeed("Tokyo Revengers", "a loser time travels to save his ex from gang violence", "Takemichi Hanagaki"),
    TitleSeed("Solo Leveling", "the weakest hunter gains a unique leveling system", "Sung Jin-Woo"),
    TitleSeed("The Climber", "a lonely student becomes obsessed with mountain climbing", "Buntaro Mori"),
    TitleSeed("Real", "a grounded basketball story about athletes with disabilities", "Tomomi Nomiya"),
    TitleSeed("Claymore", "half demon women hunt monsters with giant swords", "Clare"),
    TitleSeed("Gantz", "dead people are forced to hunt aliens in black suits", "Kei Kurono"),
    TitleSeed("Dorohedoro", "an amnesiac reptile-headed man hunts sorcerers", "Caiman"),
    TitleSeed("Golden Kamuy", "a soldier and an ainu girl search for hidden gold", "Saichi Sugimoto"),
    TitleSeed("Haikyuu!!", "a short volleyball player aims for nationals", "Shoyo Hinata"),
    TitleSeed("Blue Lock", "strikers compete in an extreme football training prison", "Yoichi Isagi"),
    TitleSeed("Slam Dunk", "a delinquent joins high school basketball", "Hanamichi Sakuragi"),
    TitleSeed("Kaguya-sama: Love Is War", "genius student council members play mind games in romance", "Kaguya Shinomiya"),
    TitleSeed("The Apothecary Diaries", "a palace servant solves poison and medicine mysteries", "Maomao"),
    TitleSeed("Oshi no Ko", "idol industry drama tied to revenge and reincarnation", "Aqua Hoshino"),
]


SEMANTIC_TEMPLATES = [
    "Find a manga or anime about {descriptor}.",
    "What is the title where {descriptor}?",
    "I want something with this vibe: {descriptor}.",
    "Resolve this story to a title: {descriptor}.",
    "Name the series about {descriptor}.",
]

KEYWORD_TEMPLATES = [
    "Series with character {keyword}.",
    "Find title: {title}.",
    "Which manga features {keyword}?",
    "I want {title}.",
    "Show me info about {title}.",
]

CACHE_TEMPLATES = [
    "{title}",
    "Please fetch {title}.",
    "Exact title lookup: {title}",
    "{title} manga",
    "MAL lookup {title}",
]


def build_agent() -> MangaRagAgent:
    index_name = os.environ.get("PINECONE_INDEX", "rag-jit-cache").strip() or "rag-jit-cache"
    return MangaRagAgent(
        pinecone_index=index_name,
        pinecone_namespace=os.environ.get("PINECONE_NAMESPACE", "manga-rag"),
        bm25_path=os.environ.get("BM25_VALUES_PATH", "rag/ingestion/bm25_values.json"),
        top_k=int(os.environ.get("RAG_TOP_K", "10")),
        score_threshold=float(os.environ.get("RAG_SCORE_THRESHOLD", "0.75")),
        recursion_limit=int(os.environ.get("RAG_RECURSION_LIMIT", "8")),
        cerebras_model=os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b"),
    )


def build_judge():
    judge_llm = ChatCerebras(
        model=os.environ.get("CEREBRAS_JUDGE_MODEL", os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")),
        api_key=os.environ["CEREBRAS_API_KEY"],
        temperature=0,
    )
    return judge_llm.with_structured_output(JudgeScores)


def _build_cases_for_method(
    *,
    method: str,
    medium: str,
    seeds: list[TitleSeed],
    templates: list[str],
    target_count: int,
    formatter: Callable[[str, TitleSeed], str] | None = None,
) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    idx = 0
    while len(cases) < target_count:
        seed = seeds[idx % len(seeds)]
        template = templates[(idx // len(seeds)) % len(templates)]
        raw_query = template.format(
            title=seed.title,
            descriptor=seed.descriptor,
            keyword=seed.keyword,
        )
        query = formatter(raw_query, seed) if formatter else raw_query
        cases.append(
            GoldenCase(
                case_id=f"{method}-{medium}-{len(cases)+1:03d}",
                medium=medium,
                method=method,
                query=query,
                target=seed.title,
            )
        )
        idx += 1
    return cases


def build_semantic_suite() -> list[GoldenCase]:
    return [
        *_build_cases_for_method(
            method="semantic",
            medium="anime",
            seeds=ANIME_SEEDS,
            templates=SEMANTIC_TEMPLATES,
            target_count=100,
        ),
        *_build_cases_for_method(
            method="semantic",
            medium="manga",
            seeds=MANGA_SEEDS,
            templates=SEMANTIC_TEMPLATES,
            target_count=100,
        ),
    ]


def build_keyword_suite() -> list[GoldenCase]:
    return [
        *_build_cases_for_method(
            method="keyword",
            medium="anime",
            seeds=ANIME_SEEDS,
            templates=KEYWORD_TEMPLATES,
            target_count=100,
        ),
        *_build_cases_for_method(
            method="keyword",
            medium="manga",
            seeds=MANGA_SEEDS,
            templates=KEYWORD_TEMPLATES,
            target_count=100,
        ),
    ]


def build_cache_hit_suite() -> list[GoldenCase]:
    return [
        *_build_cases_for_method(
            method="cache_hit",
            medium="anime",
            seeds=ANIME_SEEDS,
            templates=CACHE_TEMPLATES,
            target_count=100,
        ),
        *_build_cases_for_method(
            method="cache_hit",
            medium="manga",
            seeds=MANGA_SEEDS,
            templates=CACHE_TEMPLATES,
            target_count=100,
        ),
    ]


def run_suite(
    *,
    suite_name: str,
    cases: list[GoldenCase],
    agent: MangaRagAgent,
    judge,
    limit: int | None = None,
) -> None:
    active_cases = cases[:limit] if limit is not None else cases
    print(f"\n=== {suite_name} ({len(active_cases)} / {len(cases)}) ===")

    relevance_scores: list[int] = []
    hallucination_scores: list[int] = []
    resolve_success = 0
    cache_hit_success = 0

    for case in active_cases:
        start = time.perf_counter()
        response = agent.query(case.query)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        target_norm = case.target.lower().strip()
        resolved_norm = (response.resolved_title or "").lower().strip()
        if target_norm and resolved_norm and target_norm in resolved_norm:
            resolve_success += 1

        if case.method == "cache_hit":
            warmup = agent.query(case.query)
            second = agent.query(case.query)
            if (
                warmup.retrieval_mode in {"live_jikan", "cache_hit"}
                and second.retrieval_mode == "cache_hit"
            ):
                cache_hit_success += 1

        scores = judge.invoke(
            (
                "Score this resolve-then-fetch RAG answer.\n"
                f"Method: {case.method}\n"
                f"Medium: {case.medium}\n"
                f"User query: {case.query}\n"
                f"Expected canonical title: {case.target}\n"
                f"Resolved title: {response.resolved_title}\n"
                f"Answer: {response.answer}\n"
                "RelevanceScore 0-10: 10 means exact and useful.\n"
                "HallucinationScore 0-10: 10 means very hallucinated."
            )
        )

        relevance_scores.append(scores.RelevanceScore)
        hallucination_scores.append(scores.HallucinationScore)

        print(
            f"[{case.case_id}] mode={response.retrieval_mode} "
            f"resolved={response.resolved_title!r} "
            f"results={len(response.results)} "
            f"rel={scores.RelevanceScore}/10 "
            f"hall={scores.HallucinationScore}/10 "
            f"time={elapsed_ms}ms"
        )

    print("\nSuite Summary")
    print(f"- Cases evaluated: {len(active_cases)}")
    print(f"- Resolve success (target contained in resolved title): {resolve_success}/{len(active_cases)}")
    if active_cases:
        print(f"- Avg RelevanceScore: {mean(relevance_scores):.2f}/10")
        print(f"- Avg HallucinationScore: {mean(hallucination_scores):.2f}/10")
    if suite_name == "cache_hit":
        print(f"- Cache-hit second-call success: {cache_hit_success}/{len(active_cases)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extensive resolve->fetch RAG evaluation suite.")
    parser.add_argument(
        "--suite",
        choices=["all", "semantic", "keyword", "cache_hit"],
        default="all",
        help="Which suite to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap per suite for quicker local checks.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    agent = build_agent()
    judge = build_judge()

    suites: list[tuple[str, list[GoldenCase]]] = []
    if args.suite in {"all", "semantic"}:
        suites.append(("semantic", build_semantic_suite()))
    if args.suite in {"all", "keyword"}:
        suites.append(("keyword", build_keyword_suite()))
    if args.suite in {"all", "cache_hit"}:
        suites.append(("cache_hit", build_cache_hit_suite()))

    for suite_name, suite_cases in suites:
        run_suite(
            suite_name=suite_name,
            cases=suite_cases,
            agent=agent,
            judge=judge,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
