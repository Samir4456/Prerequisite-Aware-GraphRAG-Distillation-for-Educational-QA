"""
MetaQA EDA utilities.

The functions here power both the notebook and any script-based analysis.
They intentionally avoid hard-coded absolute paths so the project can run
from a cloned repo or from the current Windows workspace.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns


HOP_DIR_NAMES = {
    1: ("1-hop", "1hop"),
    2: ("2-hop", "2hop"),
    3: ("3-hop", "3hop"),
}


@dataclass(frozen=True)
class MetaQAPaths:
    raw_dir: Path
    kb_path: Path
    entity_path: Path | None


def default_raw_dir() -> Path:
    return Path("src/data/raw")


def resolve_paths(raw_dir: str | Path = default_raw_dir()) -> MetaQAPaths:
    raw = Path(raw_dir)
    kb_path = raw / "kb.txt"
    entity_path = raw / "entity" / "kb_entity_dict.txt"
    return MetaQAPaths(
        raw_dir=raw,
        kb_path=kb_path,
        entity_path=entity_path if entity_path.exists() else None,
    )


def hop_dir(raw_dir: str | Path, hop: int) -> Path:
    raw = Path(raw_dir)
    for name in HOP_DIR_NAMES[hop]:
        candidate = raw / name
        if candidate.exists():
            return candidate
    return raw / HOP_DIR_NAMES[hop][0]


def load_kb(kb_path: str | Path) -> list[tuple[str, str, str]]:
    triples = []
    with open(kb_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def load_entities(entity_path: str | Path | None) -> pd.DataFrame:
    if not entity_path:
        return pd.DataFrame(columns=["entity_id", "entity"])

    rows = []
    with open(entity_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                rows.append({"entity_id": parts[0], "entity": parts[1]})
            elif len(parts) == 1 and parts[0]:
                rows.append({"entity_id": len(rows), "entity": parts[0]})
    return pd.DataFrame(rows)


def extract_topic_entity(question: str) -> str | None:
    start = question.find("[")
    end = question.find("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return question[start + 1 : end]


def load_qa_file(path: str | Path, hop: int, split: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            question, raw_answers = parts
            answers = [ans for ans in raw_answers.split("|") if ans]
            rows.append(
                {
                    "hop": hop,
                    "split": split,
                    "row_id": idx,
                    "question": question,
                    "answers": answers,
                    "answer_count": len(answers),
                    "question_length": len(question.replace("[", "").replace("]", "").split()),
                    "topic_entity": extract_topic_entity(question),
                }
            )
    return pd.DataFrame(rows)


def load_all_qa(raw_dir: str | Path) -> pd.DataFrame:
    frames = []
    for hop in (1, 2, 3):
        directory = hop_dir(raw_dir, hop)
        for split in ("train", "dev", "test"):
            path = directory / f"qa_{split}.txt"
            if path.exists():
                frames.append(load_qa_file(path, hop=hop, split=split))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_graph(triples: list[tuple[str, str, str]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for subj, rel, obj in triples:
        graph.add_edge(subj, obj, relation=rel)
    return graph


def dataset_summary(qa_df: pd.DataFrame) -> pd.DataFrame:
    return (
        qa_df.groupby(["hop", "split"])
        .agg(
            questions=("question", "count"),
            avg_answers=("answer_count", "mean"),
            median_answers=("answer_count", "median"),
            avg_question_words=("question_length", "mean"),
            unique_topic_entities=("topic_entity", "nunique"),
        )
        .reset_index()
    )


def graph_summary(graph: nx.DiGraph, triples: list[tuple[str, str, str]]) -> dict:
    weak_components = list(nx.weakly_connected_components(graph))
    relation_counts = Counter(rel for _, rel, _ in triples)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "triples": len(triples),
        "relations": len(relation_counts),
        "density": nx.density(graph),
        "weakly_connected_components": len(weak_components),
        "largest_component_nodes": max((len(c) for c in weak_components), default=0),
        "avg_total_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        if graph.number_of_nodes()
        else 0,
    }


def degree_dataframe(graph: nx.DiGraph) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity": list(graph.nodes()),
            "degree": [graph.degree(node) for node in graph.nodes()],
            "in_degree": [graph.in_degree(node) for node in graph.nodes()],
            "out_degree": [graph.out_degree(node) for node in graph.nodes()],
        }
    )


def relation_dataframe(triples: list[tuple[str, str, str]]) -> pd.DataFrame:
    counts = Counter(rel for _, rel, _ in triples)
    return pd.DataFrame(
        [{"relation": relation, "count": count} for relation, count in counts.items()]
    ).sort_values("count", ascending=False)


def topic_entity_frequency(qa_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    return (
        qa_df["topic_entity"]
        .value_counts(dropna=True)
        .head(top_n)
        .rename_axis("topic_entity")
        .reset_index(name="count")
    )


def answer_count_distribution(qa_df: pd.DataFrame) -> pd.DataFrame:
    return (
        qa_df["answer_count"]
        .value_counts()
        .sort_index()
        .rename_axis("answer_count")
        .reset_index(name="questions")
    )


def infer_question_type(question: str) -> str:
    q = question.lower()
    if "directed" in q or "director" in q:
        return "director"
    if "star" in q or "actor" in q or "actress" in q:
        return "actor"
    if "genre" in q:
        return "genre"
    if "language" in q:
        return "language"
    if "written" in q or "writer" in q:
        return "writer"
    if "year" in q or "released" in q:
        return "release_year"
    if "rating" in q:
        return "rating"
    if "tag" in q:
        return "tag"
    return "other"


def question_type_distribution(qa_df: pd.DataFrame) -> pd.DataFrame:
    typed = qa_df.copy()
    typed["question_type"] = typed["question"].map(infer_question_type)
    return (
        typed.groupby(["hop", "question_type"])
        .size()
        .reset_index(name="questions")
        .sort_values(["hop", "questions"], ascending=[True, False])
    )


def sample_shortest_paths(graph: nx.DiGraph, sample_size: int = 150, seed: int = 7) -> pd.DataFrame:
    nodes = list(graph.nodes())
    rng = random.Random(seed)
    sample = rng.sample(nodes, min(sample_size, len(nodes)))
    undirected = graph.to_undirected()
    lengths = []

    for i, source in enumerate(sample):
        for target in sample[i + 1 :]:
            try:
                lengths.append(nx.shortest_path_length(undirected, source, target))
            except nx.NetworkXNoPath:
                continue

    return pd.DataFrame({"shortest_path_length": lengths})


def relation_cooccurrence_from_questions(qa_df: pd.DataFrame) -> pd.DataFrame:
    """Approximate question relation demand from keywords in question text."""
    relation_keywords = {
        "directed_by": ("directed", "director"),
        "starred_actors": ("star", "actor", "actress"),
        "has_genre": ("genre",),
        "in_language": ("language",),
        "written_by": ("written", "writer"),
        "release_year": ("year", "released"),
        "has_imdb_rating": ("rating",),
        "has_tags": ("tag",),
    }
    rows = []
    for _, row in qa_df.iterrows():
        q = row["question"].lower()
        mentioned = [
            rel
            for rel, keywords in relation_keywords.items()
            if any(keyword in q for keyword in keywords)
        ]
        for rel in mentioned:
            rows.append({"hop": row["hop"], "relation_hint": rel})
    return pd.DataFrame(rows)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.rcParams["axes.titleweight"] = "bold"


def plot_question_counts(qa_df: pd.DataFrame):
    fig, ax = plt.subplots()
    counts = qa_df.groupby(["hop", "split"]).size().reset_index(name="questions")
    sns.barplot(data=counts, x="hop", y="questions", hue="split", ax=ax)
    ax.set_title("MetaQA Question Counts by Hop and Split")
    ax.set_xlabel("Hop depth")
    ax.set_ylabel("Questions")
    return fig


def plot_answer_counts(qa_df: pd.DataFrame):
    fig, ax = plt.subplots()
    sns.histplot(data=qa_df, x="answer_count", hue="hop", multiple="stack", bins=30, ax=ax)
    ax.set_title("Answer Count Distribution")
    ax.set_xlabel("Number of valid answers")
    return fig


def plot_question_lengths(qa_df: pd.DataFrame):
    fig, ax = plt.subplots()
    sns.histplot(data=qa_df, x="question_length", hue="hop", bins=40, kde=True, ax=ax)
    ax.set_title("Question Length Distribution")
    ax.set_xlabel("Question words")
    return fig


def plot_top_topic_entities(qa_df: pd.DataFrame, top_n: int = 20):
    fig, ax = plt.subplots(figsize=(10, 7))
    top = topic_entity_frequency(qa_df, top_n=top_n)
    sns.barplot(data=top, y="topic_entity", x="count", ax=ax)
    ax.set_title(f"Top {top_n} Topic Entities")
    ax.set_xlabel("Question count")
    ax.set_ylabel("")
    return fig


def plot_relation_counts(rel_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=rel_df, y="relation", x="count", ax=ax)
    ax.set_title("Knowledge Graph Relation Frequency")
    ax.set_xlabel("Triple count")
    ax.set_ylabel("")
    return fig


def plot_degree_distribution(degrees: pd.DataFrame):
    fig, ax = plt.subplots()
    positive = degrees[degrees["degree"] > 0]
    sns.histplot(positive["degree"], bins=80, log_scale=(True, True), ax=ax)
    ax.set_title("Log-Log Entity Degree Distribution")
    ax.set_xlabel("Total degree")
    ax.set_ylabel("Entity count")
    return fig


def plot_top_degree_entities(degrees: pd.DataFrame, degree_col: str = "degree", top_n: int = 20):
    fig, ax = plt.subplots(figsize=(10, 7))
    top = degrees.sort_values(degree_col, ascending=False).head(top_n)
    sns.barplot(data=top, y="entity", x=degree_col, ax=ax)
    ax.set_title(f"Top {top_n} Entities by {degree_col.replace('_', ' ').title()}")
    ax.set_xlabel(degree_col.replace("_", " ").title())
    ax.set_ylabel("")
    return fig


def plot_in_vs_out_degree(degrees: pd.DataFrame):
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=degrees,
        x="out_degree",
        y="in_degree",
        size="degree",
        sizes=(10, 200),
        alpha=0.45,
        legend=False,
        ax=ax,
    )
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_title("In-Degree vs Out-Degree by Entity")
    return fig


def plot_shortest_paths(path_df: pd.DataFrame):
    fig, ax = plt.subplots()
    if len(path_df):
        sns.histplot(data=path_df, x="shortest_path_length", discrete=True, ax=ax)
    ax.set_title("Sampled Shortest Path Lengths")
    ax.set_xlabel("Shortest path length")
    return fig


def plot_question_types(type_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=type_df, x="question_type", y="questions", hue="hop", ax=ax)
    ax.set_title("Approximate Question Type Distribution")
    ax.set_xlabel("Question type")
    ax.tick_params(axis="x", rotation=35)
    return fig


def analyze(raw_dir: str | Path = default_raw_dir()) -> dict:
    paths = resolve_paths(raw_dir)
    triples = load_kb(paths.kb_path)
    graph = build_graph(triples)
    qa_df = load_all_qa(paths.raw_dir)
    entities_df = load_entities(paths.entity_path)
    degrees = degree_dataframe(graph)
    relations = relation_dataframe(triples)

    return {
        "paths": paths,
        "triples": triples,
        "graph": graph,
        "qa_df": qa_df,
        "entities_df": entities_df,
        "degrees": degrees,
        "relations": relations,
        "dataset_summary": dataset_summary(qa_df),
        "graph_summary": graph_summary(graph, triples),
        "answer_distribution": answer_count_distribution(qa_df),
        "topic_entities": topic_entity_frequency(qa_df),
        "question_types": question_type_distribution(qa_df),
        "sampled_paths": sample_shortest_paths(graph),
    }
