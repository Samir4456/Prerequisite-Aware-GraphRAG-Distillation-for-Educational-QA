"""
Knowledge Graph Construction — CLRS
=====================================
Step 1: Extract concepts + prerequisites from each chunk (Llama 3.1 8B)
Step 2: Build a directed prerequisite graph with NetworkX
Step 3: Run EDA — degree distribution, longest chains, bottlenecks, clusters
Step 4: Save graph as JSON + GEXF (for Gephi) + pyvis HTML (interactive)
 
Usage:
    ollama serve &
    python knowledge_graph.py            # full pipeline
    python knowledge_graph.py --eda-only # skip extraction, just re-run EDA
 
Requirements:
    pip install ollama networkx pyvis python-louvain tqdm
"""
 
import argparse
import json
import re
import time
from collections import defaultdict, Counter
from pathlib import Path
 
import networkx as nx
from tqdm import tqdm
 
try:
    import ollama
except ImportError:
    raise ImportError("pip install ollama")
 
try:
    from pyvis.network import Network
except ImportError:
    raise ImportError("pip install pyvis")
 
try:
    import community as community_louvain
except ImportError:
    raise ImportError("pip install python-louvain")
 
 
# ─── Config ──────────────────────────────────────────────────────────────────
 
CHUNKS_PATH      = Path("data/processed/clrs_chunks.json")
CONCEPTS_PATH    = Path("data/processed/clrs_concepts.json")   # raw extraction cache
GRAPH_JSON_PATH  = Path("data/processed/clrs_graph.json")      # node/edge lists
GRAPH_GEXF_PATH  = Path("data/processed/clrs_graph.gexf")      # Gephi format
GRAPH_HTML_PATH  = Path("data/processed/clrs_graph.html")      # pyvis interactive
EDA_PATH         = Path("data/processed/clrs_graph_eda.json")  # EDA stats
 
MODEL            = "llama3.1:8b"
MAX_CHUNKS       = None      # set to 200 for a quick test run, None for full
SLEEP_BETWEEN    = 0.1
MAX_JSON_RETRIES = 2
 
# Minimum times a concept must appear to be kept as a node
# (filters noise like "example" or "figure 6.3")
MIN_CONCEPT_FREQ = 2
 
 
# ─── Prompt ──────────────────────────────────────────────────────────────────
 
EXTRACTION_SYSTEM = (
    "You are a precise JSON-generating assistant. "
    "You only output valid JSON. Never add explanation or markdown."
)
 
EXTRACTION_PROMPT = """Read this passage from the CLRS algorithms textbook.
Extract the key CS concepts and their prerequisite relationships.
 
Rules:
- concept names must be short (1-4 words), lowercase, specific
- prerequisites: concepts the reader must already understand to grasp this one
- enables: concepts that this one directly unlocks or leads to
- only include concepts actually discussed in this passage
- aim for 3-8 concepts per passage
 
Passage:
\"\"\"{text}\"\"\"
 
Return ONLY this JSON, nothing else:
{{
  "concepts": [
    {{
      "name": "concept name",
      "definition": "one sentence from passage",
      "prerequisites": ["concept a", "concept b"],
      "enables": ["concept c"]
    }}
  ]
}}"""
 
 
# ─── Ligature cleanup (same as qa_generation_llama.py) ───────────────────────
 
LIGATURE_MAP = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
}
 
def clean_ligatures(text: str) -> str:
    for lig, rep in LIGATURE_MAP.items():
        text = text.replace(lig, rep)
    return text
 
 
def clean_str(s: str) -> str:
    """Remove XML-illegal control characters (U+0000–U+001F except tab/newline/CR)."""
    return re.sub(r'[--]', '', s).strip()
 
 
# ─── Ollama helpers ───────────────────────────────────────────────────────────
 
def parse_json(raw: str) -> dict | None:
    if not raw:
        return None
    raw = raw.strip()
    if "```" in raw:
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
        if m:
            raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{[\s\S]*\}', raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None
 
 
def call_llama(prompt: str, system: str) -> dict | None:
    time.sleep(SLEEP_BETWEEN)
    for attempt in range(MAX_JSON_RETRIES + 1):
        try:
            resp = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                options={"temperature": 0.3, "num_ctx": 3072, "num_predict": 600},
            )
            raw = resp["message"]["content"]
            parsed = parse_json(raw)
            if parsed:
                return parsed
            if attempt < MAX_JSON_RETRIES:
                print(f"  Bad JSON, retry {attempt+1}...")
        except Exception as e:
            print(f"  Ollama error: {e}")
            if attempt < MAX_JSON_RETRIES:
                time.sleep(2)
    return None
 
 
# ─── Step 1: Concept extraction ───────────────────────────────────────────────
 
def extract_concepts(chunks: list[dict]) -> list[dict]:
    """Run LLM extraction on every chunk. Returns list of per-chunk results."""
 
    # Load existing cache so we can resume
    if CONCEPTS_PATH.exists():
        with open(CONCEPTS_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        done_ids = {c["chunk_id"] for c in cache}
        print(f"  Resuming: {len(cache)} chunks already extracted")
    else:
        cache = []
        done_ids = set()
 
    todo = [c for c in chunks if c["chunk_id"] not in done_ids]
    print(f"  Extracting concepts from {len(todo)} chunks...")
 
    for chunk in tqdm(todo, desc="Extracting"):
        text = clean_ligatures(chunk["text"])[:2000]
        prompt = EXTRACTION_PROMPT.format(text=text)
        result = call_llama(prompt, EXTRACTION_SYSTEM)
 
        concepts = []
        if result and "concepts" in result:
            for c in result["concepts"]:
                name = clean_str(str(c.get("name", ""))).lower()
                if not name or len(name) > 60:
                    continue
                concepts.append({
                    "name":          name,
                    "definition":    clean_str(c.get("definition", ""))[:200],
                    "prerequisites": [clean_str(p).lower() for p in c.get("prerequisites", []) if isinstance(p, str) and p.strip()],
                    "enables":       [clean_str(e).lower() for e in c.get("enables", []) if isinstance(e, str) and e.strip()],
                })
 
        cache.append({
            "chunk_id": chunk["chunk_id"],
            "page":     chunk["page_number"],
            "chapter":  chunk["chapter"],
            "concepts": concepts,
        })
 
        # Save every 25 chunks
        if len(cache) % 25 == 0:
            CONCEPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONCEPTS_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)
 
    CONCEPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONCEPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
 
    print(f"  Saved concept extractions to {CONCEPTS_PATH}")
    return cache
 
 
# ─── Step 2: Build NetworkX graph ────────────────────────────────────────────
 
def build_graph(extractions: list[dict]) -> nx.DiGraph:
    """
    Nodes  = unique concepts
    Edges  = prerequisite relationships  (A → B means A is needed to understand B)
    """
    G = nx.DiGraph()
 
    # Count concept frequencies to filter noise
    freq: Counter = Counter()
    for ex in extractions:
        for c in ex["concepts"]:
            freq[c["name"]] += 1
            for p in c["prerequisites"]:
                freq[p] += 1
            for e in c["enables"]:
                freq[e] += 1
 
    kept = {name for name, count in freq.items() if count >= MIN_CONCEPT_FREQ}
    print(f"  Unique concepts extracted : {len(freq)}")
    print(f"  Kept after freq filter (>={MIN_CONCEPT_FREQ}): {len(kept)}")
 
    # Add nodes with metadata
    concept_meta: dict[str, dict] = {}
    for ex in extractions:
        for c in ex["concepts"]:
            name = c["name"]
            if name not in kept:
                continue
            if name not in concept_meta:
                concept_meta[name] = {
                    "definition": c["definition"],
                    "first_page": ex["page"],
                    "chapter":    ex["chapter"],
                    "frequency":  freq[name],
                }
            G.add_node(name, **concept_meta[name])
 
    # Add edges
    edge_counts: Counter = Counter()
    for ex in extractions:
        for c in ex["concepts"]:
            if c["name"] not in kept:
                continue
            # prerequisites → concept (prereq must come before)
            for prereq in c["prerequisites"]:
                if prereq in kept and prereq != c["name"]:
                    edge_counts[(prereq, c["name"])] += 1
            # concept → enables
            for enables in c["enables"]:
                if enables in kept and enables != c["name"]:
                    edge_counts[(c["name"], enables)] += 1
 
    for (src, dst), weight in edge_counts.items():
        if G.has_node(src) and G.has_node(dst):
            G.add_edge(src, dst, weight=weight)
 
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    return G
 
 
# ─── Step 3: EDA ─────────────────────────────────────────────────────────────
 
def run_eda(G: nx.DiGraph) -> dict:
    print("\n[EDA] Running graph analysis...")
    stats = {}
 
    # Basic counts
    stats["nodes"] = G.number_of_nodes()
    stats["edges"] = G.number_of_edges()
    stats["density"] = round(nx.density(G), 6)
 
    # Degree stats
    in_degrees  = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
 
    stats["avg_in_degree"]  = round(sum(in_degrees.values()) / max(len(in_degrees), 1), 2)
    stats["avg_out_degree"] = round(sum(out_degrees.values()) / max(len(out_degrees), 1), 2)
 
    # Top concepts by in-degree (most depended upon = foundational)
    top_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:15]
    stats["top_foundational_concepts"] = top_in
    print("\n  Top foundational concepts (highest in-degree):")
    for name, deg in top_in[:10]:
        print(f"    {name:40s}  in-degree={deg}")
 
    # Top concepts by out-degree (most enabling = gateway concepts)
    top_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:15]
    stats["top_gateway_concepts"] = top_out
    print("\n  Top gateway concepts (highest out-degree):")
    for name, deg in top_out[:10]:
        print(f"    {name:40s}  out-degree={deg}")
 
    # Longest prerequisite chains (longest paths in DAG)
    # Work on a DAG version (remove cycles if any)
    try:
        dag = nx.DiGraph()
        dag.add_nodes_from(G.nodes(data=True))
        # Add edges only if they don't create a cycle
        for u, v, d in G.edges(data=True):
            dag.add_edge(u, v, **d)
            if not nx.is_directed_acyclic_graph(dag):
                dag.remove_edge(u, v)
 
        # Find top-10 longest paths (approximate — exact is NP-hard on large graphs)
        longest_paths = []
        roots = [n for n in dag.nodes() if dag.in_degree(n) == 0]
        for root in roots[:50]:  # sample roots to keep it fast
            try:
                path = nx.dag_longest_path(dag)
                if len(path) > 2:
                    longest_paths.append(path)
            except Exception:
                pass
 
        # Use the overall longest path
        try:
            overall_longest = nx.dag_longest_path(dag)
            stats["longest_chain_length"] = len(overall_longest)
            stats["longest_chain"] = overall_longest
            print(f"\n  Longest prerequisite chain ({len(overall_longest)} steps):")
            print("    " + " → ".join(overall_longest[:8]) + ("..." if len(overall_longest) > 8 else ""))
        except Exception as e:
            stats["longest_chain_length"] = 0
            stats["longest_chain"] = []
            print(f"  Could not compute longest chain: {e}")
 
    except Exception as e:
        print(f"  DAG analysis skipped: {e}")
 
    # Bottleneck nodes (high betweenness centrality)
    print("\n  Computing betweenness centrality (may take a moment)...")
    try:
        betweenness = nx.betweenness_centrality(G, normalized=True)
        top_between = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
        stats["top_bottleneck_concepts"] = top_between
        print("  Top bottleneck concepts (betweenness centrality):")
        for name, score in top_between[:8]:
            print(f"    {name:40s}  centrality={score:.4f}")
    except Exception as e:
        stats["top_bottleneck_concepts"] = []
        print(f"  Betweenness skipped: {e}")
 
    # Community detection (Louvain on undirected version)
    print("\n  Detecting topic clusters (Louvain)...")
    try:
        G_undirected = G.to_undirected()
        partition = community_louvain.best_partition(G_undirected)
        num_communities = len(set(partition.values()))
        stats["num_communities"] = num_communities
 
        # Group nodes by community
        communities: dict[int, list] = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
 
        # Sort communities by size, show top 5
        top_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        stats["top_communities"] = [
            {"id": cid, "size": len(nodes), "sample": nodes[:5]}
            for cid, nodes in top_communities
        ]
        print(f"  Found {num_communities} topic clusters")
        for cid, nodes in top_communities:
            print(f"    Cluster {cid} ({len(nodes)} concepts): {', '.join(nodes[:5])}")
 
        # Add community label to nodes for visualization
        nx.set_node_attributes(G, partition, "community")
 
    except Exception as e:
        stats["num_communities"] = 0
        print(f"  Community detection skipped: {e}")
 
    # Weakly connected components
    wcc = list(nx.weakly_connected_components(G))
    stats["num_weakly_connected_components"] = len(wcc)
    stats["largest_component_size"] = max(len(c) for c in wcc) if wcc else 0
    print(f"\n  Weakly connected components: {len(wcc)}")
    print(f"  Largest component size: {stats['largest_component_size']} nodes")
 
    # Save EDA
    # Convert tuples to lists for JSON serialisation
    def jsonify(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [jsonify(i) for i in obj]
        return obj
 
    EDA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EDA_PATH, "w", encoding="utf-8") as f:
        json.dump(jsonify(stats), f, indent=2, ensure_ascii=False)
    print(f"\n  EDA saved to {EDA_PATH}")
    return stats
 
 
# ─── Step 4: Save graph ───────────────────────────────────────────────────────
 
def save_graph(G: nx.DiGraph):
    GRAPH_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
 
    # JSON — node + edge lists (used by RAG pipeline later)
    graph_data = {
        "nodes": [
            {"id": n, **G.nodes[n]}
            for n in G.nodes()
        ],
        "edges": [
            {"source": u, "target": v, "weight": d.get("weight", 1)}
            for u, v, d in G.edges(data=True)
        ],
    }
    with open(GRAPH_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print(f"  Graph JSON saved to {GRAPH_JSON_PATH}")
 
    # GEXF — for Gephi visualisation
    nx.write_gexf(G, str(GRAPH_GEXF_PATH))
    print(f"  Graph GEXF saved to {GRAPH_GEXF_PATH}  (open in Gephi)")
 
    # pyvis — interactive HTML
    build_pyvis(G)
 
 
def build_pyvis(G: nx.DiGraph):
    """Build an interactive HTML graph with pyvis. Color by community."""
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#333333",
    )
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)
 
    # Community → color mapping
    communities = nx.get_node_attributes(G, "community")
    palette = [
        "#4A90D9", "#7B68EE", "#50C878", "#FF6B6B", "#FFD700",
        "#FF8C00", "#00CED1", "#DA70D6", "#98FB98", "#F08080",
    ]
 
    for node in G.nodes():
        meta  = G.nodes[node]
        comm  = communities.get(node, 0)
        color = palette[comm % len(palette)]
        freq  = meta.get("frequency", 1)
        size  = max(10, min(40, freq * 3))  # scale node size by frequency
        title = (
            f"<b>{node}</b><br>"
            f"Chapter: {meta.get('chapter', 'unknown')}<br>"
            f"Frequency: {freq}<br>"
            f"Definition: {meta.get('definition', '')[:120]}"
        )
        net.add_node(
            node,
            label=node,
            title=title,
            color=color,
            size=size,
        )
 
    for u, v, d in G.edges(data=True):
        weight = d.get("weight", 1)
        net.add_edge(u, v, width=max(0.5, weight * 0.5), arrows="to")
 
    GRAPH_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(GRAPH_HTML_PATH))
    print(f"  Interactive graph saved to {GRAPH_HTML_PATH}  (open in browser)")
 
 
# ─── Subgraph query (used by GraphRAG pipeline later) ────────────────────────
 
def get_subgraph_context(G: nx.DiGraph, concept_names: list[str], hops: int = 1) -> str:
    """
    Given a list of concept names, return a text description of their
    neighbourhood in the graph. Used in Stage 3 (GraphRAG).
    """
    lines = []
    for name in concept_names:
        if name not in G:
            continue
        prereqs  = list(G.predecessors(name))
        enables  = list(G.successors(name))
        if prereqs or enables:
            line = f"'{name}'"
            if prereqs:
                line += f" requires: {', '.join(prereqs[:5])}"
            if enables:
                line += f"; enables: {', '.join(enables[:5])}"
            lines.append(line)
        if hops == 2:
            for prereq in prereqs[:3]:
                second = list(G.predecessors(prereq))
                if second:
                    lines.append(f"  '{prereq}' also requires: {', '.join(second[:3])}")
    return "\n".join(lines) if lines else "No graph context available."
 
 
# ─── Main ────────────────────────────────────────────────────────────────────
 
def main(eda_only: bool = False):
    print("=" * 55)
    print("Knowledge Graph Construction — CLRS")
    print("=" * 55)
 
    # Load graph from cache for eda-only mode
    if eda_only:
        if not GRAPH_JSON_PATH.exists():
            print("ERROR: No graph found. Run without --eda-only first.")
            return
        print("\nLoading saved graph for EDA...")
        with open(GRAPH_JSON_PATH, encoding="utf-8") as f:
            gd = json.load(f)
        G = nx.DiGraph()
        for n in gd["nodes"]:
            nid = n.pop("id")
            G.add_node(nid, **n)
        for e in gd["edges"]:
            G.add_edge(e["source"], e["target"], weight=e["weight"])
        run_eda(G)
        return
 
    # Load chunks
    if not CHUNKS_PATH.exists():
        print(f"ERROR: {CHUNKS_PATH} not found. Run clrs_pipeline.py first.")
        return
 
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        all_chunks = json.load(f)
 
    chunks = all_chunks[:MAX_CHUNKS] if MAX_CHUNKS else all_chunks
    print(f"\nLoaded {len(chunks)} chunks")
 
    # Step 1: Extract concepts
    print("\n[1/4] Extracting concepts with Llama 3.1 8B...")
    extractions = extract_concepts(chunks)
 
    # Step 2: Build graph
    print("\n[2/4] Building prerequisite graph...")
    G = build_graph(extractions)
 
    # Step 3: EDA
    print("\n[3/4] Running EDA...")
    run_eda(G)
 
    # Step 4: Save
    print("\n[4/4] Saving graph files...")
    save_graph(G)
 
    print("\n" + "=" * 55)
    print("Done. Output files:")
    print(f"  {CONCEPTS_PATH}   — raw extractions (LLM cache)")
    print(f"  {GRAPH_JSON_PATH} — graph for RAG pipeline")
    print(f"  {GRAPH_GEXF_PATH} — open in Gephi for exploration")
    print(f"  {GRAPH_HTML_PATH} — open in browser for interactive view")
    print(f"  {EDA_PATH}        — EDA stats")
    print("\nNext step: build the RAG pipeline using clrs_graph.json")
    print("=" * 55)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eda-only", action="store_true",
                        help="Skip extraction, just re-run EDA on saved graph")
    args = parser.parse_args()
    main(eda_only=args.eda_only)
 