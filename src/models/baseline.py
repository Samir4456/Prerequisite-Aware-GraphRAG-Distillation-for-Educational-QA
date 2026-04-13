"""
src/models/baseline.py

DistilBERT extractive QA baseline on MetaQA.
- 50k examples (16,667 per hop)
- 6 epochs
- fp16 mixed precision
- Dataset caching to disk (first run builds, subsequent runs load in ~3s)
- tqdm progress bars

Usage:
    python src/models/baseline.py \
        --data_dir data/raw \
        --kb_path  data/raw/kb.txt \
        --output_dir checkpoints/distilbert-baseline \
        --samples_per_hop 16667 \
        --epochs 6

    # Force rebuild cache (e.g. after changing load_kb.py):
    python src/models/baseline.py ... --no_cache
"""

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from transformers import (
    DistilBertForQuestionAnswering,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from load_kb import load_kb
from load_metaqa import load_all_splits


# ─────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────

def build_context(topic_entity: str, adjacency: dict, hops: int = 2) -> str:
    """Serialise N-hop subgraph into a short text passage."""
    lines = []
    frontier = {topic_entity}
    visited = {topic_entity}

    for _ in range(hops):
        next_frontier = set()
        for entity in frontier:
            for rel, obj in adjacency.get(entity, []):
                lines.append(f"{entity} {rel} {obj}.")
                if obj not in visited:
                    next_frontier.add(obj)
                    visited.add(obj)
        frontier = next_frontier

    return " ".join(lines) if lines else topic_entity


def find_answer_span(context: str, answers: list) -> tuple:
    """Return (start_char, end_char) for the first answer found in context."""
    context_lower = context.lower()
    for ans in answers:
        idx = context_lower.find(ans.lower())
        if idx != -1:
            return idx, idx + len(ans)
    return None


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class MetaQASpanDataset(Dataset):
    def __init__(self, qa_pairs, adjacency, tokenizer, max_length=384, hops=2, desc=""):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        skipped = 0
        for item in tqdm(qa_pairs, desc=f"  Tokenizing {desc}", leave=False):
            question = item['question']
            answers = item['answers']
            entity = item.get('topic_entity') or ''

            context = build_context(entity, adjacency, hops=hops)
            span = find_answer_span(context, answers)

            if span is None:
                skipped += 1
                continue

            start_char, end_char = span

            encoding = tokenizer(
                question,
                context,
                max_length=max_length,
                truncation="only_second",
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            offsets = encoding['offset_mapping'][0].tolist()
            sequence_ids = encoding.sequence_ids(0)

            start_token, end_token = 0, 0
            for i, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
                if seq_id != 1:
                    continue
                if offset[0] <= start_char < offset[1]:
                    start_token = i
                if offset[0] < end_char <= offset[1]:
                    end_token = i

            self.examples.append({
                'input_ids': encoding['input_ids'][0],
                'attention_mask': encoding['attention_mask'][0],
                'start_positions': torch.tensor(start_token),
                'end_positions': torch.tensor(end_token),
                'question': question,
                'context': context,
                'gold_answers': answers,
                'hop': hops,
            })

        print(f"    loaded {len(self.examples):,}  |  skipped {skipped:,}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'input_ids': ex['input_ids'],
            'attention_mask': ex['attention_mask'],
            'start_positions': ex['start_positions'],
            'end_positions': ex['end_positions'],
        }


# ─────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────

def cache_key(data_dir, samples_per_hop, split, max_length):
    """Generate a unique cache filename based on config."""
    key = f"{data_dir}_{samples_per_hop}_{split}_{max_length}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def save_cache(dataset, cache_path):
    """Save dataset examples to disk."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(dataset.examples, f)
    print(f"  Cache saved → {cache_path}")


def load_cache(cache_path):
    """Load dataset examples from disk. Returns None if not found."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    print(f"  Loading from cache: {cache_path}")
    with open(cache_path, 'rb') as f:
        examples = pickle.load(f)
    print(f"  Loaded {len(examples):,} examples in ~3s")
    return examples


# ─────────────────────────────────────────────
# Build combined dataset
# ─────────────────────────────────────────────

def build_combined_dataset(
    data_dir, adjacency, tokenizer, samples_per_hop, split,
    cache_dir="data/processed/cache", use_cache=True, max_length=384
):
    """Load and combine all 3 hops, with disk caching."""
    all_datasets = []
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)

    for hop_num in [1, 2, 3]:
        hop_dir = data_dir / f"{hop_num}hop"
        if not hop_dir.exists():
            print(f"  [WARN] {hop_dir} not found, skipping.")
            continue

        context_hops = max(hop_num, 2)
        ckey = cache_key(str(hop_dir), samples_per_hop, split, max_length)
        cpath = cache_dir / f"{hop_num}hop_{split}_{ckey}.pkl"

        # Try loading from cache
        if use_cache:
            cached = load_cache(cpath)
            if cached is not None:
                ds = MetaQASpanDataset.__new__(MetaQASpanDataset)
                ds.examples = cached
                all_datasets.append(ds)
                print(f"  {hop_num}-hop {split}: {len(ds.examples):,} examples (from cache)")
                continue

        # Build from scratch
        print(f"  {hop_num}-hop {split}: loading {samples_per_hop:,} pairs → context_hops={context_hops}")
        splits = load_all_splits(str(hop_dir), max_samples=samples_per_hop)
        pairs = splits[split]

        t0 = time.time()
        ds = MetaQASpanDataset(
            pairs, adjacency, tokenizer,
            max_length=max_length, hops=context_hops,
            desc=f"{hop_num}hop/{split}"
        )
        elapsed = time.time() - t0
        print(f"    built in {elapsed:.1f}s")

        if use_cache:
            save_cache(ds, cpath)

        all_datasets.append(ds)

    combined = ConcatDataset(all_datasets)
    combined.examples = []
    for ds in all_datasets:
        combined.examples.extend(ds.examples)

    return combined


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def exact_match(pred, gold_answers):
    pred = pred.strip().lower()
    return any(pred == g.strip().lower() for g in gold_answers)


def f1_score(pred, gold_answers):
    pred_tokens = pred.lower().split()
    best = 0.0
    for gold in gold_answers:
        gold_tokens = gold.lower().split()
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        p = len(common) / len(pred_tokens)
        r = len(common) / len(gold_tokens)
        best = max(best, 2 * p * r / (p + r))
    return best


# ─────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataset, tokenizer, device, tag="dev"):
    model.eval()
    em_scores, f1_scores, latencies = [], [], []

    for ex in tqdm(dataset.examples, desc=f"  Evaluating {tag}", leave=False):
        input_ids = ex['input_ids'].unsqueeze(0).to(device)
        attention_mask = ex['attention_mask'].unsqueeze(0).to(device)

        t0 = time.time()
        with autocast("cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        latencies.append(time.time() - t0)

        start_idx = outputs.start_logits.argmax().item()
        end_idx = outputs.end_logits.argmax().item()
        if end_idx < start_idx:
            end_idx = start_idx

        tokens = input_ids[0][start_idx: end_idx + 1]
        pred = tokenizer.decode(tokens, skip_special_tokens=True)

        em_scores.append(float(exact_match(pred, ex['gold_answers'])))
        f1_scores.append(f1_score(pred, ex['gold_answers']))

    return {
        f'{tag}/EM': round(sum(em_scores) / len(em_scores), 4) if em_scores else 0,
        f'{tag}/F1': round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0,
        f'{tag}/latency_ms': round(1000 * sum(latencies) / len(latencies), 2) if latencies else 0,
    }


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────

def train(args):
    wandb.init(
        project="pocket-graphrag",
        name=f"distilbert-baseline-{args.samples_per_hop}x3-{args.epochs}ep",
        config=vars(args),
        tags=["stage1", "baseline", "all-hops"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nLoading knowledge base...")
    triples, adjacency = load_kb(args.kb_path)
    print(f"  {len(triples):,} triples  |  {len(adjacency):,} nodes (bidirectional)")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.to(device)

    use_cache = not args.no_cache

    print("\nBuilding train dataset...")
    t0 = time.time()
    train_dataset = build_combined_dataset(
        args.data_dir, adjacency, tokenizer,
        args.samples_per_hop, "train",
        cache_dir=args.cache_dir, use_cache=use_cache,
    )
    print(f"  Total train: {len(train_dataset):,} examples  ({(time.time()-t0):.1f}s)")

    print("\nBuilding dev dataset...")
    t0 = time.time()
    dev_dataset = build_combined_dataset(
        args.data_dir, adjacency, tokenizer,
        1000, "dev",
        cache_dir=args.cache_dir, use_cache=use_cache,
    )
    print(f"  Total dev: {len(dev_dataset):,} examples  ({(time.time()-t0):.1f}s)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )

    scaler = GradScaler("cuda")
    accum_steps = args.grad_accum

    print(f"\nTraining: {args.epochs} epochs | "
          f"batch={args.batch_size} | accum={accum_steps} | "
          f"effective_batch={args.batch_size * accum_steps} | "
          f"total_steps={total_steps:,} | fp16=True\n")

    global_step = 0
    best_dev_em = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_epoch = time.time()
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_pos      = batch['start_positions'].to(device)
            end_pos        = batch['end_positions'].to(device)

            with autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_pos,
                    end_positions=end_pos,
                )
                loss = outputs.loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    wandb.log({'train/loss': loss.item() * accum_steps, 'step': global_step})

            epoch_loss += loss.item() * accum_steps
            pbar.set_postfix({'loss': f"{loss.item() * accum_steps:.4f}"})

        elapsed = time.time() - t_epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch}/{args.epochs}  avg_loss={avg_loss:.4f}  time={elapsed/60:.1f}min")
        wandb.log({'train/epoch_loss': avg_loss, 'epoch': epoch})

        # Dev evaluation
        metrics = evaluate(model, dev_dataset, tokenizer, device, tag="dev")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        wandb.log({**metrics, 'epoch': epoch})

        # Save best model
        if metrics['dev/EM'] > best_dev_em:
            best_dev_em = metrics['dev/EM']
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            print(f"  ✓ New best dev EM={best_dev_em:.4f} — model saved")

    print(f"\nBest dev EM: {best_dev_em:.4f}")

    # Final per-hop test evaluation
    print("\nFinal test evaluation (per hop, 2000 samples each)...")
    all_test_metrics = {}
    for hop_num in [1, 2, 3]:
        hop_dir = Path(args.data_dir) / f"{hop_num}hop"
        if not hop_dir.exists():
            continue
        context_hops = max(hop_num, 2)
        splits_data = load_all_splits(str(hop_dir), max_samples=2000)
        print(f"\n  {hop_num}-hop test → context_hops={context_hops}")
        test_ds = MetaQASpanDataset(
            splits_data['test'], adjacency, tokenizer,
            hops=context_hops, desc=f"{hop_num}hop/test"
        )
        m = evaluate(model, test_ds, tokenizer, device, tag=f"test_{hop_num}hop")
        for k, v in m.items():
            print(f"    {k}: {v}")
        all_test_metrics.update(m)
        wandb.log({**m, 'stage': 1})

    # Save results
    out_dir = Path(args.output_dir)
    summary = {
        'stage': 1,
        'model': 'distilbert-base-uncased',
        'samples_per_hop': args.samples_per_hop,
        'total_train': len(train_dataset),
        'epochs': args.epochs,
        'best_dev_EM': best_dev_em,
        **all_test_metrics,
    }
    with open(out_dir / "results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== BASELINE LOCKED — everything else must beat this ===")
    for k, v in all_test_metrics.items():
        print(f"  {k}: {v}")

    wandb.finish()
    return summary


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilBERT baseline — all hops, cached")
    parser.add_argument("--data_dir",        default="data/raw")
    parser.add_argument("--kb_path",         default="data/raw/kb.txt")
    parser.add_argument("--output_dir",      default="checkpoints/distilbert-baseline")
    parser.add_argument("--cache_dir",       default="data/processed/cache")
    parser.add_argument("--samples_per_hop", type=int, default=16667,
                        help="Train examples per hop (16667 x 3 ≈ 50k total)")
    parser.add_argument("--epochs",          type=int, default=6)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--grad_accum",      type=int, default=2)
    parser.add_argument("--lr",              type=float, default=3e-5)
    parser.add_argument("--no_cache",        action="store_true",
                        help="Disable caching — force rebuild dataset from scratch")
    args = parser.parse_args()

    train(args)