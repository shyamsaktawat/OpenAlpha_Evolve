#!/usr/bin/env python3
"""Archive an evolutionary run's program database into a durable, analysis-friendly form.

The live database (``program_database.json``) is a single mutable file that every
run overwrites in place, and it lives inside an ephemeral container -- so it is a
poor place to keep results you want to learn from. This script snapshots it into
``data/`` in two shapes:

  data/runs/<UTC-timestamp>[__<label>]/
      programs.jsonl   one JSON object per program (id, task_id, generation,
                       parent_id, status, fitness_scores, code) -- easy to load
                       for analysis of what the search produced and which
                       mutations improved fitness (follow parent_id chains).
      summary.json     run metadata (timestamp, label, config, model roles) plus
                       per-task counts, the best program per task, and the
                       lineage of each task's winner.

  data/best_corpus/<task_id>.jsonl
      The accumulating set of best (highest-correctness, then fastest) programs
      per task, de-duplicated by code. This is the reusable artifact for
      improving the system: seed future searches from it, or use the entries as
      few-shot exemplars in prompts.

Everything written here is meant to be committed and pushed -- that is what makes
it durable beyond the ephemeral container.

Usage:
    python scripts/archive_run.py [--db program_database.json] [--label pop12_gen8]
                                  [--config "POPULATION_SIZE=12 GENERATIONS=8"]
                                  [--data-dir data]
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

_PROGRAM_FIELDS = (
    "id", "task_id", "generation", "parent_id", "island_id", "status",
    "fitness_scores", "code",
)


def _load_db(path):
    if not os.path.exists(path):
        sys.exit(f"database not found: {path}")
    with open(path) as fh:
        return json.load(fh)


def _fitness(program):
    return program.get("fitness_scores") or {}


def _sort_key_best(program):
    """Best = highest correctness, then most tests passed, then fastest runtime."""
    f = _fitness(program)
    return (
        f.get("correctness", 0.0),
        f.get("passed_tests", 0.0),
        -f.get("runtime_ms", float("inf")),
    )


def _slim(program):
    return {k: program.get(k) for k in _PROGRAM_FIELDS}


def _lineage(program, by_id):
    """Walk parent_id back to the root, returning ids from ancestor -> program."""
    chain, seen, cur = [], set(), program
    while cur is not None and cur.get("id") not in seen:
        chain.append(cur["id"])
        seen.add(cur["id"])
        parent_id = cur.get("parent_id")
        cur = by_id.get(parent_id) if parent_id else None
    return list(reversed(chain))


def _code_hash(code):
    return hashlib.sha256((code or "").encode("utf-8")).hexdigest()[:16]


def _update_best_corpus(corpus_path, programs):
    """Merge the given programs into a per-task best corpus, de-duplicated by code."""
    existing = {}
    if os.path.exists(corpus_path):
        with open(corpus_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    existing[entry["code_hash"]] = entry
    for program in programs:
        h = _code_hash(program.get("code"))
        entry = {
            "code_hash": h,
            "source_id": program.get("id"),
            "fitness_scores": _fitness(program),
            "code": program.get("code"),
        }
        prev = existing.get(h)
        # keep the record with the better fitness for identical code
        if prev is None or _sort_key_best(program) > _sort_key_best({"fitness_scores": prev["fitness_scores"]}):
            existing[h] = entry
    ordered = sorted(existing.values(),
                     key=lambda e: _sort_key_best({"fitness_scores": e["fitness_scores"]}),
                     reverse=True)
    with open(corpus_path, "w") as fh:
        for entry in ordered:
            fh.write(json.dumps(entry) + "\n")
    return len(ordered)


def archive(db_path, data_dir, label, config, top_n_corpus):
    db = _load_db(db_path)
    by_id = {pid: {"id": pid, **prog} if "id" not in prog else prog
             for pid, prog in db.items()}
    # ensure every program carries its own id (older records key by id only)
    for pid, prog in db.items():
        prog.setdefault("id", pid)

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dirname = f"{timestamp}__{label}" if label else timestamp
    run_dir = os.path.join(data_dir, "runs", run_dirname)
    corpus_dir = os.path.join(data_dir, "best_corpus")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(corpus_dir, exist_ok=True)

    programs = list(db.values())

    # programs.jsonl -- full snapshot, one program per line
    with open(os.path.join(run_dir, "programs.jsonl"), "w") as fh:
        for prog in sorted(programs, key=lambda p: (p.get("task_id") or "", p.get("id") or "")):
            fh.write(json.dumps(_slim(prog)) + "\n")

    # per-task stats + winners
    tasks = {}
    for prog in programs:
        tasks.setdefault(prog.get("task_id"), []).append(prog)

    per_task = {}
    corpus_sizes = {}
    for task_id, progs in tasks.items():
        if task_id is None:
            continue
        best = max(progs, key=_sort_key_best)
        perfect = [p for p in progs
                   if _fitness(p).get("correctness") == 1.0]
        per_task[task_id] = {
            "program_count": len(progs),
            "perfect_count": len(perfect),
            "best_id": best.get("id"),
            "best_fitness": _fitness(best),
            "best_generation": best.get("generation"),
            "winner_lineage": _lineage(best, by_id),
        }
        # feed all perfect programs (or the best if none perfect) into the corpus
        corpus_source = perfect if perfect else [best]
        corpus_path = os.path.join(corpus_dir, f"{task_id}.jsonl")
        corpus_sizes[task_id] = _update_best_corpus(corpus_path, corpus_source)

    summary = {
        "timestamp_utc": timestamp,
        "label": label,
        "config": config,
        "models": {
            "secondary": os.getenv("LLM_SECONDARY_MODEL"),
            "primary": os.getenv("LLM_PRIMARY_MODEL"),
            "high_fitness_threshold": os.getenv("HIGH_FITNESS_THRESHOLD_FOR_PRIMARY_LLM"),
        },
        "total_programs": len(programs),
        "per_task": per_task,
        "best_corpus_sizes": corpus_sizes,
        "source_db": os.path.basename(db_path),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return run_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="program_database.json",
                        help="path to the live program database (default: program_database.json)")
    parser.add_argument("--data-dir", default="data",
                        help="root directory for archives (default: data)")
    parser.add_argument("--label", default="",
                        help="short label for this run, e.g. pop12_gen8")
    parser.add_argument("--config", default="",
                        help="free-text config note, e.g. 'POPULATION_SIZE=12 GENERATIONS=8'")
    parser.add_argument("--top-n-corpus", type=int, default=0,
                        help="reserved; corpus keeps all de-duplicated best programs")
    args = parser.parse_args(argv)

    run_dir, summary = archive(args.db, args.data_dir, args.label, args.config, args.top_n_corpus)
    print(f"Archived {summary['total_programs']} programs -> {run_dir}")
    for task_id, stats in summary["per_task"].items():
        bf = stats["best_fitness"]
        print(f"  [{task_id}] best={stats['best_id']} "
              f"correctness={bf.get('correctness')} "
              f"runtime_ms={bf.get('runtime_ms')} "
              f"(corpus now {summary['best_corpus_sizes'].get(task_id)} programs)")


if __name__ == "__main__":
    main()
