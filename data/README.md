# Evolutionary run data

Durable, analysis-friendly archives of what the evolutionary search actually
produced. The live database (`program_database.json` in the repo root) is a
single mutable file that every run overwrites in place, and it lives inside an
ephemeral container — so it is a poor place to keep results you want to learn
from. This directory is where results are preserved (committed and pushed) so
they survive the container and can be used to improve the system.

Populate it with `scripts/archive_run.py` after a run:

```bash
python scripts/archive_run.py --label pop12_gen8 --config "POPULATION_SIZE=12 GENERATIONS=8"
```

## Layout

```
data/
  runs/<UTC-timestamp>[__<label>]/
      programs.jsonl   One JSON object per program: id, task_id, generation,
                       parent_id, island_id, status, fitness_scores, code.
                       Load line-by-line to analyse a run — e.g. follow
                       parent_id chains to see which mutations improved fitness.
      summary.json     Run metadata (timestamp, label, config, model roles) plus
                       per-task counts, the best program per task, and the
                       lineage of each task's winner.
  best_corpus/<task_id>.jsonl
      The accumulating set of best programs per task (highest correctness, then
      fastest), de-duplicated by code and sorted best-first. Grows across runs.
```

## Using it to improve the system

- **Seed future searches (wired in).** The initial population is automatically
  warm-started from `best_corpus/<task_id>.jsonl`: the top entries are injected
  as gen-0 seed programs (ids `..._gen0_seed<i>`) and re-evaluated, while the
  rest of the population is still generated fresh so the search keeps exploring.
  Controlled by `SEED_FROM_CORPUS` (default on), `SEED_CORPUS_COUNT` (default 2,
  capped at `POPULATION_SIZE - 1`), and `BEST_CORPUS_DIR`. Tasks with no corpus
  file behave exactly as before (fully fresh generation).
- **Few-shot exemplars.** Use the top corpus entries as worked examples in the
  initial/mutation prompts for the same task.
- **Study what works.** `programs.jsonl` + `winner_lineage` in `summary.json`
  show which parent→child mutations moved fitness, and how fast winners emerge
  across generations — signal for tuning population size, generations, and the
  primary/secondary model split.
- **Regression tracking.** Compare `summary.json` across runs to see whether a
  change to the framework helps or hurts best-achieved fitness/runtime per task.

## Note on durability

This repo is the only durable store configured in this environment, so archives
are committed here. If you have external storage (an object store, a database,
an experiment tracker), point `archive_run.py --data-dir` at a synced location
or extend it to upload there — the JSONL/JSON formats are chosen to be portable.
