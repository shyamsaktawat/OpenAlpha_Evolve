"""Tests for warm-starting the initial population from the best-program corpus.

Verifies that TaskManagerAgent.initialize_population injects corpus seeds when a
corpus exists, always leaves room for fresh generation, and degrades gracefully
to fully-fresh generation when no corpus is present.

Uses unittest.IsolatedAsyncioTestCase to match the async-test style already used
in this repo (see tests/test_evaluator_agent.py); pytest-asyncio is not a dep.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from core.interfaces import TaskDefinition
from config import settings
from task_manager.agent import TaskManagerAgent


def _make_task(task_id):
    return TaskDefinition(
        id=task_id,
        description="dummy",
        function_name_to_evolve="f",
        allowed_imports=[],
    )


def _make_manager(task, population_size):
    tm = TaskManagerAgent(task_definition=task)
    tm.population_size = population_size
    # Stub out anything that would hit the network, Docker, or disk persistence.
    tm.code_generator.generate_code = AsyncMock(return_value="def f():\n    return 1")
    tm.database.save_program = AsyncMock()
    tm.selection_controller.initialize_islands = MagicMock()
    return tm


def _write_corpus(directory, task_id, codes):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{task_id}.jsonl")
    with open(path, "w") as fh:
        for i, code in enumerate(codes):
            fh.write(json.dumps({
                "code_hash": f"h{i}",
                "source_id": f"src{i}",
                "fitness_scores": {"correctness": 1.0, "runtime_ms": float(i)},
                "code": code,
            }) + "\n")
    return path


class CorpusSeedingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._corpus_dir = os.path.join(self._tmp.name, "best_corpus")
        # Save and override the relevant settings; restored in tearDown.
        self._saved = {
            "BEST_CORPUS_DIR": settings.BEST_CORPUS_DIR,
            "SEED_FROM_CORPUS": settings.SEED_FROM_CORPUS,
            "SEED_CORPUS_COUNT": settings.SEED_CORPUS_COUNT,
        }
        settings.BEST_CORPUS_DIR = self._corpus_dir
        settings.SEED_FROM_CORPUS = True
        settings.SEED_CORPUS_COUNT = 2

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(settings, k, v)
        self._tmp.cleanup()

    async def test_population_is_seeded_from_corpus(self):
        task = _make_task("task_with_corpus")
        _write_corpus(self._corpus_dir, task.id,
                      ["def f():\n    return 'seed_a'", "def f():\n    return 'seed_b'"])
        tm = _make_manager(task, population_size=5)

        pop = await tm.initialize_population()

        seeds = [p for p in pop if "seed" in p.id]
        fresh = [p for p in pop if "prog" in p.id]
        self.assertEqual(len(pop), 5)
        self.assertEqual(len(seeds), 2)
        self.assertEqual(len(fresh), 3)
        # seed code comes from the corpus, not the fresh-generation stub
        self.assertEqual(seeds[0].code, "def f():\n    return 'seed_a'")
        self.assertTrue(all(p.task_id == task.id for p in pop))

    async def test_no_corpus_falls_back_to_all_fresh(self):
        task = _make_task("task_without_corpus")  # no corpus file written
        tm = _make_manager(task, population_size=4)

        pop = await tm.initialize_population()

        self.assertEqual(len(pop), 4)
        self.assertEqual(sum("seed" in p.id for p in pop), 0)
        self.assertEqual(sum("prog" in p.id for p in pop), 4)

    async def test_seeds_never_fill_the_whole_population(self):
        """Even with a large corpus and high SEED_CORPUS_COUNT, at least one fresh
        program must be generated so the search keeps exploring."""
        settings.SEED_CORPUS_COUNT = 100
        task = _make_task("task_big_corpus")
        _write_corpus(self._corpus_dir, task.id, [f"def f():\n    return {i}" for i in range(20)])
        tm = _make_manager(task, population_size=3)

        pop = await tm.initialize_population()

        self.assertEqual(len(pop), 3)
        self.assertEqual(sum("seed" in p.id for p in pop), 2)  # capped at population_size - 1
        self.assertEqual(sum("prog" in p.id for p in pop), 1)

    async def test_seeding_disabled_generates_all_fresh(self):
        settings.SEED_FROM_CORPUS = False
        task = _make_task("task_disabled")
        _write_corpus(self._corpus_dir, task.id, ["def f():\n    return 'x'"])
        tm = _make_manager(task, population_size=4)

        pop = await tm.initialize_population()

        self.assertEqual(sum("seed" in p.id for p in pop), 0)
        self.assertEqual(sum("prog" in p.id for p in pop), 4)


if __name__ == "__main__":
    unittest.main()
