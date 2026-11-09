"""يشغّل اختبارات YAML على الحلول المرجعية والخاطئة. الاستخدام: python run_spec_tests.py"""
import os, sys, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_solutions import decide_next, should_skip_tests
from faulty_solutions import FAULTY

HERE = os.path.dirname(os.path.abspath(__file__))
SPECS = {"decide_next": (os.path.join(HERE, "decide_next.yaml"), decide_next),
         "should_skip_tests": (os.path.join(HERE, "should_skip_tests.yaml"), should_skip_tests)}

def run(fn, cfg):
    fails = []
    total = 0
    for group in cfg["tests"]:
        for i, tc in enumerate(group["test_cases"]):
            total += 1
            try:
                got = fn(*tc["input"])
            except Exception as e:
                got = f"EXC:{type(e).__name__}"
            if got != tc["output"] or type(got) != type(tc["output"]):
                fails.append((group["name"], i, tc["input"], tc["output"], got))
    return total, fails

ok = True
for fname, (path, ref_fn) in SPECS.items():
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    assert cfg["function_name"] == fname, "function_name mismatch"
    total, fails = run(ref_fn, cfg)
    print(f"[REF] {fname}: {total - len(fails)}/{total} passed")
    for f in fails:
        ok = False
        print("   FAIL", f)
    for name, impl in FAULTY[fname].items():
        t, f = run(impl, cfg)
        status = "CAUGHT" if f else "MISSED (weak tests!)"
        if not f:
            ok = False
        print(f"[BAD] {fname}/{name}: {len(f)}/{t} failing -> {status}")
print("ALL CHECKS OK" if ok else "SOME CHECKS FAILED")
