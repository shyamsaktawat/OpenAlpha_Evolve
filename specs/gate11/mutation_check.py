"""طفرات لكل قاعدة: كل طفرة تخرق قاعدة واحدة. يميّز الكشف السلوكي (مخرج خاطئ) عن الكشف بالاستثناء فقط."""
import os, sys, yaml
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reference_solutions import decide_next, should_skip_tests
from faulty_solutions import FAULTY

def mk_decide(MIN_REPS=5, MAX_REPS=15, lo_strict=False, hi_le=False, warm_le=False, empty_before_hung=False, warm_in_measured=False):
    def f(ratios, hung):
        BUDGET, FLOOR = 0.96, 0.90
        if empty_before_hung and not ratios:
            return "CONTINUE"
        if hung:
            return "HANG"
        if not ratios:
            return "CONTINUE"
        if (ratios[0] <= FLOOR) if warm_le else (ratios[0] < FLOOR):
            return "STOP_FAIL_EARLY"
        measured = ratios if warm_in_measured else ratios[1:]
        n = len(measured)
        if n < MIN_REPS:
            return "CONTINUE"
        lo, hi = min(measured), max(measured)
        if (lo > BUDGET) if lo_strict else (lo >= BUDGET):
            return "STOP_PASS"
        if (hi <= BUDGET) if hi_le else (hi < BUDGET):
            return "STOP_FAIL"
        return "INCONCLUSIVE" if n >= MAX_REPS else "CONTINUE"
    return f

def mk_skip(no_github=False, no_dotdot=False, docs_noslash=False, accept_RT=False, empty_true=False,
            docs_bypass_status=False, md_ci=False, skip_malformed=False):
    def f(changes):
        if not changes:
            return empty_true
        for item in changes:
            if len(item) != 2:
                if skip_malformed:
                    continue
                return False
            status, path = item
            bypass = docs_bypass_status and isinstance(path, str) and (path.startswith("docs/") or path.startswith("hdae/"))
            if not bypass:
                allowed = ("A", "M", "D", "R", "T") if accept_RT else ("A", "M", "D")
                if status not in allowed:
                    return False
            if not isinstance(path, str) or not path:
                return False
            if not no_dotdot and ".." in path.split("/"):
                return False
            if not no_github and path.startswith(".github/"):
                return False
            prefix_ok = path.startswith("docs" if docs_noslash else "docs/") or path.startswith("hdae/")
            ext_ok = path.lower().endswith(".md") if md_ci else path.endswith(".md")
            if not (prefix_ok or ext_ok):
                return False
        return True
    return f

MUTANTS = {
    "decide_next": {
        "MIN_REPS=4": mk_decide(MIN_REPS=4),
        "MAX_REPS=14": mk_decide(MAX_REPS=14),
        "lo > BUDGET بدل >=": mk_decide(lo_strict=True),
        "hi <= BUDGET بدل <": mk_decide(hi_le=True),
        "ratios[0] <= FLOOR بدل <": mk_decide(warm_le=True),
        "فحص not ratios قبل hung": mk_decide(empty_before_hung=True),
        "الإحماء ضمن measured": mk_decide(warm_in_measured=True),
    },
    "should_skip_tests": {
        "حذف فحص .github/": mk_skip(no_github=True),
        "حذف فحص ..": mk_skip(no_dotdot=True),
        'startswith("docs") بدل "docs/"': mk_skip(docs_noslash=True),
        "قبول R و T": mk_skip(accept_RT=True),
        "القائمة الفارغة -> True": mk_skip(empty_true=True),
        "docs/ و hdae/ بلا فحص status": mk_skip(docs_bypass_status=True),
        ".md بلا حساسية للأحرف (lower)": mk_skip(md_ci=True),
        "عنصر بشكل خاطئ يُتجاوز بدل False": mk_skip(skip_malformed=True),
    },
}

def evaluate(fn, cfg):
    total, wrong, exc = 0, [], []
    for group in cfg["tests"]:
        for i, tc in enumerate(group["test_cases"]):
            total += 1
            try:
                got = fn(*tc["input"])
            except Exception as e:
                exc.append((group["name"], i, tc["input"], tc["output"], type(e).__name__))
                continue
            if got != tc["output"] or type(got) != type(tc["output"]):
                wrong.append((group["name"], i, tc["input"], tc["output"], got))
    return total, wrong, exc

def load(name):
    return yaml.safe_load(open(os.path.join(HERE, name + ".yaml"), encoding="utf-8"))

REF = {"decide_next": decide_next, "should_skip_tests": should_skip_tests}
cfgs = {n: load(n) for n in REF}

print("== الحلول المرجعية ==")
for n in REF:
    total, wrong, exc = evaluate(REF[n], cfgs[n])
    print(f"REF {n}: passed {total - len(wrong) - len(exc)}/{total}")

print("== الحلول الخاطئة (FAULTY) ==")
fc = fb = 0
for n in FAULTY:
    for name, fn in FAULTY[n].items():
        fc += 1
        total, wrong, exc = evaluate(fn, cfgs[n])
        kind = "BEHAVIORAL" if wrong else ("EXCEPTION_ONLY" if exc else "MISSED")
        fb += 1 if wrong else 0
        first = (wrong or exc)[0] if (wrong or exc) else None
        print(f"FAULTY {n}/{name}: wrong={len(wrong)} exc={len(exc)} -> {kind} first={first}")
print(f"FAULTY total={fc} behavioral_kills={fb}")

print("== الطفرات (MUTANTS) ==")
mc = mb = mk = 0
for n in MUTANTS:
    for name, fn in MUTANTS[n].items():
        mc += 1
        total, wrong, exc = evaluate(fn, cfgs[n])
        kind = "BEHAVIORAL" if wrong else ("EXCEPTION_ONLY" if exc else "SURVIVED")
        mk += 1 if (wrong or exc) else 0
        mb += 1 if wrong else 0
        first = (wrong or exc)[0] if (wrong or exc) else None
        print(f"MUTANT {n}/{name}: wrong={len(wrong)} exc={len(exc)} -> {kind} first={first}")
print(f"MUTANTS total={mc} killed={mk} behavioral_kills={mb}")
