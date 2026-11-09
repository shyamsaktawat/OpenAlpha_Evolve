"""حلول مرجعية لاختبار المواصفة (القسم 4). إذا فشلت، الخطأ في الاختبار أو في المواصفة."""

def decide_next(ratios, hung):
    BUDGET, WARMUP_FLOOR, MIN_REPS, MAX_REPS = 0.96, 0.90, 5, 15
    if hung:
        return "HANG"
    if not ratios:
        return "CONTINUE"
    if ratios[0] < WARMUP_FLOOR:
        return "STOP_FAIL_EARLY"
    measured = ratios[1:]
    n = len(measured)
    if n < MIN_REPS:
        return "CONTINUE"
    lo, hi = min(measured), max(measured)
    if lo >= BUDGET:
        return "STOP_PASS"
    if hi < BUDGET:
        return "STOP_FAIL"
    return "INCONCLUSIVE" if n >= MAX_REPS else "CONTINUE"


def should_skip_tests(changes):
    if not changes:
        return False
    for item in changes:
        if len(item) != 2:
            return False
        status, path = item
        if status not in ("A", "M", "D"):
            return False
        if not isinstance(path, str) or not path:
            return False
        if ".." in path.split("/"):
            return False
        if path.startswith(".github/"):
            return False
        if not (path.startswith("docs/") or path.startswith("hdae/") or path.endswith(".md")):
            return False
    return True
