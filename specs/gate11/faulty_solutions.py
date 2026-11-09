"""حلول خاطئة عمداً. كل واحد يجب أن يفشل في اختبار واحد على الأقل."""

FAULTY = {
    "decide_next": {
        "always_continue": lambda ratios, hung: "CONTINUE",
        "hang_after_empty": lambda ratios, hung: ("CONTINUE" if not ratios else ("HANG" if hung else "STOP_PASS" if min(ratios[1:] or [0]) >= 0.96 else "STOP_FAIL")),
        "warmup_inclusive_floor": lambda ratios, hung: ("HANG" if hung else "STOP_FAIL_EARLY" if ratios and ratios[0] <= 0.90 else "CONTINUE"),
        "budget_strict_pass": lambda ratios, hung: ("HANG" if hung else "CONTINUE" if not ratios else "STOP_PASS" if ratios and min(ratios[1:] or [0]) > 0.96 else "CONTINUE"),
    },
    "should_skip_tests": {
        "always_true": lambda changes: True,
        "always_false": lambda changes: False,
        "no_dotdot_check": lambda changes: bool(changes) and all(s in ("A","M","D") and p and (p.startswith("docs/") or p.startswith("hdae/") or p.endswith(".md")) and not p.startswith(".github/") for s, p in changes),
        "no_status_check": lambda changes: bool(changes) and all(p and ".." not in p.split("/") and not p.startswith(".github/") and (p.startswith("docs/") or p.startswith("hdae/") or p.endswith(".md")) for s, p in changes),
        "no_github_check": lambda changes: bool(changes) and all(s in ("A","M","D") and p and ".." not in p.split("/") and (p.startswith("docs/") or p.startswith("hdae/") or p.endswith(".md")) for s, p in changes),
        "docs_prefix_no_slash": lambda changes: bool(changes) and all(s in ("A","M","D") and p and ".." not in p.split("/") and not p.startswith(".github/") and (p.startswith("docs") or p.startswith("hdae/") or p.endswith(".md")) for s, p in changes),
    },
}
