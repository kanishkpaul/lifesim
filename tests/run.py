"""Zero-dependency test harness: `python3 -m tests.run`.

Discovers every `test_*.py` module in this package, runs every top-level
`test_*` function, and reports pass/fail. Test logic stays pure stdlib so it runs
anywhere; if pytest happens to be installed, the same files work under it too.
"""

from __future__ import annotations

import importlib
import pkgutil
import traceback


def discover():
    import tests
    for info in pkgutil.iter_modules(tests.__path__):
        if info.name.startswith("test_"):
            yield importlib.import_module(f"tests.{info.name}")


def main() -> int:
    passed = failed = 0
    failures: list[str] = []
    for mod in discover():
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            label = f"{mod.__name__}.{name}"
            try:
                fn()
                passed += 1
                print(f"  PASS  {label}")
            except Exception as e:  # noqa: BLE001 - harness reports everything
                failed += 1
                failures.append(f"{label}: {e}\n{traceback.format_exc()}")
                print(f"  FAIL  {label}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    if failures:
        print("\n" + "=" * 66 + "\nFAILURES\n" + "=" * 66)
        for f in failures:
            print(f)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
