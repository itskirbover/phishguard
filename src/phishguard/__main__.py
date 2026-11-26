"""Allow invoking PhishGuard via ``python -m phishguard``."""

from phishguard.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
