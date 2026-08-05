"""Enable ``python -m wasmagent_protocol`` as an alias for the drift CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
