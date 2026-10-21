"""Entry point Claude Code runs for every context-graph hook event."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_graph.hooks import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
