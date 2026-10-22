from .pipeline import *  # noqa
import sys, json
from . import pipeline as P
cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
if cmd == "run": P.run()
elif cmd == "train": P.train("champion")
elif cmd == "backtest": print(json.dumps(P.backtest(), indent=1, default=str))
elif cmd == "refresh": P.D.refresh_all()
else: print("usage: python -m ai_trader [run|train|backtest|refresh]")
