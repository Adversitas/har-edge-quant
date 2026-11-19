"""Command line: ``haredge download`` and ``haredge run``."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import data, experiment


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="haredge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download UCI HAR and cache it as .npz")
    d.add_argument("--data-dir", type=Path, default=Path("data"))
    r = sub.add_parser("run", help="train, quantise, prune and benchmark")
    r.add_argument("--data-dir", type=Path, default=Path("data"))
    r.add_argument("--reports", type=Path, default=Path("reports"))
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--epochs", type=int, default=40)
    r.add_argument("--synthetic", action="store_true", help="tiny synthetic data (CI smoke run)")
    args = p.parse_args(argv)
    if args.cmd == "download":
        print(data.download(args.data_dir))
    else:
        res = experiment.run(args.data_dir, args.reports, args.seeds, args.epochs, args.synthetic)
        print(experiment.render_markdown(res))


if __name__ == "__main__":
    main()
