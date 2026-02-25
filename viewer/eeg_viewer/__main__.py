from __future__ import annotations

import argparse
from pathlib import Path

from .qt_compat import QtWidgets
from .io import scan_recordings
from .ui.main_window import EEGViewer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=None,
        help="Dataset root to recursively scan for .npy recordings",
    )
    ap.add_argument(
        "--participants-tsv",
        type=str,
        default=None,
        help="participants.tsv with column participants_ID",
    )
    ap.add_argument("--file", type=str, default=None, help="Open a specific .npy file")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else None
    tsv = (
        Path(args.participants_tsv).expanduser().resolve()
        if args.participants_tsv
        else None
    )
    initial_file = Path(args.file).expanduser().resolve() if args.file else None

    recs = (
        scan_recordings(root)
        if (root and root.exists())
        else scan_recordings(Path.cwd())
    )

    app = QtWidgets.QApplication([])
    win = EEGViewer(recs, participants_tsv=tsv, initial_file=initial_file)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
