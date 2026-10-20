#!/usr/bin/env python3
"""CLI wrapper for the *unchanged* local CXRMate-ED preprocessing pipeline."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description='Prepare linked MIMIC-CXR / MIMIC-IV-ED data.')
    parser.add_argument('--physionet-dir', type=Path, required=True,
                        help='Parent containing mimic-cxr/, mimic-cxr-jpg/, and mimic-iv-ed/.')
    parser.add_argument('--database-dir', type=Path, required=True)
    parser.add_argument('--num-workers', type=int, default=None)
    args = parser.parse_args()
    if not args.physionet_dir.is_dir():
        parser.error(f'Not a directory: {args.physionet_dir}')
    from cxrmate_ed.prepare_dataset import prepare_dataset
    prepare_dataset(physionet_dir=str(args.physionet_dir),
                    database_dir=str(args.database_dir), num_workers=args.num_workers)


if __name__ == '__main__':
    main()
