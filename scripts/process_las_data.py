#!/usr/bin/env python3
"""
scripts/process_las_data.py - Standalone LAS Log Batch Processing and Validation Tool

Encapsulates concurrent parsing of LAS (.las) well log files, extracting well headers,
curves, and depth ranges, and compiling summary metrics into structured reports.

Usage:
    python scripts/process_las_data.py --input-dir path/to/las_files --output report.json
    python scripts/process_las_data.py file1.las file2.las --verbose
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure repository root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_models import WellData
from utils.las_parser import parse_las

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("process_las_data")


class LASBatchProcessor:
    """Processes multiple LAS files concurrently and compiles validation/summary reports."""

    def __init__(self, filepaths: List[Union[str, Path]], max_workers: Optional[int] = None):
        self.filepaths = [Path(fp).resolve() for fp in filepaths]
        self.max_workers = max_workers or min(os.cpu_count() or 4, 16)
        self.wells: List[WellData] = []
        self.failed_files: Dict[str, str] = {}
        self.processed_count = 0

    def _process_single_file(self, file_path: Path) -> Optional[WellData]:
        if not file_path.is_file():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        if file_path.suffix.lower() != ".las":
            raise ValueError(f"Unsupported extension '{file_path.suffix}', expected .las")

        return parse_las(file_path)

    def run(self) -> Dict[str, Any]:
        """Execute concurrent processing across all specified files."""
        las_files = [p for p in self.filepaths if p.suffix.lower() == ".las"]
        if not las_files:
            logger.warning("No .las files found in input set.")
            return {"wells": [], "failed_files": {}, "summary": {"total": 0, "successful": 0, "failed": 0}}

        total = len(las_files)
        logger.info(f"Starting batch processing of {total} LAS files with {self.max_workers} worker threads.")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, fp): fp
                for fp in las_files
            }

            for future in as_completed(future_to_file):
                fp = future_to_file[future]
                self.processed_count += 1
                try:
                    well = future.result()
                    if well:
                        self.wells.append(well)
                        logger.debug(f"[{self.processed_count}/{total}] Parsed: {well.name} ({fp.name})")
                    else:
                        self.failed_files[fp.name] = "Parser returned None"
                except Exception as exc:
                    logger.error(f"[{self.processed_count}/{total}] Error parsing {fp.name}: {exc}")
                    self.failed_files[fp.name] = str(exc)

        logger.info(f"Completed processing: {len(self.wells)} successful, {len(self.failed_files)} failed.")
        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """Compile structured summary statistics for all processed wells."""
        well_summaries = []
        for w in self.wells:
            curves_info = {}
            for col in w.data.columns:
                curves_info[col] = {
                    "count": int(w.data[col].count()),
                    "min": float(w.data[col].min()) if not w.data[col].empty else None,
                    "max": float(w.data[col].max()) if not w.data[col].empty else None,
                }
            well_summaries.append({
                "name": w.name,
                "header": w.header,
                "top_depth": w.top_depth,
                "bottom_depth": w.bottom_depth,
                "curves": list(w.data.columns),
                "curve_stats": curves_info,
            })

        return {
            "summary": {
                "total_files": len(self.filepaths),
                "successful": len(self.wells),
                "failed": len(self.failed_files),
            },
            "failed_files": self.failed_files,
            "wells": well_summaries,
        }


def main():
    parser = argparse.ArgumentParser(description="Batch process LAS well log files.")
    parser.add_argument("files", nargs="*", help="Optional list of specific .las files to parse")
    parser.add_argument("--input-dir", "-i", type=str, help="Directory containing .las files to process")
    parser.add_argument("--output", "-o", type=str, help="Path to write output summary JSON report")
    parser.add_argument("--max-workers", "-w", type=int, default=None, help="Maximum worker threads (default: CPU count)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    target_files: List[Path] = []
    if args.input_dir:
        input_path = Path(args.input_dir)
        if not input_path.is_dir():
            logger.error(f"Input directory not found: {args.input_dir}")
            sys.exit(1)
        target_files.extend(input_path.glob("*.las"))
        target_files.extend(input_path.glob("*.LAS"))

    for f in args.files:
        target_files.append(Path(f))

    # Remove duplicates
    unique_files = list(dict.fromkeys(target_files))

    if not unique_files:
        logger.error("No LAS files provided. Use --input-dir or pass filenames directly.")
        parser.print_help()
        sys.exit(1)

    processor = LASBatchProcessor(unique_files, max_workers=args.max_workers)
    report = processor.run()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Summary report written to {out_path.resolve()}")
    else:
        print(json.dumps(report["summary"], indent=2))
        if report["failed_files"]:
            print("Failed files:", json.dumps(report["failed_files"], indent=2))


if __name__ == "__main__":
    main()
