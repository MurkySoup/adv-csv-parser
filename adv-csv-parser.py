#!/usr/bin/env python3

"""
adv-csv-parser.py - Version 1.1-Beta (Do Not Distribute)
By Rick Pelletier (galiagante@gmail.com), 12 November 2025
Last updated: 12 November 2025
AI Assist: https://grok.com

An Advanced CSV analyzer/parser/extractor with an extensible CLI. Features:
- Efficient parsing of arbitrary CSV files (handles dialects, large files)
- Streaming processing to minimize memory usage
- Modular CLI using argparse with subcommands
- Type hints, comprehensive docstrings, PEP 8 compliant
- Robust error handling and resource management
- Extensible design for future processors

Usage:
# ./python adv-csv-parser.py info path/to/file.csv
# ./python adv-csv-parser.py count path/to/file.csv
# ./python adv-csv-parser.py head path/to/file.csv --lines 10
# ./python adv-csv-parser.py select path/to/file.csv Name Age --delimiter ','

Linter: ruff check adv-csv-parser.py --extend-select F,B,UP
"""

from __future__ import annotations
import argparse
import csv
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TextIO
from collections.abc import Sequence

class CSVProcessor(ABC):
    """Abstract base class for CSV processing strategies."""

    @abstractmethod
    def process(self, reader: csv.DictReader) -> Any:
        """Process the CSV data and return result."""
        pass

class InfoProcessor(CSVProcessor):
    """Extracts structural metadata from CSV."""

    def process(self, reader: csv.DictReader) -> dict[str, Any]:
        fieldnames = reader.fieldnames
        if not fieldnames:
            return {"columns": 0, "sample": None, "dialect": None}

        sample_row = None
        row_count = 0

        for row in reader:
            row_count += 1
            if row_count == 1:
                sample_row = row
            if row_count >= 5:  # Limit sample inspection
                break

        return {
            "columns": len(fieldnames),
            "column_names": fieldnames,
            "sample_row": sample_row,
            "estimated_rows": "unknown (streaming)" if row_count == 5 else row_count,
        }

class CountProcessor(CSVProcessor):
    """Counts total number of rows (excluding header)."""

    def process(self, reader: csv.DictReader) -> int:
        return sum(1 for _ in reader)

class HeadProcessor(CSVProcessor):
    """Returns first N rows of data."""

    def __init__(self, lines: int = 5):
        self.lines = max(1, lines)

    def process(self, reader: csv.DictReader) -> list[dict[str, str]]:
        return [row for i, row in enumerate(reader) if i < self.lines]

class SelectProcessor(CSVProcessor):
    """Selects and outputs specified columns from all rows."""

    def __init__(self, fields: Sequence[str]):
        self.fields = list(fields)

        if not self.fields:
            raise ValueError("At least one field must be specified for select")

    def process(self, reader: csv.DictReader) -> list[dict[str, str]]:
        available = set(reader.fieldnames or [])
        missing = [f for f in self.fields if f not in available]

        if missing:
            raise ValueError(f"Requested fields not found in CSV: {', '.join(missing)}")

        selected = []

        for row in reader:
            selected.append({field: row.get(field, "") for field in self.fields})

        return selected

def detect_dialect(sample: str, delimiter: str | None = None) -> csv.Dialect:
    """
    Detect CSV dialect from sample text.

    Args:
        sample: First few lines of CSV content.
        delimiter: Optional explicit delimiter.

    Returns:
        Configured csv.Dialect instance.
    """
    if delimiter:
        class ExplicitDialect(csv.excel):
            delimiter = delimiter

        return ExplicitDialect

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")

        return dialect
    except csv.Error:
        return csv.excel  # Fallback to comma-separated

def open_csv_file(
    path: Path,
    delimiter: str | None = None,
    encoding: str = "utf-8",
) -> TextIO:
    """
    Open CSV file with proper encoding and error handling.

    Args:
        path: Path to CSV file.
        delimiter: Optional delimiter override.
        encoding: File encoding (default: utf-8).

    Returns:
        Open file handle in text mode.

    Raises:
        FileNotFoundError: If file does not exist.
        PermissionError: If file is not readable.
        UnicodeDecodeError: If encoding fails.
    """
    try:
        return path.open("r", encoding=encoding, newline="")
    except UnicodeDecodeError as e:
        # Try common alternative encodings
        for alt_encoding in ("utf-8-sig", "latin1", "cp1252"):
            try:
                return path.open("r", encoding=alt_encoding, newline="")
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(
            encoding, e.object, e.start, e.end, f"Failed to decode with {encoding} or fallbacks"
        ) from e

def parse_csv_stream(
    file_handle: TextIO,
    processor: CSVProcessor,
    delimiter: str | None = None,
) -> Any:
    """
    Parse CSV from open file handle using streaming reader.

    Args:
        file_handle: Open text file handle.
        processor: CSVProcessor instance to handle data.
        delimiter: Optional delimiter override.

    Returns:
        Result from processor.process().
    """
    # Read sample for dialect detection
    sample = file_handle.read(8192)
    file_handle.seek(0)

    dialect = detect_dialect(sample, delimiter)

    reader = csv.DictReader(file_handle, dialect=dialect)

    if not reader.fieldnames:
        raise ValueError("CSV file appears to have no columns (empty header)")

    return processor.process(reader)

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="An Advanced CSV Analyzer/Parser/Extractor Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # info command
    info_parser = subparsers.add_parser(
        "info", help="Show CSV structure and metadata"
    )
    info_parser.add_argument("file", type=Path, help="Path to CSV file")
    info_parser.add_argument(
        "--delimiter", type=str, help="Force delimiter (e.g., ',' or '\\t')"
    )

    # count command
    count_parser = subparsers.add_parser(
        "count", help="Count number of data rows (excludes header)"
    )
    count_parser.add_argument("file", type=Path, help="Path to CSV file")
    count_parser.add_argument(
        "--delimiter", type=str, help="Force delimiter (e.g., ',' or '\\t')"
    )

    # head command
    head_parser = subparsers.add_parser(
        "head", help="Show first N rows of data"
    )
    head_parser.add_argument("file", type=Path, help="Path to CSV file")
    head_parser.add_argument(
        "-n", "--lines", type=int, default=5, help="Number of rows to display"
    )
    head_parser.add_argument(
        "--delimiter", type=str, help="Force delimiter (e.g., ',' or '\\t')"
    )

    # select command
    select_parser = subparsers.add_parser(
        "select", help="Extract specific columns from all rows"
    )
    select_parser.add_argument("file", type=Path, help="Path to CSV file")
    select_parser.add_argument(
        "fields", nargs="+", help="One or more column names to select"
    )
    select_parser.add_argument(
        "--delimiter", type=str, help="Force delimiter (e.g., ',' or '\\t')"
    )
    select_parser.add_argument(
        "--tsv", action="store_true", help="Output as TSV instead of CSV"
    )

    return parser

def print_table(rows: list[dict[str, str]], use_tsv: bool = False) -> None:
    """
    Print list of dicts as aligned table (TSV or CSV).

    Args:
        rows: List of row dictionaries.
        use_tsv: If True, use tab delimiter; else comma.
    """
    if not rows:
        print("(no rows)")

        return

    delimiter = "\t" if use_tsv else ","
    headers = rows[0].keys()
    print(delimiter.join(headers))

    for row in rows:
        print(delimiter.join(str(row.get(h, "")) for h in headers))

def main(argv: list[str] | None = None) -> int:
    """
    Main entry point.

    Returns:
        Exit status code (0 for success).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        file_path: Path = args.file

        if not file_path.is_file():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            return 1

        delimiter = args.delimiter.replace("\\t", "\t") if args.delimiter else None

        if args.command == "info":
            processor: CSVProcessor = InfoProcessor()
        elif args.command == "count":
            processor = CountProcessor()
        elif args.command == "head":
            processor = HeadProcessor(lines=args.lines)
        elif args.command == "select":
            processor = SelectProcessor(fields=args.fields)
        else:
            parser.error(f"Unknown command: {args.command}")

        with open_csv_file(file_path, delimiter=delimiter) as f:
            result = parse_csv_stream(f, processor, delimiter=delimiter)

        # Output handling
        if args.command == "info":
            import json
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "count":
            print(result)
        elif args.command in ("head", "select"):
            print_table(result, use_tsv=getattr(args, "tsv", False))
        else:
            print(result)

        return 0

    except FileNotFoundError:
        print(f"Error: File not found: {args.file}", file=sys.stderr)

        return 1
    except PermissionError:
        print(f"Error: Permission denied reading file: {args.file}", file=sys.stderr)

        return 1
    except UnicodeDecodeError as e:
        print(f"Error: Cannot decode file (tried multiple encodings): {e}", file=sys.stderr)

        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)

        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)

        return 1

if __name__ == "__main__":
    sys.exit(main())

# end of script
