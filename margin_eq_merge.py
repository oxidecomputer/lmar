#!/usr/bin/env python3
"""margin_eq_merge.py
Merge PCIe EQ settings with lane margin summary data.

Takes a margin_summary.txt output (with --test mode B/D/F column) and
a pcie_eq_dump.txt JSON file, and adds EQ settings columns to matching
bridge rows.

Matching logic:
- Only rows with "bridge" in Description are augmented
- Device from EQ file (e.g., "N9", "M2A") is matched to Description
  (e.g., "N9(J) bridge", "M.2 East(A) bridge")
- Lane number in margin summary is used to index into the lanes array
"""

__version__ = "0.1.0"

import csv
import io
import json
import re
import sys
from argparse import ArgumentParser
from typing import Dict, List, Optional, Any


# Device name mapping from EQ file device names to Description patterns
# EQ device -> (pattern to match in Description, must also have "bridge")
DEVICE_MAP = {
    "M2A": r"M\.2 East\(A\)",
    "M2B": r"M\.2 West\(B\)",
    "N0": r"N0\([A-Z]\)",
    "N1": r"N1\([A-Z]\)",
    "N2": r"N2\([A-Z]\)",
    "N3": r"N3\([A-Z]\)",
    "N4": r"N4\([A-Z]\)",
    "N5": r"N5\([A-Z]\)",
    "N6": r"N6\([A-Z]\)",
    "N7": r"N7\([A-Z]\)",
    "N8": r"N8\([A-Z]\)",
    "N9": r"N9\([A-Z]\)",
    "T6": r"T6",
}


def load_eq_data(eq_file: str) -> Dict[str, Any]:
    """Load PCIe EQ dump file (multiple concatenated JSON objects).

    Returns a dict keyed by device name.
    """
    devices = {}
    with open(eq_file, 'r') as f:
        content = f.read()

    # The file contains multiple JSON objects concatenated together.
    # We need to parse them one by one using a streaming approach.
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        # Skip whitespace
        while pos < len(content) and content[pos] in ' \t\n\r':
            pos += 1
        if pos >= len(content):
            break

        try:
            obj, consumed = decoder.raw_decode(content[pos:])
            devices[obj['device']] = obj
            pos += consumed
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error at position {pos}: {e}", file=sys.stderr)
            break

    return devices


def parse_margin_summary(summary_file: str) -> List[Dict[str, str]]:
    """Parse the margin summary tabulated output into a list of row dicts.

    Returns list of dicts with column headers as keys.
    Uses the separator line (dashes) to determine column boundaries.
    """
    rows = []

    with open(summary_file, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        return rows

    header_line = lines[0]
    sep_line = lines[1]

    # Find column boundaries from the separator line
    # Each column is a sequence of dashes separated by spaces
    col_ranges = []
    in_col = False
    col_start = 0
    for i, ch in enumerate(sep_line):
        if ch == '-':
            if not in_col:
                col_start = i
                in_col = True
        else:
            if in_col:
                col_ranges.append((col_start, i))
                in_col = False
    if in_col:
        col_ranges.append((col_start, len(sep_line)))

    # Extract header names using column ranges
    headers = []
    for start, end in col_ranges:
        header = header_line[start:end].strip()
        headers.append(header)

    # Parse data lines using column positions
    for line in lines[2:]:
        if not line.strip():
            continue

        row = {}
        for i, (start, end) in enumerate(col_ranges):
            if i < len(headers):
                # For last column, take rest of line
                if i == len(col_ranges) - 1:
                    value = line[start:].strip()
                else:
                    value = line[start:end].strip()
                row[headers[i]] = value

        if row:
            rows.append(row)

    return rows


def match_device(description: str, eq_devices: Dict[str, Any]) -> Optional[str]:
    """Find matching EQ device for a Description.

    Only matches if Description contains "bridge".
    Returns the device name if found, None otherwise.
    """
    if 'bridge' not in description.lower():
        return None

    for device_name, pattern in DEVICE_MAP.items():
        if device_name in eq_devices and re.search(pattern, description):
            return device_name

    return None


def format_eq_columns(eq_device: Dict[str, Any], lane: int) -> Dict[str, str]:
    """Extract EQ columns for a specific lane.

    Returns dict with column names and values.
    """
    result = {}

    # Preset search info (same for all lanes of this device)
    preset_search = eq_device.get('preset_search', {})
    result['PS_Mask'] = preset_search.get('mask', '')
    result['PS_Presets'] = ','.join(preset_search.get('presets', []))

    # Lane-specific data
    lanes = eq_device.get('lanes', [])
    if lane < len(lanes):
        lane_data = lanes[lane]

        # best_eq_settings
        best_eq = lane_data.get('best_eq_settings', {})
        result['BEQ_Raw'] = best_eq.get('raw', '')
        result['BEQ_Preset'] = str(best_eq.get('preset', ''))
        result['BEQ_Pre'] = str(best_eq.get('precursor', ''))
        result['BEQ_Cur'] = str(best_eq.get('cursor', ''))
        result['BEQ_Post'] = str(best_eq.get('postcursor', ''))
        result['BEQ_FOM'] = str(best_eq.get('fom', ''))

        # lc_cntl5
        lc = lane_data.get('lc_cntl5', {})
        result['LC5_Raw'] = lc.get('raw', '')
        result['LC5_Preset'] = str(lc.get('preset', ''))
        result['LC5_Pre'] = str(lc.get('precursor', ''))
        result['LC5_Cur'] = str(lc.get('cursor', ''))
        result['LC5_Post'] = str(lc.get('postcursor', ''))
        result['LC5_UsePreset'] = str(lc.get('use_preset', ''))
    else:
        # Lane not found in EQ data
        result['BEQ_Raw'] = ''
        result['BEQ_Preset'] = ''
        result['BEQ_Pre'] = ''
        result['BEQ_Cur'] = ''
        result['BEQ_Post'] = ''
        result['BEQ_FOM'] = ''
        result['LC5_Raw'] = ''
        result['LC5_Preset'] = ''
        result['LC5_Pre'] = ''
        result['LC5_Cur'] = ''
        result['LC5_Post'] = ''
        result['LC5_UsePreset'] = ''

    return result


def merge_and_output(rows: List[Dict[str, str]], eq_devices: Dict[str, Any],
                     out_file: Optional[str]) -> None:
    """Merge EQ data into margin summary rows and output as CSV."""

    # Define output columns
    base_cols = ['Vendor ID', 'Device ID', 'Lane', 'Time margin (% UI)',
                 'Voltage margin (V)', 'Description', 'B/D/F']
    eq_cols = ['PS_Mask', 'PS_Presets', 'BEQ_Raw', 'BEQ_Preset', 'BEQ_Pre',
               'BEQ_Cur', 'BEQ_Post', 'BEQ_FOM', 'LC5_Raw', 'LC5_Preset',
               'LC5_Pre', 'LC5_Cur', 'LC5_Post', 'LC5_UsePreset']

    all_cols = base_cols + eq_cols

    # Build output rows
    output_rows = []
    for row in rows:
        out_row = {col: row.get(col, '') for col in base_cols}

        # Try to match and merge EQ data
        device_name = match_device(row.get('Description', ''), eq_devices)
        if device_name:
            lane = int(row.get('Lane', 0))
            eq_cols_data = format_eq_columns(eq_devices[device_name], lane)
            out_row.update(eq_cols_data)
        else:
            # No match - empty EQ columns
            for col in eq_cols:
                out_row[col] = ''

        output_rows.append(out_row)

    # Write CSV output
    if out_file:
        with open(out_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_cols)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"Wrote {out_file}", file=sys.stderr)
    else:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(output_rows)
        print(output.getvalue())


def main(argv: List[str]) -> None:
    parser = ArgumentParser(
        description="Merge PCIe EQ settings with lane margin summary data.")
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "summary",
        help="Input margin summary file (from margin_summary.py --test)"
    )
    parser.add_argument(
        "--eq", default="pcie_eq.json",
        help="PCIe EQ dump file in JSON format (default: pcie_eq.json)"
    )
    parser.add_argument(
        "-o", "--out", default=None,
        help="Output file path (default: stdout)"
    )

    ns = parser.parse_args(argv)

    # Load data
    try:
        eq_devices = load_eq_data(ns.eq)
        print(f"Loaded {len(eq_devices)} devices from EQ file: {list(eq_devices.keys())}",
              file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: EQ file not found: {ns.eq}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading EQ file: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        rows = parse_margin_summary(ns.summary)
        print(f"Loaded {len(rows)} rows from margin summary", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Summary file not found: {ns.summary}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading summary file: {e}", file=sys.stderr)
        sys.exit(1)

    # Merge and output
    merge_and_output(rows, eq_devices, ns.out)


if __name__ == "__main__":
    main(sys.argv[1:])
