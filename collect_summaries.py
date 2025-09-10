#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import io
import math
import os
import re
import sys
import textwrap
from contextlib import redirect_stdout
from statistics import mean, stdev
from typing import List, Dict, Any, Tuple, Optional

try:
    from tabulate import tabulate
    HAVE_TABULATE = True
except Exception:
    HAVE_TABULATE = False


ROW_RE_SPLIT = re.compile(r"\s{2,}")  # split on 2+ spaces

# Old patterns (kept for backward compat)
PAT_SN_1 = re.compile(r".*margin_summary_(?P<sn>[^_]+)_(?P<run>\d+)\.txt$", re.IGNORECASE)
PAT_SN_2 = re.compile(r".*margin-(?P<sn>[^_]+)_summary_(?P<run>\d+)\.txt$", re.IGNORECASE)
# New, tolerant pattern: capture SN even if label is not numeric (e.g., _plain, _auto, _token)
PAT_SN_ANY = re.compile(r".*margin_summary_(?P<sn>[A-Za-z0-9]+)_[A-Za-z0-9\-]+\.txt$", re.IGNORECASE)


def _looks_like_board(dirname: str) -> bool:
    """Board serials must contain letters AND digits (e.g., BRM13250013)."""
    return bool(re.match(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+$", dirname))


def _find_board_dir(path: str) -> Optional[str]:
    """Walk up from a path to find a directory whose name looks like a board serial."""
    cur = os.path.abspath(path)
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    while True:
        name = os.path.basename(cur)
        if _looks_like_board(name):
            return name
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def parse_sn_from_filename(path: str) -> Tuple[str, Optional[str]]:
    """
    Return (SN, label) from summary filename. 'label' may be a run number or a token.
    If the filename doesn't carry a plausible SN (e.g., margin_summary_raw_1.txt),
    fall back to the nearest ancestor directory that looks like a board serial.
    """
    fname = os.path.basename(path)

    for pat in (PAT_SN_1, PAT_SN_2, PAT_SN_ANY):
        m = pat.match(fname)
        if m:
            sn = m.group("sn")
            run = m.groupdict().get("run")
            if _looks_like_board(sn):
                return sn, run
            break  # matched but SN looked like 'raw' -> use dir fallback

    bd = _find_board_dir(path)
    if bd:
        return bd, None
    return "UNKNOWN", None


def description_to_port(desc: str) -> str:
    """Keep Description verbatim (bridge vs device remain distinct for (Port, Lane) stats)."""
    return desc.strip()


def canonical_port(name: str) -> str:
    """
    Normalize a port label to '<LETTERS><NUMBER>' with no leading zeros, uppercased.
    Examples:
      'N7', 'n07', 'N7 bridge', 'N7-A', ' n7  '  -> 'N7'
    """
    s = name.strip().upper()
    s = re.sub(r"\s+BRIDGE$", "", s)
    m = re.match(r"^([A-Z]+)\s*0*([0-9]+)", s)
    if m:
        return f"{m.group(1)}{int(m.group(2))}"
    return s


def load_pci_ids(path: str) -> Tuple[Dict[int, str], Dict[Tuple[int, int], str]]:
    """
    Parse pci.ids (from https://pci-ids.ucw.cz/) into:
      vend_map[vendor] -> vendor name
      dev_map[(vendor, device)] -> device name
    """
    vend_map: Dict[int, str] = {}
    dev_map: Dict[Tuple[int, int], str] = {}
    cur_vendor: Optional[int] = None
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if not line.startswith("\t"):
                m = re.match(r"^([0-9A-Fa-f]{4})\s+(.+)$", line)
                if m:
                    cur_vendor = int(m.group(1), 16)
                    vend_map[cur_vendor] = m.group(2).strip()
            else:
                m = re.match(r"^\t([0-9A-Fa-f]{4})\s+(.+)$", line)
                if m and cur_vendor is not None:
                    dev = int(m.group(1), 16)
                    dev_map[(cur_vendor, dev)] = m.group(2).strip()
    return vend_map, dev_map


def trunc(s: Optional[str], n: int) -> str:
    if not s:
        return ""
    # ASCII-only ellipsis for wide compat.
    return s if len(s) <= n else (s[: max(1, n - 3)] + "...")


def parse_summary_file(path: str) -> List[Dict[str, Any]]:
    """
    Parse a single summary*.txt file emitted by margin_summary.py / analyze.py summarize.
    Expects columns:
      Vendor ID | Device ID | Lane | Time margin (% UI) | Voltage margin (V) | Description
    Returns a list of dicts with SN, Port, Lane, Vendor, Device, Width, Height
    (dropping rows where width/height is NaN or 0.0).
    """
    sn, _ = parse_sn_from_filename(path)
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # Find header
    hdr_idx = None
    for i, ln in enumerate(lines):
        if "Vendor ID" in ln and "Device ID" in ln and "Lane" in ln and "Time margin" in ln and "Voltage margin" in ln:
            hdr_idx = i
            break
    if hdr_idx is None:
        return rows

    # Data starts after header + underline row (usually dashes)
    i = hdr_idx + 1
    if i < len(lines) and re.match(r"^-+\s+-+\s+-+\s+-+\s+-+\s+-+\s*$", lines[i]):
        i += 1

    for ln in lines[i:]:
        s = ln.strip()
        if not s:
            continue
        if set(s) <= {"-", " "}:
            continue
        cols = ROW_RE_SPLIT.split(s)
        if len(cols) < 6:
            # fallback: loose split then join description tail
            parts = s.split()
            if len(parts) < 6:
                continue
            cols = parts[:5] + [" ".join(parts[5:])]
        elif len(cols) > 6:
            cols = cols[:5] + [" ".join(cols[5:])]

        vendor_s, device_s, lane_s, width_s, height_s, desc = cols
        try:
            vendor = int(vendor_s, 16) if vendor_s.lower().startswith("0x") else int(vendor_s)
            device = int(device_s, 16) if device_s.lower().startswith("0x") else int(device_s)
            lane = int(lane_s)
            width = float(width_s)
            height = float(height_s)
        except Exception:
            continue

        # Drop NaNs and zero-width/zero-height (invalid/no data collected)
        if math.isnan(width) or math.isnan(height) or width == 0.0 or height == 0.0:
            continue

        port = description_to_port(desc)
        rows.append({
            "SN": sn,
            "Port": port,      # keep 'bridge' suffix if present
            "Lane": lane,
            "Vendor": vendor,
            "Device": device,
            "Width": width,    # %UI
            "Height": height,  # V
        })
    return rows


def safe_stats(vals: List[float]) -> Tuple[float, float, float, Optional[float]]:
    if not vals:
        return (float("nan"), float("nan"), float("nan"), None)
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0], None)
    try:
        return (min(vals), max(vals), mean(vals), stdev(vals))
    except Exception:
        return (min(vals), max(vals), mean(vals), None)


def group_by(rows: List[Dict[str, Any]], keys: Tuple[str, ...]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    out: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        k = tuple(r[k] for k in keys)
        out.setdefault(k, []).append(r)
    return out


def gate_ports_by_limits(rows: List[Dict[str, Any]],
                         width_limit: float,
                         height_limit: float) -> List[Dict[str, Any]]:
    """
    Keep a port label only if lanes {0,1,2,3} are all present AND, for each lane,
    min(Width) >= width_limit and min(Height) >= height_limit across any duplicate
    rows for that lane. Bridge and device entries are treated as distinct ports.
    """
    required_lanes = {0, 1, 2, 3}
    by_sn_port = group_by(rows, ("SN", "Port"))
    passing_keys = set()

    for key, items in by_sn_port.items():
        lanes: Dict[int, List[Dict[str, Any]]] = {}
        for r in items:
            lanes.setdefault(r["Lane"], []).append(r)

        if set(lanes.keys()) != required_lanes:
            continue

        ok = True
        for ln in required_lanes:
            wmin = min(x["Width"] for x in lanes[ln])
            hmin = min(x["Height"] for x in lanes[ln])
            if wmin < width_limit or hmin < height_limit:
                ok = False
                break
        if ok:
            passing_keys.add(key)

    return [r for r in rows if (r["SN"], r["Port"]) in passing_keys]


def print_table(headers: List[str], rows: List[List[Any]]) -> None:
    """Legacy single-header printer. Kept for compatibility."""
    if HAVE_TABULATE:
        print(tabulate(rows, headers=headers))
    else:
        widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
        fmt = "  ".join("{:%d}" % w for w in widths)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in widths))
        for r in rows:
            print(fmt.format(*r))


def print_grouped_headers_table(
    header_top: List[str],
    header_sub: List[str],
    rows: List[List[Any]]
) -> None:
    """
    Print a two-line header:
      top labels group columns (e.g., 'Width (%UI)' over 4 subcolumns),
      sub labels show 'min mean max sd'.
    """
    # Normalize rows to strings
    srows: List[List[str]] = [[str(c) for c in row] for row in rows]
    # Compute per-column widths considering both header lines and rows
    num_cols = len(header_top)
    assert len(header_sub) == num_cols
    widths: List[int] = []
    for i in range(num_cols):
        col_vals = [header_top[i], header_sub[i]] + [r[i] for r in srows]
        widths.append(max(len(str(v)) for v in col_vals))
    fmt = "  ".join("{:%d}" % w for w in widths)

    # Print headers and a separator
    print(fmt.format(*header_top))
    print(fmt.format(*header_sub))
    print("  ".join("-" * w for w in widths))

    # Print rows
    for r in srows:
        print(fmt.format(*r))


def build_output_for_files(files: List[str], args: argparse.Namespace) -> str:
    """
    Run the stats pipeline for a set of summary files and return the text output.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        # --- begin pipeline ---

        files_sorted = sorted(set(files))
        if not files_sorted:
            print("No input summary files.")
            return buf.getvalue()

        combined: List[Dict[str, Any]] = []
        for path in files_sorted:
            combined.extend(parse_summary_file(path))

        if not combined:
            print("No usable rows (after dropping NaNs/zeros).")
            return buf.getvalue()

        # Port-level gating: require lanes 0..3 and per-lane mins ≥ limits
        combined = gate_ports_by_limits(combined, args.width_limit, args.height_limit)
        if not combined:
            print("No rows after port-level gating.")
            return buf.getvalue()


        # Optional name maps (CSV may include names even if --names is not set)
        vend_map: Dict[int, str] = {}
        dev_map: Dict[Tuple[int, int], str] = {}
        have_name_maps = False
        if args.pci_ids:
            try:
                vend_map, dev_map = load_pci_ids(args.pci_ids)
                have_name_maps = True
            except Exception:
                have_name_maps = False

        # Optional CSV dump of the filtered dataset
        if args.csv_out:
            with open(args.csv_out, "w", newline="", encoding="utf-8") as fh:
                fieldnames = ["SN", "Port", "Lane", "Vendor", "Device", "Width", "Height",
                              "VendorName", "DeviceName", "PASS"]
                w = csv.DictWriter(fh, fieldnames=fieldnames)
                w.writeheader()
                for r in combined:
                    row = dict(r)
                    row["PASS"] = "Y" if (r["Width"] >= args.pass_width and r["Height"] > args.pass_height) else "N"
                    if have_name_maps:
                        row["VendorName"] = vend_map.get(r["Vendor"], "")
                        row["DeviceName"] = dev_map.get((r["Vendor"], r["Device"]), "")
                    else:
                        row["VendorName"] = ""
                        row["DeviceName"] = ""
                    w.writerow(row)
            print(f"Wrote CSV: {args.csv_out}")

        def mark_min(val: float, threshold: float, *, strict: bool = False) -> str:
            if math.isnan(val):
                return "nan"
            s = f"{val:.3f}"
            fail = (val <= threshold) if strict else (val < threshold)
            return s + "!" if fail else s

        # Stats by (Port, Lane) — device vs bridge remain distinct
        by_port_lane = group_by(combined, ("Port", "Lane"))
        rows_pl: List[List[Any]] = []
        for key, items in sorted(by_port_lane.items()):
            widths = [it["Width"] for it in items]
            heights = [it["Height"] for it in items]
            n = len(items)
            wmin, wmax, wmean, wstd = safe_stats(widths)
            hmin, hmax, hmean, hstd = safe_stats(heights)

            vname: Optional[str] = None
            dname: Optional[str] = None
            if args.names:
                pairs = {(it["Vendor"], it["Device"]) for it in items}
                if len(pairs) == 1:
                    v, d = next(iter(pairs))
                    vname = trunc(vend_map.get(v, "unknown"), args.name_width)
                    dname = trunc(dev_map.get((v, d), "unknown"), args.name_width)
                else:
                    vname = "mixed"
                    dname = "mixed"

            pass_flag = "Y" if (wmin >= args.pass_width and hmin > args.pass_height) else "N"

            row: List[Any] = [
                key[0], key[1], n,
                mark_min(wmin, args.pass_width, strict=False), f"{wmean:.3f}", f"{wmax:.3f}", (f"{wstd:.3f}" if wstd is not None else "NA"),
                mark_min(hmin, args.pass_height, strict=True), f"{hmean:.3f}", f"{hmax:.3f}", (f"{hstd:.3f}" if hstd is not None else "NA"),
            ]
            if args.names:
                row += [vname, dname]
            row += [pass_flag]
            rows_pl.append(row)

        print("\nStats by (Port, Lane):")
        header_top_pl_base = ["Port", "Lane", "N", "Width (%UI)", "", "", "", "Height (V)", "", "", ""]
        header_sub_pl_base = ["", "", "", "min", "mean", "max", "sd", "min", "mean", "max", "sd"]
        if args.names:
            header_top_pl = header_top_pl_base + ["Vendor name", "Device name", "PASS"]
            header_sub_pl = header_sub_pl_base + ["", "", ""]
        else:
            header_top_pl = header_top_pl_base + ["PASS"]
            header_sub_pl = header_sub_pl_base + [""]

        print_grouped_headers_table(header_top=header_top_pl, header_sub=header_sub_pl, rows=rows_pl)

        # Stats by (Vendor, Device) with omit filters
        omit_keys: set[str] = set()
        for p in (args.omit_ports or []):
            if p.strip():
                omit_keys.add(canonical_port(p))
        if args.omit_n7_9:
            omit_keys.update({"N7", "N8", "N9"})

        combined_vd = [r for r in combined if canonical_port(r["Port"]) not in omit_keys]
        by_vd = group_by(combined_vd, ("Vendor", "Device"))
        rows_vd: List[List[Any]] = []
        for key, items in sorted(by_vd.items()):
            widths = [it["Width"] for it in items]
            heights = [it["Height"] for it in items]
            n = len(items)
            wmin, wmax, wmean, wstd = safe_stats(widths)
            hmin, hmax, hmean, hstd = safe_stats(heights)
            vname = trunc(vend_map.get(key[0], "unknown"), args.name_width) if args.names else None
            dname = trunc(dev_map.get((key[0], key[1]), "unknown"), args.name_width) if args.names else None
            pass_flag = "Y" if (wmin >= args.pass_width and hmin > args.pass_height) else "N"

            row_vd: List[Any] = [
                f"0x{key[0]:04x}", f"0x{key[1]:04x}", n,
                mark_min(wmin, args.pass_width, strict=False), f"{wmean:.3f}", f"{wmax:.3f}", (f"{wstd:.3f}" if wstd is not None else "NA"),
                mark_min(hmin, args.pass_height, strict=True), f"{hmean:.3f}", f"{hmax:.3f}", (f"{hstd:.3f}" if hstd is not None else "NA"),
            ]
            if args.names:
                row_vd += [vname, dname]
            row_vd += [pass_flag]
            rows_vd.append(row_vd)

        title = "\nStats by (Vendor, Device):"
        if omit_keys:
            if omit_keys == {"N7", "N8", "N9"}:
                title += "  (omitting N7-9)"
            else:
                title += "  (omitting " + ", ".join(sorted(omit_keys)) + ")"
        print(title)

        header_top_vd_base = ["Vendor", "Device", "N",
                              "Width (%UI)", "", "", "",
                              "Height (V)", "", "", ""]
        header_sub_vd_base = ["", "", "",
                              "min", "mean", "max", "sd",
                              "min", "mean", "max", "sd"]
        if args.names:
            header_top_vd = header_top_vd_base + ["Vendor name", "Device name", "PASS"]
            header_sub_vd = header_sub_vd_base + ["", "", ""]
        else:
            header_top_vd = header_top_vd_base + ["PASS"]
            header_sub_vd = header_sub_vd_base + [""]

        print_grouped_headers_table(header_top=header_top_vd, header_sub=header_sub_vd, rows=rows_vd)

        # --- end pipeline ---
    return buf.getvalue()


# ---------- Discovery helpers for scan mode ----------

def _find_board_dirs(scan_root: str) -> Dict[str, str]:
    """
    If scan_root itself looks like a board, return {board: scan_root}.
    Otherwise, return all immediate subdirs that look like boards.
    """
    scan_root = os.path.abspath(scan_root)
    base = os.path.basename(os.path.normpath(scan_root))
    if _looks_like_board(base):
        return {base: scan_root}
    out: Dict[str, str] = {}
    for d in sorted(os.listdir(scan_root)):
        p = os.path.join(scan_root, d)
        if os.path.isdir(p) and _looks_like_board(d):
            out[d] = p
    return out


def _collect_board_summary_files(board_dir: str) -> List[str]:
    """
    Find summary files anywhere under board_dir (recursive):
      - **/margin_summary_*.txt
      - **/margin-*_summary_*.txt   (legacy)
    """
    patterns = [
        os.path.join(board_dir, "**", "margin_summary_*.txt"),
        os.path.join(board_dir, "**", "margin-*_summary_*.txt"),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    return sorted(set(files))


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Collect margin summaries and compute stats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:

              # Direct mode: glob inputs and print to stdout
              {prog} summaries/margin_summary_BRM13250002_*.txt

              # Direct mode with --out
              {prog} summaries/*.txt --out stats_BRM13250002_all.txt

              # Enforce dataset gating and tweak PASS thresholds
              {prog} --width-limit 12.5 --height-limit 0.085 --pass-width 30 --pass-height 0.015 summaries/*.txt

              # Append vendor/device names (requires pci.ids)
              {prog} --names --pci-ids /path/to/pci.ids summaries/*.txt

              # Scan a single board and write per-board stats into that board dir
              {prog} --scan-root BRM13250013 --outdir BRM13250013/stats

              # Scan a parent containing many boards, write per-board stats into one outdir
              {prog} --scan-root . --outdir ./_stats_all

              # Also emit a combined 'all boards' file
              {prog} --scan-root . --outdir ./_stats_all --combine-all

              # Emit only the combined 'all boards' file (skip per-board)
              {prog} --scan-root . --outdir ./_stats_all --only-combined
            """).format(prog=os.path.basename(__file__)),
    )
    # Scan control
    ap.add_argument("--scan-root", default=None,
                    help="Scan a board directory or a parent containing many boards.")
    ap.add_argument("--combine-all", action="store_true",
                    help="In scan mode, also write a single combined stats_all_boards.txt across all boards.")
    ap.add_argument("--only-combined", action="store_true",
                    help="In scan mode, write ONLY the combined stats_all_boards.txt (skip per-board outputs).")
    ap.add_argument("--outdir", default=".",
                    help="Directory to write stats in scan mode, or with --out (default: current dir).")
    ap.add_argument("--out", default=None,
                    help="Direct mode only: write a single combined stats output to this file.")
    # Direct-mode inputs (files/dirs/globs)
    ap.add_argument("--input", nargs="+", default=None,
                    help="Direct mode: glob(s) of summary files to read (compat).")
    ap.add_argument("inputs", nargs="*",
                    help="Direct mode: files/dirs/globs of summaries to read.")
    # Options
    ap.add_argument("--csv-out", default=None,
                    help="Write the filtered, per-row dataset used for stats to this CSV file.")
    ap.add_argument("--width-limit", type=float, default=0.0,
                    help="Minimum acceptable eye width in %%UI for dataset gating (default: 0.0).")
    ap.add_argument("--height-limit", type=float, default=0.0,
                    help="Minimum acceptable eye height in V for dataset gating (default: 0.0).")
    ap.add_argument("--pass-width", type=float, default=30.0,
                    help="PASS threshold for width in %%UI used in summary tables (default: 30.0).")
    ap.add_argument("--pass-height", type=float, default=0.015,
                    help="PASS threshold for height in V used in summary tables (default: 0.015).")
    ap.add_argument("--omit-ports", nargs="+", default=[],
                    help="Port labels to omit from Vendor/Device stats (canonicalized: e.g., 'n07', 'N7 bridge' -> N7).")
    ap.add_argument("--omit-n7-9", action="store_true",
                    help="Convenience flag to omit ports N7, N8, and N9 from Vendor/Device stats.")
    ap.add_argument("--pci-ids", default=None,
                    help="Path to pci.ids for name resolution (download from pci-ids.ucw.cz).")
    ap.add_argument("--names", action="store_true",
                    help="Append vendor/device names to tables (requires --pci-ids).")
    ap.add_argument("--name-width", type=int, default=25,
                    help="Max chars for name columns (default: 25).")
    ap.add_argument("-V", "--version", action="version", version="%(prog)s 2.3")

    args = ap.parse_args()

    # Names sanity
    if args.names and not args.pci_ids:
        ap.error("--names requires --pci-ids (path to a pci.ids file)")

    # Scan mode -> per-board and/or combined outputs
    if args.scan_root:
        boards = _find_board_dirs(args.scan_root)
        if not boards:
            ap.error(f"No board directories found under {args.scan_root}")
        os.makedirs(args.outdir, exist_ok=True)

        # Per-board outputs (unless only-combined)
        any_written = False
        if not args.only_combined:
            for board, board_dir in boards.items():
                files = _collect_board_summary_files(board_dir)
                if not files:
                    print(f"[{board}] No summaries found under {board_dir}")
                    continue
                text = build_output_for_files(files, args)
                outpath = os.path.join(args.outdir, f"stats_{board}_all.txt")
                with open(outpath, "w", encoding="utf-8") as fh:
                    fh.write(text.rstrip() + "\n")
                print(f"Wrote {outpath}")
                any_written = True

        # Combined output (if requested)
        if args.combine_all or args.only_combined:
            all_files: List[str] = []
            for _, board_dir in boards.items():
                all_files.extend(_collect_board_summary_files(board_dir))
            all_files = sorted(set(all_files))
            if not all_files:
                print("No summaries found under any board directory")
                if not any_written:
                    sys.exit(2)
                return
            text_all = build_output_for_files(all_files, args)
            out_all = os.path.join(args.outdir, "stats_all_boards.txt")
            with open(out_all, "w", encoding="utf-8") as fh:
                fh.write(text_all.rstrip() + "\n")
            print(f"Wrote {out_all}")
            any_written = True

        if not any_written:
            # No matches across all boards -> exit non-zero so CI scripts can catch it
            sys.exit(2)
        return

    # ---------- Direct mode ----------
    # Resolve inputs from positional or --input globs; support dirs and globs, recursively.
    resolved: List[str] = []
    pats: List[str] = []
    if args.inputs:
        pats.extend(args.inputs)
    if args.input:
        pats.extend(args.input)

    if not pats:
        ap.error("either provide inputs (files/dirs/globs) or use --scan-root")

    for pat in pats:
        if os.path.isdir(pat):
            # Recursive search inside the directory
            resolved.extend(glob.glob(os.path.join(pat, "**", "margin_summary_*.txt"), recursive=True))
            resolved.extend(glob.glob(os.path.join(pat, "**", "margin-*_summary_*.txt"), recursive=True))
            continue
        # Otherwise, treat as glob (could be recursive if pattern has **)
        matches = glob.glob(pat, recursive=True)
        if matches:
            resolved.extend(matches)
        else:
            if os.path.isfile(pat):
                resolved.append(pat)

    if not resolved:
        ap.error("no files matched the given inputs; see --help for examples")

    text = build_output_for_files(resolved, args)

    if args.out:
        os.makedirs(args.outdir, exist_ok=True)
        outpath = args.out if os.path.isabs(args.out) else os.path.join(args.outdir, args.out)
        with open(outpath, "w", encoding="utf-8") as fh:
            fh.write(text.rstrip() + "\n")
        print(f"Wrote {outpath}")
    else:
        print(text.rstrip())


if __name__ == "__main__":
    main()

