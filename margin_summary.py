#!/usr/bin/env python3
"""margin_summary.py
Summarize lane margining results (files, dirs, or archives).
Derived from analyze.py:summarize, with archive + scan + robust output naming.

Behavior:
- One summary file PER DATASET:
  * dataset = a single archive (.zip/.tar/.tar.gz/.tgz), OR
  * the set of plain margin-results* files directly in a directory.
- Outputs go to --outdir (default "."). We no longer dump the table to stdout
  unless --out is used with exactly one dataset.

Output naming:
- Run subdir: margin_summary_<BOARD>_<RUN>[ _i].txt
- Board root:
    If archive basename ends with "_<RUN>" before extension (e.g., margin_1.zip,
    margin_<date>_2.tar.gz, margin_<serial>_3.zip), use that RUN.
    If multiple datasets share the same RUN, suffix with _2, _3, ...
    If no RUN in name, auto-number per board: _1.txt, _2.txt, ...
- Plain (unarchived) set at a directory:
    In run dir: ..._<RUN>.txt (or ..._<RUN>_2.txt if multiple)
    At board root: ..._plain.txt (dedup as plain_2, plain_3)
"""

from tabulate import tabulate
from argparse import ArgumentParser
import os
import os.path
import numpy as np
import re
import sys
import zipfile
import tarfile
import tempfile
from typing import List, Dict, Tuple, Optional
import shutil

# Keep the same COSMO_MAP and parsing style from analyze.py (unchanged).
COSMO_MAP = {
        (0x00, 0x1, 0x3): "M.2 East(A) bridge",
            (0x02, 0x0, 0x0): "M.2 East(A)",
        (0x20, 0x1, 0x3): "M.2 West(B) bridge",
            (0x21, 0x0, 0x0): "M.2 West(B)",
        (0x40, 0x1, 0x1): "N9(J) bridge",
            (0x43, 0x0, 0x0): "N9(J)",
        (0x40, 0x1, 0x2): "N5(F) bridge",
            (0x44, 0x0, 0x0): "N5(F)",
        (0x40, 0x1, 0x3): "N4(E) bridge",
            (0x45, 0x0, 0x0): "N4(E)",
        (0x60, 0x1, 0x1): "N8(I) bridge",
            (0x61, 0x0, 0x0): "N8(I)",
        (0x60, 0x1, 0x2): "N7(H) bridge",
            (0x62, 0x0, 0x0): "N7(H)",
        (0x60, 0x1, 0x3): "N6(G) bridge",
            (0x63, 0x0, 0x0): "N6(G)",
        (0xa0, 0x1, 0x1): "N0(A) bridge",
            (0xa2, 0x0, 0x0): "N0(A)",
        (0xa0, 0x1, 0x2): "N1(B) bridge",
            (0xa3, 0x0, 0x0): "N1(B)",
        (0xa0, 0x1, 0x3): "N2(C) bridge",
            (0xa4, 0x0, 0x0): "N2(C)",
        (0xa0, 0x1, 0x4): "N3(D) bridge",
            (0xa5, 0x0, 0x0): "N3(D)",
        (0xc0, 0x1, 0x1): "T6 bridge",
            (0xc1, 0x0, 0x4): "T6",
        (0xe0, 0x1, 0x2): "Backplane bridge",
            (0xe3, 0x0, 0x0): "Backplane",
}

def compute_margin(results: Dict[str, np.ndarray], key: str) -> float:
    """
    Eye diagram margin calculator:
    - Finds the contiguous PASS run centered around 0 on the independent axis.
    - Calculates margin as 2 × min(left_edge, right_edge) from center.
    - For time: 2 × min(abs(left), abs(right)) in %UI
    - For voltage: 2 × min(abs(top), abs(bottom)) in V
    - This represents the symmetric eye opening around center.
    """
    passed = np.asarray(results["passed"], dtype=bool)
    independent_axis = np.asarray(results[key])
    n_points = len(passed)

    if n_points == 0:
        return np.nan
    n_passes = int(np.sum(passed))
    if n_passes == 0:
        return 0.0

    # Find contiguous PASS runs as (start_idx, end_idx), inclusive.
    transitions = np.diff(passed.astype(np.int8))
    run_starts = list(np.where(transitions == 1)[0] + 1)
    run_ends = list(np.where(transitions == -1)[0])

    if passed[0]:
        run_starts = [0] + run_starts
    if passed[-1]:
        run_ends = run_ends + [n_points - 1]

    pass_runs = list(zip(run_starts, run_ends))
    if not pass_runs:
        return 0.0

    # Index of the point closest to 0 on the sweep axis (the center).
    center_index = int(np.argmin(np.abs(independent_axis)))

    # Prefer the run that contains the center.
    selected_run = None
    for start_idx, end_idx in pass_runs:
        if start_idx <= center_index <= end_idx:
            selected_run = (start_idx, end_idx)
            break

    # If center is not in any pass run, choose the run nearest to center.
    if selected_run is None:
        def distance_to_center(run: Tuple[int, int]) -> int:
            start_idx, end_idx = run
            if center_index < start_idx:
                return start_idx - center_index
            if center_index > end_idx:
                return center_index - end_idx
            return 0
        selected_run = min(pass_runs, key=distance_to_center)

    start_idx, end_idx = selected_run

    # If all points pass, return the full range
    if n_passes == n_points:
        return float(np.ptp(independent_axis))

    # Calculate eye margin as 2 × min(left_edge, right_edge) from center
    # Left edge: distance from center to start of pass region
    # Right edge: distance from center to end of pass region
    left_edge = abs(independent_axis[start_idx] - independent_axis[center_index])
    right_edge = abs(independent_axis[end_idx] - independent_axis[center_index])

    # Return 2× the minimum edge (symmetric eye opening)
    return 2.0 * float(min(left_edge, right_edge))


def load_results(file: str, pass_err_cnt: Optional[int] = None) -> Dict[str, object]:
    """Load the lane margining results from a file.

    Returns a dict with results for time and voltage sweeps.

    Args:
        file: Path to the margin results file
        pass_err_cnt: Maximum error count for a point to be considered PASS.
                      If provided, ignores the PASS column and recalculates based on count <= pass_err_cnt.
                      If None (default), uses the PASS column from the file.
    """
    with open(file, "rb") as f:
        vendor_id = int(f.readline().rstrip().rsplit()[-1], base=16)
        device_id = int(f.readline().rstrip().rsplit()[-1], base=16)
        lane = int(f.readline().rstrip().rsplit()[-1])

    # Extract bXX-dYY-fZZ triplet from the filename to derive description
    bus = dev = func = 0
    match = re.search(r"b([0-9a-f]+)-d([0-9a-f]+)-f([0-9a-f]+)", file, flags=re.IGNORECASE)
    if match:
        bus, dev, func = (int(x, 16) for x in match.groups())
    descr = COSMO_MAP.get((bus, dev, func), "?")

    # Load the table body (tab-separated) skipping the 3 header lines + 1 blank
    results = np.loadtxt(file, skiprows=4, delimiter="\t")

    # Handle empty files (header only, no data rows)
    if results.size == 0 or results.ndim == 1:
        # Return empty results that will yield NaN margins
        time_results = dict(
            time=np.array([]),
            count=np.array([], dtype=np.int64),
            duration=np.array([]),
            passed=np.array([], dtype=bool),
            xlabel="Time (% UI)",
        )
        voltage_results = dict(
            voltage=np.array([]),
            count=np.array([], dtype=np.int64),
            duration=np.array([]),
            passed=np.array([], dtype=bool),
            xlabel="Voltage (V)",
        )
        return dict(
            vendor_id=vendor_id,
            device_id=device_id,
            bus=bus, dev=dev, func=func,
            descr=descr,
            lane=lane,
            time=time_results,
            voltage=voltage_results,
        )

    (time, voltage, duration, count, passed_col) = range(5)
    is_time = results[:, time] != 0
    is_voltage = np.logical_not(is_time)

    # Determine PASS status: either from count threshold or original PASS column
    if pass_err_cnt is not None:
        # Recalculate PASS based on error count threshold: PASS if count <= pass_err_cnt
        time_passed = results[is_time, count] <= pass_err_cnt
        voltage_passed = results[is_voltage, count] <= pass_err_cnt
    else:
        # Use original PASS column from the file
        time_passed = results[is_time, passed_col].astype(bool)
        voltage_passed = results[is_voltage, passed_col].astype(bool)

    time_results = dict(
        time=results[is_time, time],
        count=results[is_time, count].astype(np.int64),
        duration=results[is_time, duration],
        passed=time_passed.astype(bool),
        xlabel="Time (% UI)",
    )
    voltage_results = dict(
        voltage=results[is_voltage, voltage],
        count=results[is_voltage, count].astype(np.int64),
        duration=results[is_voltage, duration],
        passed=voltage_passed.astype(bool),
        xlabel="Voltage (V)",
    )
    return dict(
        vendor_id=vendor_id,
        device_id=device_id,
        bus=bus, dev=dev, func=func,
        descr=descr,
        lane=lane,
        time=time_results,
        voltage=voltage_results,
    )


def summarize(files: List[str], pass_count_required: int, pass_err_cnt: Optional[int] = None,
              test_mode: bool = False, archive_path: Optional[str] = None,
              limits: Optional[Tuple[float, float]] = None) -> str:
    """Return the summary table string for the provided files.

    Args:
        files: List of margin-results files to process
        pass_count_required: Pass count filter (when not using pass_err_cnt)
        pass_err_cnt: Error count threshold for PASS determination
        test_mode: Enable test mode features (BDFL columns, NaN file extraction)
        archive_path: Original archive path if files came from an archive
        limits: Optional (width_limit_UI, height_limit_V) tuple for PASS/FAIL determination
    """
    table = dict(vendor_id=[], device_id=[],
                 lane=[], time_margin=[], voltage_margin=[],
                 descr=[])

    if test_mode:
        table['bdf'] = []

    if limits:
        table['pass'] = []
        width_limit, height_limit = limits

    # Track statistics for summary message
    total_lanes = 0
    nan_count = 0
    expected_lanes = 96

    for file in files:
        results = load_results(file, pass_err_cnt=pass_err_cnt)

        # If using pass_err_cnt, use results as-is
        # If using original PASS column, apply pass_count_required filtering
        if pass_err_cnt is not None:
            # Using error count threshold - use results directly
            time_gated = results["time"]
            voltage_gated = results["voltage"]
        else:
            # Using original PASS column - apply pass_count_required filter
            time_gated = dict(results["time"])
            time_gated["passed"] = np.logical_and(results["time"]["passed"],
                                                  results["time"]["count"] == pass_count_required)

            voltage_gated = dict(results["voltage"])
            voltage_gated["passed"] = np.logical_and(results["voltage"]["passed"],
                                                     results["voltage"]["count"] == pass_count_required)

        time_margin = compute_margin(time_gated, "time")
        voltage_margin = compute_margin(voltage_gated, "voltage")

        # Check for bad data and print warnings to stderr
        descr = results["descr"]
        lane = results["lane"]
        bus = results["bus"]
        dev = results["dev"]
        func = results["func"]
        bdf_str = f"{bus:02x}/{dev:x}/{func:x}"
        has_nan = False

        total_lanes += 1

        if np.isnan(time_margin):
            print(f"WARNING: {descr}, {bdf_str}, lane {lane}: time margin is NaN (no valid data)", file=sys.stderr)
            has_nan = True
        elif time_margin == 0.0:
            print(f"WARNING: {descr}, {bdf_str}, lane {lane}: time margin is 0.0", file=sys.stderr)

        if np.isnan(voltage_margin):
            print(f"WARNING: {descr}, {bdf_str}, lane {lane}: voltage margin is NaN (no valid data)", file=sys.stderr)
            has_nan = True
        elif voltage_margin == 0.0:
            print(f"WARNING: {descr}, {bdf_str}, lane {lane}: voltage margin is 0.0", file=sys.stderr)

        if has_nan:
            nan_count += 1

        # Extract problem files from archives and show location
        if has_nan:
            if archive_path:
                _extract_problem_file(archive_path, file)
            else:
                # Plain file - just show its location
                print(f"  Problem file location: {file}", file=sys.stderr)

        table["vendor_id"].append("0x{:04x}".format(results["vendor_id"]))
        table["device_id"].append("0x{:04x}".format(results["device_id"]))
        table["descr"].append(descr)
        table["lane"].append(lane)
        table["time_margin"].append(time_margin)
        table["voltage_margin"].append(voltage_margin)

        # TEST MODE: Add B/D/F column
        if test_mode:
            table["bdf"].append(bdf_str)

        # LIMITS: Add PASS column
        if limits:
            # Check if both margins meet limits (handle NaN as fail)
            time_pass = not np.isnan(time_margin) and time_margin >= width_limit
            voltage_pass = not np.isnan(voltage_margin) and voltage_margin >= height_limit
            overall_pass = time_pass and voltage_pass
            table["pass"].append("Y" if overall_pass else "N")

    # Print summary of dataset processing
    print(f"Lanes Processed: {total_lanes}", file=sys.stderr)
    if total_lanes < expected_lanes:
        print(f"WARNING: Processed lanes {total_lanes} < expected of {expected_lanes} for full sled", file=sys.stderr)

    # Build headers based on mode and options
    headers = [
        "Vendor ID",
        "Device ID",
        "Lane",
        "Time margin (% UI)",
        "Voltage margin (V)",
        "Description",
    ]

    if test_mode:
        headers.append("B/D/F")

    if limits:
        headers.append("PASS")

    headers = tuple(headers)

    return tabulate(table, headers=headers)


# ---------- File collection helpers ----------

def _extract_problem_file(archive_path: str, temp_file_path: str) -> None:
    """Extract a problem file from an archive to the archive's directory.

    Args:
        archive_path: Path to the original archive (.zip or .tar*)
        temp_file_path: Path to the file in the temporary extraction directory
    """
    # Determine the filename from the temp path
    filename = os.path.basename(temp_file_path)
    archive_dir = os.path.dirname(archive_path)
    base_output_path = os.path.join(archive_dir, filename)

    # Find a unique filename if it already exists (.1, .2, .3, etc.)
    output_path = base_output_path
    counter = 1
    while os.path.exists(output_path):
        output_path = f"{base_output_path}.{counter}"
        counter += 1

    # Copy the file from temp directory to archive directory
    try:
        shutil.copy2(temp_file_path, output_path)
        print(f"  Extracted problem file to: {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"  WARNING: Failed to extract problem file: {e}", file=sys.stderr)


def _basename(name: str) -> str:
    return os.path.basename(name).rstrip("/")


def _is_margin_results(name: str) -> bool:
    # Match "margin-results*" (case-insensitive) on the basename
    return _basename(name).lower().startswith("margin-results")


def _is_archive(name: str) -> bool:
    n = name.lower()
    return n.endswith(".zip") or n.endswith(".tar") or n.endswith(".tgz") or n.endswith(".tar.gz")


def _safe_join(root: str, name: str) -> str:
    # Prevent path traversal when extracting archives
    name = name.lstrip("/").replace("\\", "/")
    parts = []
    for p in name.split("/"):
        if p in ("", ".", ".."):
            continue
        parts.append(p)
    return os.path.join(root, *parts)


def collect_from_archive(archive_path: str, tmpdir: str) -> List[str]:
    """Extract margin-results* files from a zip/tar archive into tmpdir and return paths."""
    out: List[str] = []
    # Zip
    is_zip = False
    try:
        is_zip = zipfile.is_zipfile(archive_path)
    except Exception:
        is_zip = False
    if is_zip:
        with zipfile.ZipFile(archive_path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if not _is_margin_results(info.filename):
                    continue
                dest = _safe_join(tmpdir, info.filename)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zf.open(info, "r") as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                out.append(dest)
        return out

    # Tar
    is_tar = False
    try:
        is_tar = tarfile.is_tarfile(archive_path)
    except Exception:
        is_tar = False
    if is_tar:
        with tarfile.open(archive_path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                if not _is_margin_results(member.name):
                    continue
                dest = _safe_join(tmpdir, member.name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with open(dest, "wb") as dst:
                    dst.write(src.read())
                out.append(dest)
        return out

    # Not an archive
    return []


def collect_plain_files(dirpath: str) -> List[str]:
    """Non-recursive: return margin-results* files directly in dirpath."""
    out: List[str] = []
    try:
        for fn in sorted(os.listdir(dirpath)):
            full = os.path.join(dirpath, fn)
            if os.path.isfile(full) and _is_margin_results(full):
                out.append(full)
    except FileNotFoundError:
        pass
    return out


# ---------- Identification helpers (board/run/labels) ----------

def _looks_like_board(dirname: str) -> bool:
    # Must contain at least one letter (so numeric run dirs like "1" won't match).
    return bool(re.match(r"^(?=.*[A-Za-z])[A-Za-z0-9]+$", dirname))


def _is_run_dir(name: str) -> bool:
    return bool(re.match(r"^[0-9]+$", name))


def _remove_archive_ext(basename: str) -> str:
    """Strip .tar.gz/.tgz/.tar/.zip in that order."""
    n = basename
    for ext in (".tar.gz", ".tgz", ".tar", ".zip"):
        if n.lower().endswith(ext):
            return n[: -len(ext)]
    return n


def _parse_run_from_name(basename_noext: str) -> Optional[str]:
    """
    Return trailing run number if name ends with _<digits>, else None.
    Examples: margin_1 -> "1", margin_2024-08-01_2 -> "2", brm13250010_7 -> "7"
    """
    m = re.search(r"_([0-9]+)$", basename_noext)
    return m.group(1) if m else None


def _parse_timestamp_from_name(name: str) -> Optional[str]:
    """
    Extract HH-MM-SS from names containing 1986-MM-DDTHH-MM-SS pattern.
    Uses 1986 as anchor (system startup time). Works with suffixes like -1.zip, _info.zip.

    Examples:
        margin-1986-12-28T21-50-48.zip -> "21-50-48"
        margin-1986-12-28T21-50-48-1.zip -> "21-50-48"
        margin-1986-12-28T21-50-48_info.zip -> "21-50-48"
    """
    # Look for 1986-MM-DDTHH-MM-SS pattern (8 chars for HH-MM-SS after T)
    match = re.search(r'1986-\d{2}-\d{2}T(\d{2}-\d{2}-\d{2})', name)
    if match:
        return match.group(1)  # Return just HH-MM-SS
    return None


def _unique_suffix(counter: Dict[str, int], key: str) -> str:
    """
    Manage suffixes _2, _3, ... per key (usually '<BOARD>_<RUN>' or '<BOARD>_plain' or '<BOARD>_auto').
    Returns '' for the first occurrence, '_2' for the second, etc.
    """
    n = counter.get(key, 0) + 1
    counter[key] = n
    return "" if n == 1 else f"_{n}"


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


# ---------- Scan mode orchestration ----------

def _scan_board(board: str, board_dir: str, outdir: str, pass_count_required: int, pass_err_cnt: Optional[int], test_mode: bool = False, limits: Optional[Tuple[float, float]] = None) -> None:
    """Write summaries for one board directory."""
    os.makedirs(outdir, exist_ok=True)
    suffix_counter: Dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="margin_summary_scan_") as tmpdir:
        # First handle run subdirs (e.g., 1, 2, ...)
        run_dirs = [d for d in sorted(os.listdir(board_dir))
                    if os.path.isdir(os.path.join(board_dir, d)) and _is_run_dir(d)]

        for run in run_dirs:
            this_dir = os.path.join(board_dir, run)
            # Gather archives directly within run dir
            archives = [os.path.join(this_dir, fn) for fn in sorted(os.listdir(this_dir))
                        if os.path.isfile(os.path.join(this_dir, fn)) and _is_archive(fn)]
            # Gather plain margin-results files directly within run dir
            plain = collect_plain_files(this_dir)

            datasets: List[Tuple[str, List[str], Optional[str]]] = []  # (label, files, archive_path)

            # Each archive is a separate dataset
            for arch in archives:
                files = collect_from_archive(arch, tmpdir)
                if files:
                    datasets.append((f"{board}_{run}", files, arch))

            # Plain files (if any) are their own dataset
            if plain:
                datasets.append((f"{board}_{run}", plain, None))

            if not datasets:
                continue

            # If multiple datasets share same '<board>_<run>' label, suffix _2, _3, ...
            for label, files, archive_path in datasets:
                base_key = label  # e.g., 'BRM22250002_1'
                suf = _unique_suffix(suffix_counter, base_key)
                outname = f"margin_summary_{label}{suf}.txt"
                text = summarize(files, pass_count_required, pass_err_cnt, test_mode=test_mode, archive_path=archive_path, limits=limits)
                outpath = os.path.join(outdir, outname)
                with open(outpath, "w", encoding="utf-8") as fh:
                    fh.write(text + "\n")
                print(f"Wrote {outpath}")

        # Now handle archives/files directly under the board dir (no run dir)
        top_archives = [os.path.join(board_dir, fn) for fn in sorted(os.listdir(board_dir))
                        if os.path.isfile(os.path.join(board_dir, fn)) and _is_archive(fn)]
        top_plain = collect_plain_files(board_dir)

        # Each top-level archive becomes a dataset; try to infer RUN from basename.
        for arch in top_archives:
            base = _basename(arch)
            stem = _remove_archive_ext(base)
            run = _parse_run_from_name(stem)  # may be None
            if run is not None:
                label = f"{board}_{run}"
                base_key = label
            else:
                # No explicit run -> auto-number per board
                label = f"{board}_auto"
                base_key = label

            files = collect_from_archive(arch, tmpdir)
            if not files:
                continue

            # For explicit RUN: suffix duplicates with _2, _3...
            if base_key != f"{board}_auto":
                suf = _unique_suffix(suffix_counter, base_key)
                outname = f"margin_summary_{label}{suf}.txt"
            else:
                # auto case: generate ..._<n>.txt with n = 1,2,3...
                suf = _unique_suffix(suffix_counter, base_key)  # updates counter
                n = suffix_counter[base_key]
                outname = f"margin_summary_{board}_{n}.txt"  # no extra suffix

            text = summarize(files, pass_count_required, pass_err_cnt, test_mode=test_mode, archive_path=arch, limits=limits)
            outpath = os.path.join(outdir, outname)
            with open(outpath, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"Wrote {outpath}")

        # Top-level plain files as one dataset
        if top_plain:
            label = f"{board}_plain"
            suf = _unique_suffix(suffix_counter, label)
            outname = f"margin_summary_{label}{suf}.txt"
            text = summarize(top_plain, pass_count_required, pass_err_cnt, test_mode=test_mode, archive_path=None, limits=limits)
            outpath = os.path.join(outdir, outname)
            with open(outpath, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"Wrote {outpath}")


def scan_and_write(scan_root: str, outdir: str, pass_count_required: int, pass_err_cnt: Optional[int], test_mode: bool = False, limits: Optional[Tuple[float, float]] = None) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Support scan_root being either a single board dir or a parent of many board dirs.
    root_base = os.path.basename(os.path.normpath(scan_root))
    board_map: Dict[str, str] = {}

    if _looks_like_board(root_base):
        board_map[root_base] = os.path.abspath(scan_root)
    else:
        for d in sorted(os.listdir(scan_root)):
            p = os.path.join(scan_root, d)
            if os.path.isdir(p) and _looks_like_board(d):
                board_map[d] = os.path.abspath(p)

    if not board_map:
        print(f"No board directories found under {scan_root}", file=sys.stderr)
        sys.exit(2)

    for board, board_dir in board_map.items():
        _scan_board(board, board_dir, outdir, pass_count_required, pass_err_cnt, test_mode=test_mode, limits=limits)


# ---------- Direct mode (per-archive outputs; keep tempdir alive) ----------

def direct_mode(inputs: List[str], outdir: str, out_path: Optional[str], pass_count_required: int, pass_err_cnt: Optional[int], test_mode: bool = False, limits: Optional[Tuple[float, float]] = None) -> None:
    """Produce summaries for given inputs. Each archive/dataset -> its own file."""
    os.makedirs(outdir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="margin_summary_direct_") as tmpdir:
        datasets: List[Tuple[str, List[str], str, Optional[str]]] = []  # (logical_key, files, name_key, archive_path)

        # Collect datasets (extract into tmpdir)
        for path in inputs:
            if os.path.isdir(path):
                # Treat like scan: one dataset per archive under this dir (recursively),
                # plus any plain set directly in a directory.
                for dirpath, _, filenames in os.walk(path):
                    # archives in this dir
                    for fn in sorted(filenames):
                        full = os.path.join(dirpath, fn)
                        if os.path.isfile(full) and _is_archive(full):
                            files = collect_from_archive(full, tmpdir)
                            if not files:
                                continue
                            board = _find_board_dir(full) or "BOARD"
                            parent = os.path.basename(os.path.dirname(full))

                            # Try timestamp first, then run number, then auto
                            timestamp = _parse_timestamp_from_name(_basename(full))
                            if timestamp:
                                key = f"{board}_{timestamp}"
                                name_key = key
                            else:
                                run = parent if _is_run_dir(parent) else _parse_run_from_name(_remove_archive_ext(_basename(full)))
                                if run is not None:
                                    key = f"{board}_{run}"
                                    name_key = key
                                else:
                                    key = f"{board}_auto"
                                    name_key = key
                            datasets.append((key, files, name_key, full))
                    # plain set directly in dirpath
                    plain = collect_plain_files(dirpath)
                    if plain:
                        board = _find_board_dir(dirpath) or "BOARD"
                        parent = os.path.basename(dirpath)

                        # Try timestamp first, then run number, then plain
                        timestamp = _parse_timestamp_from_name(parent)
                        if timestamp:
                            key = f"{board}_{timestamp}"
                            name_key = key
                        elif _is_run_dir(parent):
                            key = f"{board}_{parent}"
                            name_key = key
                        else:
                            key = f"{board}_plain"
                            name_key = key
                        datasets.append((key, plain, name_key, None))
                continue

            if _is_archive(path):
                files = collect_from_archive(path, tmpdir)
                if files:
                    board = _find_board_dir(path) or "BOARD"
                    parent = os.path.basename(os.path.dirname(path))

                    # Try timestamp first, then run number, then auto
                    timestamp = _parse_timestamp_from_name(_basename(path))
                    if timestamp:
                        key = f"{board}_{timestamp}"
                        name_key = key
                    else:
                        run = parent if _is_run_dir(parent) else _parse_run_from_name(_remove_archive_ext(_basename(path)))
                        if run is not None:
                            key = f"{board}_{run}"
                            name_key = key
                        else:
                            key = f"{board}_auto"
                            name_key = key
                    datasets.append((key, files, name_key, path))
                continue

            # Plain single file (not typical): group by its parent dir
            if os.path.isfile(path) and _is_margin_results(path):
                board = _find_board_dir(path) or "BOARD"
                parent = os.path.basename(os.path.dirname(path))

                # Try timestamp first, then run number, then plain
                timestamp = _parse_timestamp_from_name(parent)
                if timestamp:
                    key = f"{board}_{timestamp}"
                    name_key = key
                elif _is_run_dir(parent):
                    key = f"{board}_{parent}"
                    name_key = key
                else:
                    key = f"{board}_plain"
                    name_key = key
                datasets.append((key, [path], name_key, None))

        if not datasets:
            print("No margin-results datasets found in inputs.", file=sys.stderr)
            sys.exit(2)

        # If --out is specified, require exactly one dataset, and write it here (while tmpdir exists)
        if out_path:
            if len(datasets) != 1:
                print("--out may be used only when exactly one dataset is provided.", file=sys.stderr)
                sys.exit(2)
            text = summarize(datasets[0][1], pass_count_required, pass_err_cnt, test_mode=test_mode, archive_path=datasets[0][3], limits=limits)

            # Use --out path as-is; only join with outdir if it's just a filename (no path separators)
            if os.path.isabs(out_path) or os.path.dirname(out_path):
                out_full = out_path
            else:
                # Just a filename without path - use outdir
                out_full = os.path.join(outdir, out_path)

            # Check if the output path is a directory
            if os.path.isdir(out_full):
                print(f"ERROR: --out must be a file path, not a directory: {out_full}", file=sys.stderr)
                print(f"  Either specify a filename: --out ./P1/summary.txt", file=sys.stderr)
                print(f"  Or omit --out to auto-generate filenames in --outdir", file=sys.stderr)
                sys.exit(2)

            with open(out_full, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"Wrote {out_full}")
            return

        # Otherwise, write auto-named files per dataset (while tmpdir exists)
        suffix_counter: Dict[str, int] = {}
        for key, files, name_key, archive_path in datasets:
            # name_key looks like '<BOARD>_<timestamp>' or '<BOARD>_<RUN>' or '<BOARD>_plain' or '<BOARD>_auto'
            # Split into board and identifier
            if '_' in name_key:
                parts = name_key.rsplit('_', 1)  # Split on last underscore
                board_part = parts[0]
                id_part = parts[1]

                if id_part == "auto":
                    # auto-numbered per board: ..._<n>.txt
                    _ = _unique_suffix(suffix_counter, name_key)  # updates counter
                    n = suffix_counter[name_key]
                    outname = f"{board_part}_margin_summary_{n}.txt"
                else:
                    # timestamp, run number, or "plain"
                    suf = _unique_suffix(suffix_counter, name_key)
                    outname = f"{board_part}_margin_summary_{id_part}{suf}.txt"
            else:
                # No underscore - just use name_key as-is (shouldn't happen, but fallback)
                suf = _unique_suffix(suffix_counter, name_key)
                outname = f"{name_key}_margin_summary{suf}.txt"

            text = summarize(files, pass_count_required, pass_err_cnt, test_mode=test_mode, archive_path=archive_path, limits=limits)
            outpath = os.path.join(outdir, outname)
            with open(outpath, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"Wrote {outpath}")


# ---------- Deep scan mode ----------

def deep_scan_mode(input_dir: str, outdir: str, pass_count_required: int,
                   pass_err_cnt: Optional[int], test_mode: bool,
                   limits: Optional[Tuple[float, float]], inspect: bool) -> None:
    """
    Deep scan mode: traverse directory tree from input_dir, collect all datasets,
    name outputs based on full path components + timestamp.

    Output naming: <component1>_<component2>_..._<HH-MM-SS>.txt
    Example: ./2WJV75RW/P6/margin-1986-12-28T21-50-48.zip -> 2WJV75RW_P6_21-50-48.txt
    """
    os.makedirs(outdir, exist_ok=True)
    input_abs = os.path.abspath(input_dir)

    with tempfile.TemporaryDirectory(prefix="margin_summary_deep_") as tmpdir:
        datasets: List[Tuple[str, List[str], Optional[str]]] = []  # (output_name, files, archive_path)
        suffix_counter: Dict[str, int] = {}

        # Walk the tree
        for dirpath, dirnames, filenames in os.walk(input_abs):
            # Check archives in this directory
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                if os.path.isfile(full) and _is_archive(full):
                    files = collect_from_archive(full, tmpdir)
                    if not files:
                        continue

                    # Build output name from path components
                    rel_path = os.path.relpath(dirpath, input_abs)
                    components = [c for c in rel_path.split(os.sep) if c and c != '.']

                    # Extract timestamp from archive name
                    timestamp = _parse_timestamp_from_name(fn)

                    # Build name
                    if components:
                        name_base = "_".join(components)
                        if timestamp:
                            output_name = f"{name_base}_margin_summary_{timestamp}.txt"
                        else:
                            output_name = f"{name_base}_margin_summary.txt"
                    else:
                        # At root level
                        if timestamp:
                            output_name = f"margin_summary_{timestamp}.txt"
                        else:
                            stem = _remove_archive_ext(_basename(full))
                            output_name = f"{stem}.txt"

                    datasets.append((output_name, files, full))

            # Check plain files directly in this directory
            plain = collect_plain_files(dirpath)
            if plain:
                # For plain files, try to extract timestamp from directory name
                rel_path = os.path.relpath(dirpath, input_abs)
                components = [c for c in rel_path.split(os.sep) if c and c != '.']

                # Try to get timestamp from directory name
                dir_name = os.path.basename(dirpath)
                timestamp = _parse_timestamp_from_name(dir_name)

                # If timestamp was found in the directory name itself, exclude that directory from components
                if timestamp and components and _parse_timestamp_from_name(components[-1]):
                    components = components[:-1]

                # Build name
                if components:
                    name_base = "_".join(components)
                    if timestamp:
                        output_name = f"{name_base}_margin_summary_{timestamp}.txt"
                    else:
                        output_name = f"{name_base}_margin_summary.txt"
                else:
                    if timestamp:
                        output_name = f"margin_summary_{timestamp}.txt"
                    else:
                        output_name = "margin_summary_plain.txt"

                datasets.append((output_name, plain, None))

        if not datasets:
            print("No margin-results datasets found.", file=sys.stderr)
            sys.exit(2)

        # Process datasets (handle duplicates with suffixes)
        for output_name, files, archive_path in datasets:
            # Handle duplicate names
            base_name = output_name
            suf = _unique_suffix(suffix_counter, base_name)
            if suf:
                # Insert suffix before .txt extension
                output_name = output_name.replace('.txt', f'{suf}.txt')

            text = summarize(files, pass_count_required, pass_err_cnt,
                           test_mode=test_mode, archive_path=archive_path, limits=limits)
            outpath = os.path.join(outdir, output_name)

            if inspect:
                print(f"[PREVIEW] Would write {outpath}")
            else:
                with open(outpath, "w", encoding="utf-8") as fh:
                    fh.write(text + "\n")
                print(f"Wrote {outpath}")


# ---------- CLI ----------

def main(argv: List[str]) -> None:
    parser = ArgumentParser(
        description=("Summarize lane margining results from files, directories, or archives. "
                     "Use --scan-root with either a top-level dir containing board dirs, "
                     "or a single board dir (e.g., BRM13250002)."))
    parser.add_argument(
        "-c", "--pass-count-required", type=int, default=0,
        help="Treat a row as PASS only if Pass==1 and Count==THIS (default: 0).",
    )
    parser.add_argument(
        "--pass-err-cnt", type=lambda x: None if x.lower() == 'none' else int(x), default=12,
        help="Maximum error count to consider a point as PASS. Points with count <= this value are PASS. Default is 12 (BER9 @ 99.99%% Confidence). Use 'None' to use the PASS column from the data file instead.",
    )
    parser.add_argument("--scan-root", default=None,
                        help="Scan a top directory containing board directories, "
                             "or a single board directory (e.g., BRM13250010).")
    parser.add_argument("--deep-scan", action="store_true",
                        help="Deep scan mode: recursively traverse input directory, "
                             "name outputs using full path components + timestamp (e.g., BOARD_P6_HH-MM-SS.txt). "
                             "Mutually exclusive with --scan-root.")
    parser.add_argument("-i", "--inspect", action="store_true",
                        help="Preview mode: show what would be written without creating files. "
                             "Warnings and errors are still displayed.")
    parser.add_argument("--outdir", default=".",
                        help="Directory to write summary files (used in all modes).")
    parser.add_argument("--out", default=None,
                        help="Write a single summary to this file path (only if exactly one dataset).")
    parser.add_argument("--test", action="store_true",
                        help="Enable test mode: add Bus/Device/Function/Lane columns to output for traceability.")
    parser.add_argument("--limits", nargs=2, type=float, metavar=('WIDTH_UI', 'HEIGHT_V'),
                        help="Add PASS column based on limits: WIDTH in %%UI (e.g., 28.0), HEIGHT in V (e.g., 0.048). "
                             "Lane passes if Time margin >= WIDTH and Voltage margin >= HEIGHT.")
    parser.add_argument("inputs", nargs="*",
                        help="Direct mode: input files/dirs/archives to summarize. "
                             "If omitted, you must use --scan-root.")
    ns = parser.parse_args(argv)

    # Parse limits into tuple if provided
    limits = tuple(ns.limits) if ns.limits else None

    # Validate mutually exclusive modes
    if ns.scan_root and ns.deep_scan:
        parser.error("--scan-root and --deep-scan are mutually exclusive")

    # Deep scan mode
    if ns.deep_scan:
        if not ns.inputs:
            parser.error("--deep-scan requires at least one input directory")
        if len(ns.inputs) != 1:
            parser.error("--deep-scan requires exactly one input directory")
        if ns.out:
            print("WARNING: --out is ignored in --deep-scan mode", file=sys.stderr)
        deep_scan_mode(ns.inputs[0], ns.outdir, ns.pass_count_required, ns.pass_err_cnt,
                      test_mode=ns.test, limits=limits, inspect=ns.inspect)
        return

    # Scan root mode
    if ns.scan_root:
        if ns.inputs:
            print("Ignore positional inputs when using --scan-root.", file=sys.stderr)
        if ns.inspect:
            print("WARNING: --inspect not supported in --scan-root mode (use --deep-scan instead)", file=sys.stderr)
        scan_and_write(ns.scan_root, ns.outdir, ns.pass_count_required, ns.pass_err_cnt, test_mode=ns.test, limits=limits)
        return

    # Direct mode (default)
    if not ns.inputs:
        parser.error("either provide positional inputs or use --scan-root or --deep-scan")

    if ns.inspect:
        print("WARNING: --inspect not fully supported in direct mode (use --deep-scan instead)", file=sys.stderr)

    direct_mode(ns.inputs, ns.outdir, ns.out, ns.pass_count_required, ns.pass_err_cnt, test_mode=ns.test, limits=limits)


if __name__ == "__main__":
    main(sys.argv[1:])

