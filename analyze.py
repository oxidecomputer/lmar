#!/usr/bin/env python3
"""lmar.py
Analyize lane margining results
Copyright 2022 Oxide Computer Company
"""

from tabulate import tabulate
from argparse import ArgumentParser
import os.path
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

# This is a temporary aid for annotating the output on Cosmo until
# we have topology data and can build this dynamically. For reference, here is
# the current PCI hierarchy on a representative Cosmo:
#
# [0/1/3] - Turin GPP Bridge
#     [2/0/0] - 7450 PRO NVMe SSD (nvme0)
# [0/7/1] - Turin Internal PCIe GPP Bridge to Bus [D:C]
#     [1/0/0] - Turin PCIe Dummy Function (--)
# [20/1/3] - Turin GPP Bridge
#     [21/0/0] - FireCuda 540 SSD (nvme1)
# [40/1/1] - Turin GPP Bridge
#     [43/0/0] - Unknown device: 0x5302 (nvme20)
# [40/1/2] - Turin GPP Bridge
#     [44/0/0] - 9550 PRO NVMe SSD (nvme17)
# [40/1/3] - Turin GPP Bridge
#     [45/0/0] - Unknown device: 0x2751 (nvme23)
# [40/7/1] - Turin Internal PCIe GPP Bridge to Bus [D:C]
#     [41/0/0] - Turin PCIe Dummy Function (--)
#     [41/0/2] - Unknown device: 0x14c0 (--)
#     [41/0/3] - Secondary vNTB (--)
#     [41/0/4] - Turin USB 3.1 xHCI (--)
#     [41/0/5] - Turin CCP/ASP (--)
#     [41/0/6] - Unknown device: 0x14cb (--)
#     [41/0/7] - Unknown device: 0x14cc (--)
# [40/7/2] - Turin Internal PCIe GPP Bridge to Bus [D:C]
#     [42/0/0] - FCH SATA Controller [AHCI mode] (--)
#     [42/0/1] - FCH SATA Controller [AHCI mode] (--)
# [60/1/1] - Turin GPP Bridge
#     [61/0/0] - Unknown device: 0x5302 (nvme21)
# [60/1/2] - Turin GPP Bridge
#     [62/0/0] - NVMe DC SSD [Atomos Prime] (nvme18)
# [60/1/3] - Turin GPP Bridge
#     [63/0/0] - NVMe DC SSD [Atomos Prime] (nvme19)
# [a0/1/1] - Turin GPP Bridge
#     [a2/0/0] - NVMe SSD Controller CD8P (nvme12)
# [a0/1/2] - Turin GPP Bridge
#     [a3/0/0] - NVMe SSD Controller CD8P (nvme14)
# [a0/1/3] - Turin GPP Bridge
#     [a4/0/0] - Unknown device: 0x2751 (nvme13)
# [a0/1/4] - Turin GPP Bridge
#     [a5/0/0] - 9550 PRO NVMe SSD (nvme22)
# [a0/7/1] - Turin Internal PCIe GPP Bridge to Bus [D:C]
#     [a1/0/0] - Turin PCIe Dummy Function (--)
#     [a1/0/2] - Unknown device: 0x14c0 (--)
#     [a1/0/3] - Secondary vNTB (--)
#     [a1/0/4] - Turin USB 3.1 xHCI (--)
#     [a1/0/5] - Turin CCP/ASP (--)
#     [a1/0/6] - Unknown device: 0x14cb (--)
#     [a1/0/7] - Unknown device: 0x14cc (--)
# [c0/1/1] - Turin GPP Bridge
#     [c1/0/0] - T62100-KR Unified Wire Ethernet Controller (--)
#     [c1/0/1] - T62100-KR Unified Wire Ethernet Controller (--)
#     [c1/0/2] - T62100-KR Unified Wire Ethernet Controller (--)
#     [c1/0/3] - T62100-KR Unified Wire Ethernet Controller (--)
#     [c1/0/4] - T62100-KR Unified Wire Ethernet Controller (t4nex0)
#     [c1/0/5] - T62100-KR Unified Wire Storage Controller (--)
#     [c1/0/6] - T62100-KR Unified Wire Storage Controller (--)
# [e0/1/2] - Turin GPP Bridge
#     [e3/0/0] - I210 Gigabit Network Connection (igb0)
# [e0/7/1] - Turin Internal PCIe GPP Bridge to Bus [D:C]
#     [e1/0/0] - Turin PCIe Dummy Function (--)
# [e0/7/2] - Turin Internal PCIe GPP Bridge to Bus [D:C]
#     [e2/0/0] - FCH SATA Controller [AHCI mode] (--)
#
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

def compute_margin(results, key) -> float:
    """
    Robust margin finder:
    - Prefers the contiguous PASS run centered around 0 on the independent axis.
    - If the center sample is FAIL, uses the PASS run nearest to center.
    - Ignores stray PASS islands at the window edges.
    """
    passed = np.asarray(results["passed"], dtype=bool)
    independent_axis = np.asarray(results[key])
    n_points = len(passed)

    if n_points == 0:
        return np.nan
    n_passes = int(np.sum(passed))
    if n_passes == 0:
        return 0.0
    if n_passes == n_points:
        return float(np.ptp(independent_axis))

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
    for start_idx, end_idx in pass_runs:
        if start_idx <= center_index <= end_idx:
            return float(independent_axis[end_idx] - independent_axis[start_idx])

    # Otherwise, choose the run nearest to the center by index distance.
    def distance_to_center(run):
        start_idx, end_idx = run
        if center_index < start_idx:
            return start_idx - center_index
        if center_index > end_idx:
            return center_index - end_idx
        return 0

    start_idx, end_idx = min(pass_runs, key=distance_to_center)
    return float(independent_axis[end_idx] - independent_axis[start_idx])


def load_results(file: str) -> dict[dict]:
    """Load the lane margining results from a file.

    Returns a dict with the results for left and right separately.
    """
    with open(file, "rb") as f:
        vendor_id = int(f.readline().rstrip().rsplit()[-1], base=16)
        device_id = int(f.readline().rstrip().rsplit()[-1], base=16)
        lane = int(f.readline().rstrip().rsplit()[-1])

        bus = dev = func = 0
        match = re.search(r"b([0-9a-f]+)-d([0-9a-f]+)-f([0-9a-f]+)", file)
        if match:
            bus, dev, func = (int(x, 16) for x in match.groups())
        descr = COSMO_MAP.get((bus, dev, func), "?")

    results = np.loadtxt(file, skiprows=4, delimiter="\t")
    (time, voltage, duration, count, passed) = range(5)
    is_time = results[:, time] != 0
    is_voltage = np.logical_not(is_time)
    time_results = dict(
        time=results[is_time, time],
        count=results[is_time, count].astype(np.int64),
        duration=results[is_time, duration],
        passed=results[is_time, passed].astype(bool),
        xlabel="Time (% UI)",
    )
    voltage_results = dict(
        voltage=results[is_voltage, voltage],
        count=results[is_voltage, count].astype(np.int64),
        duration=results[is_voltage, duration],
        passed=results[is_voltage, passed].astype(bool),
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


def plot_results(ax, key, results):
    """Plot the results on an axis"""
    passed = results["passed"]
    independent_axis_passed = results[key][passed]
    margin = compute_margin(results, key)
    failed = np.logical_not(passed)
    ax.plot(
        independent_axis_passed,
        results["count"][passed],
        marker="o",
        color="g",
        mec="w",
        mew=0.5,
        linestyle="none",
    )
    ax.plot(
        results[key][failed],
        results["count"][failed],
        marker="x",
        color="r",
        linestyle="none",
    )
    ax.set_xlabel(results["xlabel"])
    ax.set_title(f"Margin: {margin:.03}", fontsize=14)
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_fontsize(12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def format_plot(results, fig, axes):
    axes[0].set_ylabel("Count")
    leg = axes[0].legend(
        ("Success", "Fail"), title="Margin result", fontsize=10, title_fontsize=12
    )
    leg.set_draggable(True)
    fig.tight_layout()
    fig.canvas.manager.set_window_title(
            "Vendor: {:x}, Device: {:x}, Lane {:d} -- {:}".format(
            results["vendor_id"],
            results["device_id"],
            results["lane"],
            results["descr"],
        )
    )


def summarize(namespace):
    files = namespace.files
    # NEW: configurable PASS gating by Count (default 0)
    pass_count_required = getattr(namespace, "pass_count_required", 0)

    table = dict(vendor_id=[], device_id=[],
                 lane=[], time_margin=[], voltage_margin=[],
                 descr=[])
    for file in files:
        results = load_results(file)

        # Minimal-change gating: clone the per-sweep dicts and AND 'passed' with (count == required)
        time_gated = dict(results["time"])
        time_gated["passed"] = np.logical_and(results["time"]["passed"],
                                              results["time"]["count"] == pass_count_required)

        voltage_gated = dict(results["voltage"])
        voltage_gated["passed"] = np.logical_and(results["voltage"]["passed"],
                                                 results["voltage"]["count"] == pass_count_required)

        table["vendor_id"].append("0x{:04x}".format(results["vendor_id"]))
        table["device_id"].append("0x{:04x}".format(results["device_id"]))
        table["descr"].append(results["descr"])
        table["lane"].append(results["lane"])
        # Pass gated dicts into the unchanged margin function
        table["time_margin"].append(compute_margin(time_gated, "time"))
        table["voltage_margin"].append(compute_margin(voltage_gated, "voltage"))
    headers = (
        "Vendor ID",
        "Device ID",
        "Lane",
        "Time margin (% UI)",
        "Voltage margin (V)",
        "Description",
    )
    print(tabulate(table, headers=headers))


def plot(namespace):
    files = namespace.files
    save = namespace.save

    figs = []
    for file in files:
        results = load_results(file)
        fig, axes = plt.subplots(1, 2, sharey="row", figsize=(8, 3))

        for ax, key in zip(axes, ("time", "voltage")):
            plot_results(ax, key, results[key])
        format_plot(results, fig, axes)

        if save:
            root, ext = os.path.splitext(file)
            savefile = f"{root}.pdf"
            fig.savefig(savefile)
            plt.close(fig)
        else:
            figs.append(fig)
    if not save:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plot_parser = subparsers.add_parser("plot", help="Plot margining data")
    plot_parser.add_argument(
        "-s",
        "--save",
        help="Save each plot, rather than displaying",
        action="store_true",
    )
    plot_parser.add_argument("files", help="Input data file(s)", nargs="+")
    plot_parser.set_defaults(func=plot)

    summarize_parser = subparsers.add_parser("summarize", help="Print a summary table")
    # NEW: allow gating PASS by Count in summarize (default 0)
    summarize_parser.add_argument(
        "-c", "--pass-count-required", type=int, default=0,
        help="Treat a row as PASS only if Pass==1 and Count==THIS (default: 0).",
    )
    summarize_parser.add_argument("files", help="Input data file(s)", nargs="+")
    summarize_parser.set_defaults(func=summarize)

    namespace = parser.parse_args(sys.argv[1:])
    namespace.func(namespace)

