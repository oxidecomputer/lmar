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
        (0x60, 0x1, 0x2): "N8(I) bridge",
            (0x62, 0x0, 0x0): "N8(I)",
        (0x60, 0x1, 0x3): "N8(I) bridge",
            (0x63, 0x0, 0x0): "N8(I)",
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
    passed = results["passed"]
    independent_axis = results[key]

    # First, handle the cases when there are no passes or no failures
    n_passes = np.sum(passed)
    if n_passes == 0:
        return 0.0
    if n_passes == len(passed):
        return np.ptp(independent_axis)

    """
    np.diff compares n to n+1. If they differ, n is marked as true.
    Assuming a `passed` of something like:
    [False, False, True, True, False]
    np.diff would then yield:
    [False, True, False, True, False]
    np.where of that then would be
    [(1,3),]
    Note that our actual eye is [2,3], but those calculations result
    in it being [1,3] which is too wide! Thus, we increment the first
    index to where our eye actually is.
    """
    passed_indices = np.where(np.diff(passed))[0]
    passed_indices[0] = passed_indices[0] + 1

    # Try to compute the last - first point at which the margining
    # passed. If there are exactly two such points, then we're done.
    if len(passed_indices) == 2:
        lower_bound = passed_indices[0]
        upper_bound = passed_indices[1]
    elif len(passed_indices) > 2:
        # If there are more than 2 such indices, then take the longest
        # run, i.e., the ones with the largest difference
        run_length = np.diff(passed_indices)
        max_run = np.argmax(run_length)
        lower_bound = passed_indices[max_run]
        upper_bound = passed_indices[max_run + 1]
    else:
        # Not sure, hunk a nan in there
        return np.nan

    return independent_axis[upper_bound] - independent_axis[lower_bound];


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
        passed=results[is_time, passed].astype(np.bool),
        xlabel="Time (% UI)",
    )
    voltage_results = dict(
        voltage=results[is_voltage, voltage],
        count=results[is_voltage, count].astype(np.int64),
        duration=results[is_voltage, duration],
        passed=results[is_voltage, passed].astype(np.bool),
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
    table = dict(vendor_id=[], device_id=[],
                 lane=[], time_margin=[], voltage_margin=[],
                 descr=[])
    for file in files:
        results = load_results(file)
        table["vendor_id"].append("0x{:04x}".format(results["vendor_id"]))
        table["device_id"].append("0x{:04x}".format(results["device_id"]))
        table["descr"].append(results["descr"])
        table["lane"].append(results["lane"])
        table["time_margin"].append(compute_margin(results["time"], "time"))
        table["voltage_margin"].append(compute_margin(results["voltage"], "voltage"))
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
    summarize_parser.add_argument("files", help="Input data file(s)", nargs="+")
    summarize_parser.set_defaults(func=summarize)

    namespace = parser.parse_args(sys.argv[1:])
    namespace.func(namespace)
