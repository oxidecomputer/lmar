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
import sys


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
        "Vendor: {:x}, Device: {:x}, Lane {:d}".format(
            results["vendor_id"],
            results["device_id"],
            results["lane"],
        )
    )


def summarize(namespace):
    files = namespace.files
    table = dict(vendor_id=[], device_id=[], lane=[], time_margin=[], voltage_margin=[])
    for file in files:
        results = load_results(file)
        table["vendor_id"].append("0x{:04x}".format(results["vendor_id"]))
        table["device_id"].append("0x{:04x}".format(results["device_id"]))
        table["lane"].append(results["lane"])
        table["time_margin"].append(compute_margin(results["time"], "time"))
        table["voltage_margin"].append(compute_margin(results["voltage"], "voltage"))
    headers = (
        "Vendor ID",
        "Device ID",
        "Lane",
        "Time margin (% UI)",
        "Voltage margin (V)",
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
    subparsers = parser.add_subparsers()
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
