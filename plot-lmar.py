#!/usr/bin/env python3

from argparse import ArgumentParser
import os.path
import numpy as np
import matplotlib.pyplot as plt
import sys


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
        count=results[is_time, count].astype(np.int),
        duration=results[is_time, duration],
        passed=results[is_time, passed].astype(np.bool),
        xlabel="Time (% UI)",
    )
    voltage_results = dict(
        voltage=results[is_voltage, voltage],
        count=results[is_voltage, count].astype(np.int),
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
    # TODO-correctness: Peak-to-peak isn't quite right.
    # We probably want the longest uninterrupted / contiguous run
    # of passes.
    margin = independent_axis_passed.ptp()
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
    fig.canvas.set_window_title(
        "Vendor: {:x}, Device: {:x}, Lane {:d}".format(
            results["vendor_id"],
            results["device_id"],
            results["lane"],
        )
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--save",
        help="Save each plot, rather than displaying",
        action="store_true",
    )
    parser.add_argument("files", help="Input data file(s)", nargs="+")
    namespace = parser.parse_args(sys.argv[1:])

    figs = []
    for file in namespace.files:
        results = load_results(file)
        fig, axes = plt.subplots(1, 2, sharey="row", figsize=(8, 3))

        for ax, key in zip(axes, ("time", "voltage")):
            plot_results(ax, key, results[key])
        format_plot(results, fig, axes)

        if namespace.save:
            root, ext = os.path.splitext(file)
            savefile = f"{root}.pdf"
            fig.savefig(savefile)
            plt.close(fig)
        else:
            figs.append(fig)
    if not namespace.save:
        plt.ioff()
        plt.show()
