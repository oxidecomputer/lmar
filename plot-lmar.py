#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys


def load_results(file: str) -> dict[dict]:
    """Load the lane margining results from a file.

    Returns a dict with the results for left and right separately.
    """
    results = np.loadtxt(file, skiprows=1, delimiter="\t")
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
    return dict(time=time_results, voltage=voltage_results)


def plot_results(ax, key, results):
    """Plot the results on an axis"""
    passed = results["passed"]
    failed = np.logical_not(passed)
    ax.plot(
        results[key][passed],
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
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_fontsize(12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def format_plot(fig, axes):
    axes[0].set_ylabel("Count")
    leg = axes[0].legend(
        ("Success", "Fail"), title="Margin result", fontsize=10, title_fontsize=12
    )
    leg.set_draggable(True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./plot-lmar.py <FILE>")
        sys.exit(1)

    results = load_results(sys.argv[1])

    plt.ioff()
    fig, axes = plt.subplots(1, 2, sharey="row", figsize=(8, 3))

    for ax, (key, res) in zip(axes, results.items()):
        plot_results(ax, key, res)
    format_plot(fig, axes)
