# lmar

PCIe Lane Margining at the Receiver for illumos systems.

## Overview

This tool runs the Lane Margining at the Receiver protocol for PCIe devices at
Gen 4 or later. This can be used to assess signal integrity issues, which is
critical for high-speed data links like PCIe.

For such systems, one can imagine the receivers as sampling the analog waveform
corresponding to the bits the transmitter sent. The goal is to recover that bit
pattern exactly from the analog signal. A huge amount of signal processing
circuitry is involved in effectively recovering the bits, so much so that common
bit error rates are 10e-12 or less.

Though it might be implemented in any number of ways, we can model this process
as the receiver _sampling_ the incoming waveform at point in both time, relative
to the recovered clock, and in voltage, relative either to an average or a peak
amplitude. The _margin_ refers to the distance in either direction that this
sampling point can be moved, while maintaining an acceptable error rate.

The `lmar` tool can be used to determine these margins. It instructs the device
to step the sampler in either time or voltage, and then count the number of
errors it sees. The results are saved in a file, and can be analyzed with the
accompanying `analyze.py` Python tool.

> Important: This tool only runs on illumos systems.

## Usage

```bash
$ cargo build
$ pfexec ./target/debug/lmar 1/0/0 upstream
```

The first argument is the bus/device/function for the PCIe endpoint to be
targeted. The second is whether to target the upstream or downstream port of
that endpoint. For a root complex, that should be `downstream`. For a drive or
other similar device, it should be `upstream`.

The BDF can be retrieved from the output of `/usr/lib/pci/pcieadm show-devs`, in
the first column.

The other options to the program control the details of the margining process,
and can be seen with `cargo run -- --help`.

## Analysis

Once the data has been collected, the small Python tool `analyze.py` can be
pointed at the results file to analyze the data. The `summarize` subcommand will
print a tabular summary of the time and possibly voltage margin for each
reported device / lane. The `plot` subcommand will generate a plot window for
each reported device / lane, showing the results of each margined point. Any
number of files may be provided.

The tool requires a few packages:

- `matplotlib`
- `numpy`
- `tabulate`
