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

For probing all PCIe devices
```bash
$ pfexec ./target/debug/lmar -p
```

The other options to the program control the details of the margining process,
and can be seen with `cargo run -- --help`.

## Recommended options

As of version 0.3.13 there are three options that are highly recommended for
most runs, particularly on SP5-socket systems:

```bash
$ pfexec ./target/debug/lmar -p --fast-sweep --serial-lanes --force-parallel
```

- `--fast-sweep`

  Implements a middle-out binary(ish) edge search. This greatly speeds up the
  measurement, at the cost of not sampling every inner eye point along the axis.

- `--serial-lanes`

  Margins each lane sequentially rather than in parallel. Endpoints (U.2 SSDs)
  report that they support margining all lanes at the same time, but some drives
  were found to have significant noise and poor repeatability during a parallel
  measurement. After testing a number of methods, margining each lane
  sequentially was found to be the cleanest and most reliable.

- `--force-parallel`

  This concerns Root Complex ports (CPU side). AMD does not advertise the
  capability for parallel margining of its ports. The latest version is updated
  to honor that advertised capability, so the default `-p` scan now margins RC
  ports sequentially. However, parallel and sequential data were observed to be
  identical and have very tight distribution over multiple runs. Therefore, to
  speed up measurement time, this arg will force RC ports to margin in parallel.

## Command-line options

Positional arguments (single-target mode; omit both when using `-p`):

| Argument | Description |
| --- | --- |
| `<BDF>` | Bus/device/function of the PCIe endpoint to target, in hexadecimal (e.g. `1/0/0`). |
| `<PORT>` | Which port to use on that endpoint: `upstream` or `downstream`. Root complex ports are `downstream`. |

Options:

| Option | Default | Description |
| --- | --- | --- |
| `-d`, `--duration <SECONDS>` | `1.0` | Time to spend margining each point, in seconds. |
| `-l`, `--lanes <LANES>` | `0` | Lane(s) to margin. A comma- or dash-separated list of lane numbers, or `all` for every lane on the device. Margined in parallel unless `--serial-lanes` is given. |
| `-e`, `--error-count <COUNT>` | `22` | Maximum acceptable error count at a point. More than this is a failure at that margining point. |
| `-v`, `--verbose` | | Print verbose information about the device and margining process. May be repeated for more detail. |
| `-r`, `--report-only` | | Only report the margining capabilities of the device; do not actually margin. |
| `-p`, `--probe` | | Probe for and margin all PCIe devices on the system. |
| `-s`, `--seq-scan` | | Scan ports sequentially when probing (`-p`). Scanning is parallel by default. |
| `-4`, `--four-point` | | Run a 4-point test rather than a full sweep. |
| `-F`, `--force-parallel` | | Margin ports in parallel when probing (`-p`), even if they do not advertise an independent error sampler. See above. |
| `--serial-lanes` | | Margin lanes one at a time instead of in parallel. See above. |
| `--retry-point` | | Retry a margin point once on timeout or error. |
| `--fast-sweep` | | Use a fast edge search (exponential bracket + binary search) instead of measuring every point. See above. |
| `-z`, `--zip` | | Create a zip of the output directory. Single-target mode only; probe runs are always zipped. |
| `--timing` / `--no-timing` | `--timing` | Run or skip timing margining. |
| `--voltage` / `--no-voltage` | `--voltage` | Run or skip voltage margining, if supported. |
| `--bridges` / `--no-bridges` | `--bridges` | Margin or skip bridges when probing (`-p`). |
| `--children` / `--no-children` | `--children` | Margin or skip child devices when probing (`-p`). |
| `-h`, `--help` | | Print help. |
| `-V`, `--version` | | Print version. |

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

## Additional scripts

Only the common options are shown below; each script accepts `--help` for the
full list.

### `run_margin.py`

Drives an end-to-end run: uploads `lmar` to a remote host over SSH, runs it
there, fetches the archives back, then invokes `margin_summary.py` and
`collect_summaries.py` on the results. It standardizes the run layout under
`<outdir>/<board-sn>/{raw,summaries,stats}/<session>/`, so downstream consumers
always see the same structure. Arguments after `--` are passed through to `lmar`.

```
usage: run_margin.py --remote [user@]HOST [--repo-dir DIR] [--outdir DIR]
                     [--board-sn SN] [--session LABEL] [--cargo-build] [--sudo]
                     [--stats-scope {board,run}] [--skip-remote]
                     [-- LMAR_ARGS ...]
```

**Examples**
```bash
# probe all supported ports on a remote board
run_margin.py --remote root@mb-0 --repo-dir /staff/tom/git/lmar --outdir . -- -p

# single-target BDF, all lanes
run_margin.py --remote root@mb-0 --repo-dir /staff/tom/git/lmar --outdir . \
  --stats-scope run -- 60/1/3 downstream -l all

# re-summarize an existing local run without touching a remote host
run_margin.py --skip-remote --board-sn BRM13250013 \
  --repo-dir /staff/tom/git/lmar --outdir .
```

### `margin_summary.py`

Parses one or more `margin-results-b<bus>-d<dev>-f<func>-l<lane>` files and emits
compact per-lane summaries (time/voltage margins, basic pass/fail stats). Summaries
are written into a `summaries/` subdirectory alongside the run artifacts to keep
consumers consistent. Accepts files, directories, or archives directly, or a tree
of board directories via `--scan-root`.

```
usage: margin_summary.py [-i] [--scan-root DIR | --deep-scan] [--outdir DIR]
                         [--out FILE] [--limits WIDTH_UI HEIGHT_V]
                         [--pass-err-cnt N] [--test] [inputs ...]
```

**Examples**
```bash
# summarize a single run directory
margin_summary.py margin-2025-09-10T12-34-56/

# summarize specific files
margin_summary.py margin-*/margin-results-b*-l*

# scan a board directory (or a parent of several), adding a PASS column
margin_summary.py --scan-root BRM13250013/ --limits 28.0 0.048
```

### `collect_summaries.py`

Walks one or more run directories, discovers `summaries/` outputs, and consolidates
them into a single table (stdout or optional CSV). Useful for aggregating results
across multiple runs/zips.

```
usage: collect_summaries.py [--scan-root DIR] [--combine-all] [--only-combined]
                            [--outdir DIR] [--out FILE] [--csv-out FILE]
                            [--pass-width W] [--pass-height H] [--names]
                            [--omit-ports PORT ...] [inputs ...]
```

**Examples**
```bash
# aggregate summaries across several runs
collect_summaries.py margin-2025-09-10T12-34-56/ margin-2025-09-10T14-02-11/

# write a CSV
collect_summaries.py margin-* --csv-out all-summaries.csv
```
