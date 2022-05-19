# lmar

Prototype PCIe Lane Margining at the Receiver

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

## Plotting

Once the data has been collected, the small Python tool `plot-lmar.py` can be
pointed at the results file to plot them. It'll show the time and voltage
margining on two separate plots, with the number of errors at each position.
Note that one of the plots may be empty if that flavor of margining isn't
supported by the endpoint.
