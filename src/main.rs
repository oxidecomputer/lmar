// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! PCIe lane margining at the receiver

use anyhow::{ensure, Context, Result};
use clap::ArgAction;
use clap::ArgEnum;
use clap::Error as ClapError;
use clap::ErrorKind;
use clap::Parser;
use indicatif::MultiProgress;
use indicatif::ProgressBar;
use indicatif::ProgressDrawTarget;
use indicatif::ProgressStyle;
use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io;
use std::io::{Read, Write};
use std::num::NonZeroU8;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::thread::sleep;
use std::time::Duration;
use std::time::Instant;
use thiserror::Error;
use walkdir::WalkDir;
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

const NVMEADM: &str = "/usr/sbin/nvmeadm";
const PCIEADM: &str = "/usr/lib/pci/pcieadm";

#[derive(Copy, Clone, Debug, ArgEnum)]
enum Port {
    Upstream,
    Downstream,
}

// Consts for verbosity levels
mod verbosity {

    // Print the device's link status.
    pub const LINK_STATUS: u64 = 1;

    // Print the device's capabilities and margin limits
    pub const CAPABILITIES: u64 = 1;

    // Print the margin duration
    pub const MARGIN_DURATION: u64 = 1;

    // Print the output file name for saving the results.
    pub const FILENAME: u64 = 1;

    // Print a summary of margining progress, clearing the line between each
    // point
    pub const PROGRESS_SUMMARY: u64 = 1;

    // Print all commands in Rust debug format
    pub const COMMANDS: u64 = 3;

    // Print all command register bit fields.
    pub const COMMAND_DETAIL: u64 = 4;
}

#[derive(Clone, Debug)]
enum LaneDescription {
    ByIndex(Vec<u8>),
    All,
}

impl LaneDescription {
    fn to_indices(
        &self,
        link_status: &LinkStatus,
    ) -> anyhow::Result<Vec<Lane>> {
        let to_lane = |l: u8| {
            ensure!(l < link_status.width.0, "Lane {} out of range", l);
            Lane::new(l).context("Invalid lane")
        };
        match self {
            LaneDescription::ByIndex(ref lanes) => {
                lanes.iter().map(|l| to_lane(*l)).collect()
            }
            LaneDescription::All => {
                (0..u8::from(link_status.width)).map(to_lane).collect()
            }
        }
    }
}

impl std::str::FromStr for LaneDescription {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "all" {
            return Ok(LaneDescription::All);
        }
        let mut indices = Vec::new();
        for part in s.split(',') {
            if let Some((lhs, rhs)) = part.split_once('-') {
                let start = lhs.parse::<u8>()?;
                let end = rhs.parse::<u8>()?;
                indices.extend(start..=end);
            } else {
                indices.push(part.parse()?);
            }
        }
        Ok(LaneDescription::ByIndex(indices))
    }
}

#[derive(Parser, Clone, Debug)]
#[clap(author, version, about, long_about = None)]
/// The `lmar` program is a prototype tool to run the PCIe Lane Margining at the
/// Receiver protocol on an attached PCIe endpoint.
///
/// Lane margining refers to the process of determining the maximum shift in
/// either the timing or voltage of a receiver's sampling machinery, at which an
/// acceptable bit-error rate may still be achieved. The protocol involves
/// "stepping" the receiver's sampling left or right (in time) or up and down
/// (in voltage), and determining the number of bit errors at those new sampling
/// parameters. This generates a "margin" or acceptable leeway in the sampling
/// parameters, and is a useful tool for diagnosing signal integrity problems.
struct Args {
    /// The bus/device/function to run the protocol against.
    ///
    /// Note that values are provided in hexadecimal.
    bdf: Option<Bdf>,

    /// Which port to use on the chosen PCIe endpoint.
    ///
    /// For example, for the root component this should always be downstream.
    /// For an endpoint like a drive, this should be upstream.
    #[clap(arg_enum)]
    port: Option<Port>,

    /// The time to spend margining each point, in seconds.
    #[clap(short, long, default_value_t = 1.0)]
    duration: f64,

    /// The lane(s) on the endpoint to be margined.
    ///
    /// This defaults to margining just the first lane (lane 0). It may be
    /// specified as a comma- or dash-separated list of values, to indicate the
    /// exact lane numbers to margin; or as the string "all" to margin all lanes
    /// on the device.
    ///
    /// Multiple lanes will be margined in parallel.
    #[clap(short, long, default_value = "0")]
    lanes: LaneDescription,

    /// The maximum acceptable error count at a point.
    ///
    /// If the number of errors is not greater than this value, margining is
    /// deemed to succeed. Any value greater than this is a failure at that
    /// margining point.
    #[clap(short, long)]
    error_count: Option<u8>,

    /// Print verbose information about the device and margining process.
    #[clap(short, long, parse(from_occurrences))]
    verbose: u64,

    /// Only report the margining capabilities of the device, but do not
    /// actually run the margining protocol.
    #[clap(short, long)]
    report_only: bool,

    /// Probe for PCIe devices
    #[clap(short, long)]
    probe: bool,

    /// Don't run Timing margining.
    #[clap(long = "no-timing", action = ArgAction::SetFalse)]
    timing: bool,
    /// Run timing margining [default].
    #[clap(long = "timing", overrides_with = "timing")]
    _no_timing: bool,

    /// Don't run Voltage margining.
    #[clap(long = "no-voltage", action = ArgAction::SetFalse)]
    voltage: bool,
    /// Run Voltage margining if supported [default].
    #[clap(long = "voltage", overrides_with = "voltage")]
    _no_voltage: bool,
}

/// Errors working with a PCIe device
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error("Invalid PCIe read encountered: offset {0:#x}, data: {1:#x}")]
    InvalidPcieRead(usize, u64),

    #[error("Invalid value for a PCIe configuration parameter '{parameter}': {value}")]
    InvalidPcieParameter { parameter: &'static str, value: u64 },

    #[error("Unsupported PCIe extended capability: {0:?}")]
    UnsupportedExtendedCap(ExtendedCapabilityId),

    #[error("Invalid bus/device/function string: \"{0}\"")]
    InvalidBdf(String),

    #[error("Invalid lane: {0}")]
    InvalidLane(u8),

    #[error("An error occurred during lane margining: {0}")]
    Margin(String),

    #[error(
        "Failed to decode response for command {command:?}, reason {reason}"
    )]
    DecodeFailed { command: MarginCommand, reason: String },

    #[error("Capability or feature not implemented yet: {0}")]
    Unimplemented(String),
}

// Ioctl command definitions
const PCITOOL_IOC: i32 =
    (('P' as i32) << 24) | (('C' as i32) << 16) | (('T' as i32) << 8);
const PCITOOL_DEVICE_GET_REG: i32 = PCITOOL_IOC | 1;
const PCITOOL_DEVICE_SET_REG: i32 = PCITOOL_IOC | 2;

// TODO-correctness: The register file should not matter so much, since x86 will
// ultimately look up the device just by the BDF. But we should still do the
// right thing eventually and map the requested BDF to the right illumos
// register devices file.
const PCI_REG_DEVICE_FILE: &str = "/devices/pci@0,0:reg";

#[cfg(target_endian = "little")]
const PCI_ATTR_ENDIANNESS: u32 = 0x0;
#[cfg(not(target_endian = "little"))]
const PCI_ATTR_ENDIANNESS: u32 = 0x100;

const PCIE_CONFIGURATION_SPACE_SIZE: usize = 4096;
const PCIE_VENDOR_ID_OFFSET: usize = 0x00;
const PCIE_DEVICE_ID_OFFSET: usize = 0x02;

// Offset of the PCIe Capabilities Pointer register, within the standard PCIe
// configuration header.
const PCIE_CAP_POINTER: usize = 0x34;

// Offset to the PCIe Extended Capabilities Header, in configuration space.
const PCIE_EXTENDED_CAPABILITY_HEADER_OFFSET: usize = 0x100;
const LANE_MARGINING_CAPABILITY_ID: u16 = 0x27;

/// A PCIe Capability ID
///
/// NOTE: We're only identifying the "PCIe" capability for now, since that
/// contains information about the extended capability registers, which has lane
/// margining information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CapabilityId {
    Pcie,
    Other(u8),
}

impl From<u8> for CapabilityId {
    fn from(x: u8) -> Self {
        match x {
            0x10 => CapabilityId::Pcie,
            _ => CapabilityId::Other(x),
        }
    }
}

/// A PCIe Extended Capability ID.
///
/// We're only interested in the lane margining cap, for now.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExtendedCapabilityId {
    None,
    LaneMargining,
    Other(u16),
}

impl From<u16> for ExtendedCapabilityId {
    fn from(x: u16) -> Self {
        match x {
            0x00 => ExtendedCapabilityId::None,
            LANE_MARGINING_CAPABILITY_ID => ExtendedCapabilityId::LaneMargining,
            _ => ExtendedCapabilityId::Other(x),
        }
    }
}

/// The `PciRegister` is the struct used to read and write a PCIe device's
/// registers, usually configuration space, using the OS's `ioctl(2)` interface.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct PciRegister {
    user_version: u16,
    driver_version: u16,
    bus_no: u8,
    dev_no: u8,
    func_no: u8,
    barnum: u8,
    offset: u64,
    acc_attr: u32,
    _padding1: u32,
    data: u64,
    status: u32,
    _padding2: u32,
    phys_addr: u64,
}

/// Represents a single PCIe Capability Header.
#[derive(Debug, Clone, Copy)]
pub struct PcieCapabilityHeader {
    id: CapabilityId,
    next: u8,
}

impl From<u16> for PcieCapabilityHeader {
    fn from(word: u16) -> Self {
        let id = CapabilityId::from((word & 0xFF) as u8);
        let next = (word >> 8) as u8;
        Self { id, next }
    }
}

/// Information about a single PCIe extended capability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtendedCapability {
    pub id: ExtendedCapabilityId,
    pub version: u8,
}

impl From<u32> for ExtendedCapability {
    fn from(x: u32) -> Self {
        let id = ExtendedCapabilityId::from((x & 0x0000_FFFF) as u16);
        let version = ((x & 0x000F_0000) >> 16) as u8;
        Self { id, version }
    }
}

/// Represents a single PCIe Extended Capability Header.
#[derive(Debug, Clone, Copy)]
pub struct PcieExtendedCapabilityHeader {
    capability: ExtendedCapability,
    next: u16,
}

impl From<u32> for PcieExtendedCapabilityHeader {
    fn from(x: u32) -> Self {
        let capability = ExtendedCapability::from(x);
        let next = ((x & 0xFFF0_0000) >> 20) as u16;
        Self { capability, next }
    }
}

/// The width of a PCIe link
#[derive(Debug, Clone, Copy)]
pub struct LinkWidth(u8);

impl From<LinkWidth> for u8 {
    fn from(w: LinkWidth) -> u8 {
        w.0
    }
}

impl fmt::Display for LinkWidth {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x{}", self.0)
    }
}

impl TryFrom<u8> for LinkWidth {
    type Error = Error;

    fn try_from(w: u8) -> Result<Self, Self::Error> {
        const MASK: u8 = 0x3F;
        match w & MASK {
            0b00_0001 => Ok(LinkWidth(1)),
            0b00_0010 => Ok(LinkWidth(2)),
            0b00_0100 => Ok(LinkWidth(4)),
            0b00_1000 => Ok(LinkWidth(8)),
            0b00_1100 => Ok(LinkWidth(12)),
            0b01_0000 => Ok(LinkWidth(16)),
            0b10_0000 => Ok(LinkWidth(32)),
            _ => Err(Error::InvalidPcieParameter {
                parameter: "link-width",
                value: w.into(),
            }),
        }
    }
}

/// Speed of a PCIe link, in GT/s.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct LinkSpeed(f32);

impl TryFrom<u8> for LinkSpeed {
    type Error = Error;

    fn try_from(word: u8) -> Result<Self, Self::Error> {
        const MASK: u8 = 0x0F;
        match word & MASK {
            0b0001 => Ok(LinkSpeed(2.5)),
            0b0010 => Ok(LinkSpeed(5.0)),
            0b0011 => Ok(LinkSpeed(8.0)),
            0b0100 => Ok(LinkSpeed(16.0)),
            0b0101 => Ok(LinkSpeed(32.0)),
            _ => Err(Error::InvalidPcieParameter {
                parameter: "link-speed",
                value: word.into(),
            }),
        }
    }
}

impl fmt::Display for LinkSpeed {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            LinkSpeed(2.5) => "Gen1",
            LinkSpeed(5.0) => "Gen2",
            LinkSpeed(8.0) => "Gen3",
            LinkSpeed(16.0) => "Gen4",
            LinkSpeed(32.0) => "Gen5",
            _ => &format!("{:?}", self),
        };
        write!(f, "{}", s)
    }
}

/// The PCIe Link Status Register
#[derive(Debug, Clone, Copy)]
pub struct LinkStatus {
    pub speed: LinkSpeed,
    pub width: LinkWidth,
    pub training: bool,
    pub active: bool,
}

impl LinkStatus {
    pub fn valid_lane(&self, lane: Lane) -> bool {
        lane.0 < self.width.0
    }
}

impl LinkStatus {
    pub const REGISTER_OFFSET: usize = 0x12;
}

impl TryFrom<u16> for LinkStatus {
    type Error = Error;

    fn try_from(word: u16) -> Result<Self, Self::Error> {
        let speed = LinkSpeed::try_from(word as u8)?;
        let width = LinkWidth::try_from((word >> 4) as u8)?;
        let training = (word & (1 << 11)) != 0;
        let active = (word & (1 << 13)) != 0;
        Ok(Self { speed, width, training, active })
    }
}

impl fmt::Display for LinkStatus {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{}", self.speed, self.width)
    }
}

#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub struct Lane(u8);

impl Lane {
    pub fn new(lane: u8) -> Result<Self, Error> {
        if lane < 32 {
            Ok(Self(lane))
        } else {
            Err(Error::InvalidLane(lane))
        }
    }
}

impl From<Lane> for u8 {
    fn from(lane: Lane) -> u8 {
        lane.0
    }
}

impl fmt::Display for Lane {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The PCIe Lane Margining at the Receiver Extended Capability Header
#[derive(Debug, Clone)]
pub struct LaneMarginingCapabilityHeader {
    _port_capabilities: MarginingPortCapabilities,
    port_status: MarginingPortStatus,
    lanes: Vec<MarginingLane>,
}

impl LaneMarginingCapabilityHeader {
    pub fn lane_control_offset(&self, lane: Lane) -> Result<usize, Error> {
        let lane_u8 = u8::from(lane);
        let lane_us = usize::from(lane_u8);
        if lane_us < self.lanes.len() {
            Ok(lane_us * std::mem::size_of::<u16>() * 2 + 8)
        } else {
            Err(Error::InvalidLane(lane_u8))
        }
    }

    pub fn lane_status_offset(&self, lane: Lane) -> Result<usize, Error> {
        Ok(self.lane_control_offset(lane)? + std::mem::size_of::<u16>())
    }
}

impl TryFrom<&[u8]> for LaneMarginingCapabilityHeader {
    type Error = Error;

    fn try_from(mut buf: &[u8]) -> Result<Self, Self::Error> {
        use bytes::Buf;
        // Must have at least port cap / status (4B), and one lane of ctl /
        // status (another 4B).
        if buf.len() < 8 {
            return Err(Error::Margin(String::from(
                "Lane margining extended capability header too small",
            )));
        }
        let port_capabilities =
            MarginingPortCapabilities::from(buf.get_u16_le());
        let port_status = MarginingPortStatus::from(buf.get_u16_le());

        // Each lane has a control + status register
        let n_lanes = buf.len() / (2 * std::mem::size_of::<u16>());
        let mut lanes = Vec::with_capacity(n_lanes);
        for _ in 0..n_lanes {
            let control = MarginingLaneControl::from(buf.get_u16_le());
            let status = MarginingLaneStatus::decode_for_cmd(
                control.cmd,
                buf.get_u16_le(),
            )?;
            lanes.push(MarginingLane { control, status });
        }
        Ok(LaneMarginingCapabilityHeader {
            _port_capabilities: port_capabilities,
            port_status,
            lanes,
        })
    }
}

/// The Lane Margining Port Capabilities Register
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct MarginingPortCapabilities {
    uses_driver_software: bool,
}

impl From<u16> for MarginingPortCapabilities {
    fn from(word: u16) -> Self {
        Self { uses_driver_software: (word & 0x1) != 0 }
    }
}

/// The Lane Margining Port Status Register
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct MarginingPortStatus {
    ready: bool,
    software_ready: bool,
}

impl From<u16> for MarginingPortStatus {
    fn from(word: u16) -> Self {
        let ready = (word & 0b01) != 0;
        let software_ready = (word & 0b10) != 0;
        Self { ready, software_ready }
    }
}

/// A lane margining Margin Command
#[derive(Debug, Clone, Copy)]
pub enum MarginCommand {
    NoCommand,
    Report(ReportRequest),
    SetErrorCountLimit(ErrorCount),
    GoToNormalSettings,
    ClearErrorLog,
    StepLeftRight(StepLeftRight),
    StepUpDown(StepUpDown),
}

impl MarginCommand {
    pub fn margin_type(&self) -> u8 {
        match self {
            MarginCommand::NoCommand => 0b111,
            MarginCommand::Report(_) => 0b001,
            MarginCommand::SetErrorCountLimit(_)
            | MarginCommand::GoToNormalSettings
            | MarginCommand::ClearErrorLog => 0b010,
            MarginCommand::StepLeftRight(_) => 0b011,
            MarginCommand::StepUpDown(_) => 0b100,
        }
    }

    pub fn payload(&self) -> u8 {
        match self {
            MarginCommand::NoCommand => 0x9C,
            MarginCommand::Report(ReportRequest::Capabilities) => 0x88,
            MarginCommand::Report(ReportRequest::NumVoltageSteps) => 0x89,
            MarginCommand::Report(ReportRequest::NumTimingSteps) => 0x8A,
            MarginCommand::Report(ReportRequest::MaxTimingOffset) => 0x8B,
            MarginCommand::Report(ReportRequest::MaxVoltageOffset) => 0x8C,
            MarginCommand::Report(ReportRequest::SamplingRateVoltage) => 0x8D,
            MarginCommand::Report(ReportRequest::SamplingRateTiming) => 0x8E,
            MarginCommand::Report(ReportRequest::SampleCount) => 0x8F,
            MarginCommand::Report(ReportRequest::MaxLanes) => 0x90,
            MarginCommand::SetErrorCountLimit(count) => u8::from(*count),
            MarginCommand::GoToNormalSettings => 0x0F,
            MarginCommand::ClearErrorLog => 0x55,
            MarginCommand::StepLeftRight(step) => u8::from(*step),
            MarginCommand::StepUpDown(step) => u8::from(*step),
        }
    }

    /// Return true if the provided response is expected for the command in
    /// `self`.
    pub fn expects_response(&self, response: MarginResponse) -> bool {
        match (self, response) {
            (MarginCommand::NoCommand, MarginResponse::NoCommand) => true,
            (
                MarginCommand::Report(ReportRequest::Capabilities),
                MarginResponse::Report(ReportResponse::Capabilities(_)),
            ) => true,
            (
                MarginCommand::Report(ReportRequest::NumVoltageSteps),
                MarginResponse::Report(ReportResponse::NumVoltageSteps(_)),
            ) => true,
            (
                MarginCommand::Report(ReportRequest::NumTimingSteps),
                MarginResponse::Report(ReportResponse::NumTimingSteps(_)),
            ) => true,
            (
                MarginCommand::Report(ReportRequest::MaxTimingOffset),
                MarginResponse::Report(ReportResponse::MaxTimingOffset(_)),
            ) => true,
            (
                MarginCommand::Report(ReportRequest::MaxVoltageOffset),
                MarginResponse::Report(ReportResponse::MaxVoltageOffset(_)),
            ) => true,
            (
                MarginCommand::Report(ReportRequest::SamplingRateVoltage),
                MarginResponse::Report(ReportResponse::SamplingRateVoltage(_)),
            ) => true,
            (
                MarginCommand::Report(ReportRequest::SamplingRateTiming),
                MarginResponse::Report(ReportResponse::SamplingRateTiming(_)),
            ) => true,
            (
                MarginCommand::Report(ReportRequest::SampleCount),
                MarginResponse::Report(ReportResponse::SampleCount(_)),
            ) => true,
            (
                MarginCommand::Report(ReportRequest::MaxLanes),
                MarginResponse::Report(ReportResponse::MaxLanes(_)),
            ) => true,
            (
                MarginCommand::SetErrorCountLimit(_),
                MarginResponse::ErrorCountLimit(_),
            ) => true,
            (
                MarginCommand::GoToNormalSettings,
                MarginResponse::GoToNormalSettings,
            ) => true,
            (MarginCommand::ClearErrorLog, MarginResponse::ClearErrorLog) => {
                true
            }
            (
                MarginCommand::StepLeftRight(_),
                MarginResponse::StepExecutionStatus { .. },
            ) => true,
            (
                MarginCommand::StepUpDown(_),
                MarginResponse::StepExecutionStatus { .. },
            ) => true,
            (_, _) => false,
        }
    }
}

impl From<u16> for MarginCommand {
    fn from(word: u16) -> Self {
        let (cmd, payload) = {
            let octets = word.to_le_bytes();
            (octets[0], octets[1])
        };
        let receiver = cmd & 0b111;
        let margin_type = (cmd >> 3) & 0b111;
        match margin_type {
            0b111 => {
                assert_eq!(payload, 0x9C);
                assert_eq!(receiver, 0);
                MarginCommand::NoCommand
            }
            0b001 => MarginCommand::Report(ReportRequest::from(payload)),
            0b010 => match payload {
                0x0F => MarginCommand::GoToNormalSettings,
                0x55 => MarginCommand::ClearErrorLog,
                _ => {
                    MarginCommand::SetErrorCountLimit(ErrorCount::from(payload))
                }
            },
            0b011 => MarginCommand::StepLeftRight(StepLeftRight::from(payload)),
            0b100 => MarginCommand::StepUpDown(StepUpDown::from(payload)),
            _ => unreachable!(),
        }
    }
}

/// Data that can be reported via a margin command
#[derive(Debug, Clone, Copy)]
pub enum ReportRequest {
    Capabilities,
    NumVoltageSteps,
    NumTimingSteps,
    MaxTimingOffset,
    MaxVoltageOffset,
    SamplingRateVoltage,
    SamplingRateTiming,
    SampleCount,
    MaxLanes,
}

impl From<u8> for ReportRequest {
    fn from(word: u8) -> Self {
        use ReportRequest::*;
        match word {
            0x88 => Capabilities,
            0x89 => NumVoltageSteps,
            0x8A => NumTimingSteps,
            0x8B => MaxTimingOffset,
            0x8C => MaxVoltageOffset,
            0x8D => SamplingRateVoltage,
            0x8E => SamplingRateTiming,
            0x8F => SampleCount,
            0x90 => MaxLanes,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ReportCapabilities {
    independent_error_sampler: bool,
    sample_reporting_method: SampleReportingMethod,
    independent_left_right_sampling: bool,
    independent_up_down_voltage: bool,
    voltage_supported: bool,
}

impl From<u8> for ReportCapabilities {
    fn from(word: u8) -> ReportCapabilities {
        let independent_error_sampler = (word & 0b10000) != 0;
        let sample_reporting_method = if (word & 0b01000) != 0 {
            SampleReportingMethod::Rate
        } else {
            SampleReportingMethod::Count
        };
        let independent_left_right_sampling = (word & 0b00100) != 0;
        let independent_up_down_voltage = (word & 0b00010) != 0;
        let voltage_supported = (word & 0b00001) != 0;
        ReportCapabilities {
            independent_error_sampler,
            sample_reporting_method,
            independent_left_right_sampling,
            independent_up_down_voltage,
            voltage_supported,
        }
    }
}

impl From<ReportCapabilities> for u8 {
    fn from(r: ReportCapabilities) -> u8 {
        r.voltage_supported as u8
            | (r.independent_up_down_voltage as u8) << 1
            | (r.independent_left_right_sampling as u8) << 2
            | if matches!(
                r.sample_reporting_method,
                SampleReportingMethod::Rate
            ) {
                1 << 3
            } else {
                0
            }
            | (r.independent_error_sampler as u8) << 4
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SampleReportingMethod {
    Rate,
    Count,
}

#[derive(Debug, Clone, Copy)]
pub enum ReportResponse {
    Capabilities(ReportCapabilities),
    NumVoltageSteps(u8),
    NumTimingSteps(u8),
    MaxTimingOffset(u8),
    MaxVoltageOffset(u8),
    SamplingRateVoltage(u8),
    SamplingRateTiming(u8),
    SampleCount(u8),
    MaxLanes(u8),
}

#[derive(Debug, Clone, Copy)]
pub enum StepMarginExecutionStatus {
    Nak,
    InProgress,
    Setup,
    TooManyErrors,
}

impl From<u8> for StepMarginExecutionStatus {
    fn from(word: u8) -> Self {
        match (word >> 6) & 0b11 {
            0b11 => StepMarginExecutionStatus::Nak,
            0b10 => StepMarginExecutionStatus::InProgress,
            0b01 => StepMarginExecutionStatus::Setup,
            0b00 => StepMarginExecutionStatus::TooManyErrors,
            _ => unreachable!(),
        }
    }
}

impl From<StepMarginExecutionStatus> for u8 {
    fn from(s: StepMarginExecutionStatus) -> u8 {
        let bits = match s {
            StepMarginExecutionStatus::Nak => 0b11,
            StepMarginExecutionStatus::InProgress => 0b10,
            StepMarginExecutionStatus::Setup => 0b01,
            StepMarginExecutionStatus::TooManyErrors => 0b00,
        };
        bits << 6
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorCount(u8);

impl From<u8> for ErrorCount {
    fn from(word: u8) -> Self {
        Self(word & 0b11111)
    }
}

impl From<ErrorCount> for u8 {
    fn from(count: ErrorCount) -> u8 {
        count.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepLeftRight {
    direction: Option<LeftRight>,
    steps: Steps,
}

impl From<StepLeftRight> for u8 {
    fn from(step: StepLeftRight) -> u8 {
        let bits = match step.direction {
            None => 0b0,
            Some(dir) => u8::from(dir) << 6,
        };
        bits | u8::from(step.steps)
    }
}

impl From<u8> for StepLeftRight {
    fn from(word: u8) -> StepLeftRight {
        Self {
            direction: Some(LeftRight::from(word)),
            // Only bits 5:0 are valid
            steps: Steps::from(word & 0b0011_1111),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LeftRight {
    Left,
    Right,
}

impl From<LeftRight> for u8 {
    fn from(lr: LeftRight) -> u8 {
        match lr {
            LeftRight::Left => 0b1,
            LeftRight::Right => 0b0,
        }
    }
}

impl From<u8> for LeftRight {
    fn from(lr: u8) -> Self {
        match lr & 0b0100_0000 {
            0b0100_0000 => LeftRight::Left,
            0b0000_0000 => LeftRight::Right,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepUpDown {
    direction: Option<UpDown>,
    steps: Steps,
}

impl From<StepUpDown> for u8 {
    fn from(step: StepUpDown) -> u8 {
        let bits = match step.direction {
            None => 0b0,
            Some(dir) => u8::from(dir) << 7,
        };
        bits | u8::from(step.steps)
    }
}

impl From<u8> for StepUpDown {
    fn from(word: u8) -> StepUpDown {
        Self {
            direction: Some(UpDown::from(word)),
            // Only bits 6:0 are valid
            steps: Steps::from(word & 0b0111_1111),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum UpDown {
    Up,
    Down,
}

impl From<UpDown> for u8 {
    fn from(ud: UpDown) -> u8 {
        match ud {
            UpDown::Up => 0b0,
            UpDown::Down => 0b1,
        }
    }
}

impl From<u8> for UpDown {
    fn from(ud: u8) -> Self {
        match ud & 0b1000_0000 {
            0b1000_0000 => UpDown::Down,
            0b0000_0000 => UpDown::Up,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Steps(u8);

impl From<u8> for Steps {
    fn from(word: u8) -> Steps {
        Self(word & 0b0111_1111)
    }
}

impl From<Steps> for u8 {
    fn from(steps: Steps) -> u8 {
        steps.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Receiver(u8);

impl Receiver {
    pub fn new(rx: u8) -> Result<Self, Error> {
        if rx <= 0b110 {
            Ok(Self(rx))
        } else {
            Err(Error::Margin(format!("Invalid receiver: {}", rx)))
        }
    }

    pub fn broadcast() -> Self {
        Self(0)
    }

    pub fn downstream() -> Self {
        Self(0b001)
    }

    pub fn upstream() -> Self {
        Self(0b110)
    }
}

impl From<Receiver> for u8 {
    fn from(rx: Receiver) -> u8 {
        rx.0
    }
}

/// A lane margining Margin Response
#[derive(Debug, Clone, Copy)]
pub enum MarginResponse {
    Empty,
    NoCommand,
    Report(ReportResponse),
    ErrorCountLimit(ErrorCount),
    GoToNormalSettings,
    ClearErrorLog,
    StepExecutionStatus {
        status: StepMarginExecutionStatus,
        error_count: ErrorCount,
    },
}

impl MarginResponse {
    // Responses cannot be uniquely decoded from the u16 word alone, since the
    // interpretation of the payload depends on the payload of the corresponding
    // margin command word.
    pub fn decode_for_cmd(
        cmd: MarginCommand,
        word: u16,
    ) -> Result<Self, Error> {
        let (response_cmd, payload) = {
            let octets = word.to_le_bytes();
            (octets[0], octets[1])
        };
        let receiver = response_cmd & 0b111;
        let response_margin_type = (response_cmd >> 3) & 0b111;
        match response_margin_type {
            0b111 => {
                if payload != 0x9C {
                    return Err(Error::DecodeFailed {
                        command: cmd,
                        reason: format!(
                            "Expected payload 0x9C, found: {:#x}",
                            payload
                        ),
                    });
                }
                if receiver != 0 {
                    return Err(Error::DecodeFailed {
                        command: cmd,
                        reason: format!(
                            "Expected receiver 0x0, found: {:#x}",
                            receiver
                        ),
                    });
                }
                Ok(MarginResponse::NoCommand)
            }
            0b001 => {
                if let MarginCommand::Report(report) = cmd {
                    let report_response = match report {
                        ReportRequest::Capabilities => {
                            ReportResponse::Capabilities(
                                ReportCapabilities::from(payload),
                            )
                        }
                        ReportRequest::NumVoltageSteps => {
                            ReportResponse::NumVoltageSteps(payload)
                        }
                        ReportRequest::NumTimingSteps => {
                            ReportResponse::NumTimingSteps(payload)
                        }
                        ReportRequest::MaxTimingOffset => {
                            ReportResponse::MaxTimingOffset(payload)
                        }
                        ReportRequest::MaxVoltageOffset => {
                            ReportResponse::MaxVoltageOffset(payload)
                        }
                        ReportRequest::SamplingRateVoltage => {
                            ReportResponse::SamplingRateVoltage(payload)
                        }
                        ReportRequest::SamplingRateTiming => {
                            ReportResponse::SamplingRateTiming(payload)
                        }
                        ReportRequest::SampleCount => {
                            ReportResponse::SampleCount(payload)
                        }
                        ReportRequest::MaxLanes => {
                            ReportResponse::MaxLanes(payload)
                        }
                    };
                    Ok(MarginResponse::Report(report_response))
                } else {
                    return Err(Error::DecodeFailed {
                        command: cmd,
                        reason: format!(
                            "Found a Report response (margin type = {:#03b}), \
                                        for a non-Report command",
                            response_margin_type
                        ),
                    });
                }
            }
            0b010 => match payload {
                0x0F => Ok(MarginResponse::GoToNormalSettings),
                0x55 => Ok(MarginResponse::ClearErrorLog),
                _ => {
                    if payload & 0xC0 != 0xC0 {
                        Err(Error::DecodeFailed {
                            command: cmd,
                            reason: format!(
                                "Expected upper 2 bits set in the \
                            margin payload for a Set Error Count Limit response"
                            ),
                        })
                    } else {
                        Ok(MarginResponse::ErrorCountLimit(ErrorCount::from(
                            payload,
                        )))
                    }
                }
            },
            0b011 => Ok(MarginResponse::StepExecutionStatus {
                status: StepMarginExecutionStatus::from(payload),
                error_count: ErrorCount::from(payload),
            }),
            0b100 => Ok(MarginResponse::StepExecutionStatus {
                status: StepMarginExecutionStatus::from(payload),
                error_count: ErrorCount::from(payload),
            }),
            0b000 => Ok(MarginResponse::Empty),
            _ => Err(Error::DecodeFailed {
                command: cmd,
                reason: format!(
                    "found unexpected response margin type: {:#x?}",
                    response_margin_type
                ),
            }),
        }
    }

    pub fn margin_type(&self) -> u8 {
        match self {
            MarginResponse::NoCommand => 0b111,
            MarginResponse::Report(_) => 0b001,
            MarginResponse::ErrorCountLimit(_)
            | MarginResponse::GoToNormalSettings
            | MarginResponse::ClearErrorLog => 0b010,
            MarginResponse::StepExecutionStatus { .. } => 0b011,
            MarginResponse::Empty => 0b000,
        }
    }

    pub fn payload(&self) -> u8 {
        match self {
            MarginResponse::NoCommand => 0x9C,
            MarginResponse::Report(ReportResponse::Capabilities(p)) => {
                (*p).into()
            }
            MarginResponse::Report(ReportResponse::NumVoltageSteps(p)) => *p,
            MarginResponse::Report(ReportResponse::NumTimingSteps(p)) => *p,
            MarginResponse::Report(ReportResponse::MaxTimingOffset(p)) => *p,
            MarginResponse::Report(ReportResponse::MaxVoltageOffset(p)) => *p,
            MarginResponse::Report(ReportResponse::SamplingRateVoltage(p)) => {
                *p
            }
            MarginResponse::Report(ReportResponse::SamplingRateTiming(p)) => *p,
            MarginResponse::Report(ReportResponse::SampleCount(p)) => *p,
            MarginResponse::Report(ReportResponse::MaxLanes(p)) => *p,
            MarginResponse::ErrorCountLimit(count) => u8::from(*count),
            MarginResponse::GoToNormalSettings => 0x0F,
            MarginResponse::ClearErrorLog => 0x55,
            MarginResponse::StepExecutionStatus { status, error_count } => {
                u8::from(*status) | u8::from(*error_count)
            }
            MarginResponse::Empty => 0x00,
        }
    }
}

/// Combination of the margining control and status registers for a lane
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct MarginingLane {
    control: MarginingLaneControl,
    status: MarginingLaneStatus,
}

/// The Margining Lane Control Register
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct MarginingLaneControl {
    receiver: Receiver,
    cmd: MarginCommand,
}

impl From<u16> for MarginingLaneControl {
    fn from(word: u16) -> Self {
        let receiver = Receiver::new((word & 0b111) as u8).unwrap();
        let cmd = MarginCommand::from(word);
        let usage_model = ((word >> 6) & 0b1) as u8;
        assert_eq!(usage_model, 0);
        Self { receiver, cmd }
    }
}

/// The Margining Lane Status Register
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct MarginingLaneStatus {
    receiver: Receiver,
    response: MarginResponse,
}

impl MarginingLaneStatus {
    pub fn decode_for_cmd(
        cmd: MarginCommand,
        word: u16,
    ) -> Result<Self, Error> {
        let receiver = Receiver::new((word & 0b111) as u8).unwrap();
        let response = MarginResponse::decode_for_cmd(cmd, word)?;
        Ok(Self { receiver, response })
    }
}

#[derive(Debug)]
struct LaneMarginInner {
    device: PcieDevice,
    receiver: Receiver,
    lane: Lane,
    header: LaneMarginingCapabilityHeader,
    cmd_offset: usize,
    sts_offset: usize,
    verbosity: u64,
}

impl LaneMarginInner {
    fn report(&self, request: ReportRequest) -> Result<ReportResponse, Error> {
        let cmd = MarginCommand::Report(request);
        self.write_command(cmd)?;
        match self.wait_for_response(cmd)? {
            MarginResponse::Report(report) => Ok(report),
            other => Err(Error::Margin(format!(
                "Expected response for {:?}, found: {:?}",
                request, other
            ))),
        }
    }

    fn report_capabilities(&self) -> Result<ReportCapabilities, Error> {
        if let ReportResponse::Capabilities(caps) =
            self.report(ReportRequest::Capabilities)?
        {
            Ok(caps)
        } else {
            Err(Error::Margin(format!(
                "Expected ReportResponse::ReportCapabilities"
            )))
        }
    }

    fn no_command(&self) -> Result<(), Error> {
        let cmd = MarginCommand::NoCommand;
        self.write_command(cmd)?;
        match self.wait_for_response(cmd)? {
            MarginResponse::NoCommand => Ok(()),
            other => Err(Error::Margin(format!(
                "Margining failed, expected No Command response, found: {:?}",
                other
            ))),
        }
    }

    fn clear_error_log(&self) -> Result<(), Error> {
        let cmd = MarginCommand::ClearErrorLog;
        self.write_command(cmd)?;
        match self.wait_for_response(cmd)? {
            MarginResponse::ClearErrorLog => Ok(()),
            other => Err(Error::Margin(format!(
                "Margining failed, expected Clear Error Log response, found: {:?}",
                other
            ))),
        }
    }

    fn go_to_normal_settings(&self) -> Result<(), Error> {
        let cmd = MarginCommand::GoToNormalSettings;
        self.write_command(cmd)?;
        match self.wait_for_response(cmd)? {
            MarginResponse::GoToNormalSettings => Ok(()),
            other => Err(Error::Margin(format!(
                "Margining failed, expected Go To Normal Settings response, found: {:?}",
                other
            ))),
        }
    }

    fn build_command(&self, cmd: MarginCommand) -> u16 {
        let margin_type = cmd.margin_type();
        let payload = cmd.payload();
        let receiver = if matches!(cmd, MarginCommand::NoCommand) {
            0
        } else {
            u8::from(self.receiver)
        };
        if self.verbosity >= verbosity::COMMAND_DETAIL {
            println!(
                "-> {cmd:?}, Margin type: {margin_type:#03b}, \
                 Receiver: {receiver:#03b}, Payload: {payload:#08b}"
            );
        } else if self.verbosity >= verbosity::COMMANDS {
            println!("-> {cmd:?}");
        }
        let words = [margin_type << 3 | receiver, payload];
        u16::from_le_bytes(words)
    }

    fn wait_for_response(
        &self,
        cmd: MarginCommand,
    ) -> Result<MarginResponse, Error> {
        const WAIT_INTERVAL: Duration = Duration::from_micros(10);
        const MAX_WAIT_TIME: Duration = Duration::from_millis(10);
        let now = Instant::now();
        let response = loop {
            // while now.elapsed() < MAX_WAIT_TIME {
            sleep(WAIT_INTERVAL);

            // If we find the wrong command, we actually continue up to the
            // maximum wait time.
            let response = self.read_command_response(cmd);
            if let Ok(response) = response {
                if self.verbosity >= verbosity::COMMAND_DETAIL {
                    println!(
                        "<- {response:?}, Margin type: {:#03b}, \
                        Receiver: {:#03b}, Payload: {:#08b}",
                        response.margin_type(),
                        u8::from(self.receiver),
                        response.payload(),
                    );
                } else if self.verbosity >= verbosity::COMMANDS {
                    println!("<- {response:?}");
                }
                if cmd.expects_response(response) {
                    return Ok(response);
                }
            }

            if now.elapsed() < MAX_WAIT_TIME {
                continue;
            }
            break response;
        };
        Err(Error::Margin(format!(
            concat!(
                "margining failed, did not find expected response ",
                "for command {:?} within time limit of {:?}: ",
                "instead found {:?}",
            ),
            cmd, MAX_WAIT_TIME, response
        )))
    }

    fn write_command(&self, cmd: MarginCommand) -> Result<(), Error> {
        let word = self.build_command(cmd);
        write_configuration_space(
            &self.device.file,
            &self.device.bdf,
            self.cmd_offset,
            word,
        )
    }

    fn read_command_response(
        &self,
        cmd: MarginCommand,
    ) -> Result<MarginResponse, Error> {
        let word = read_configuration_space(
            &self.device.file,
            &self.device.bdf,
            self.sts_offset,
        )?;
        MarginResponse::decode_for_cmd(cmd, word)
    }

    fn gather_limits(
        &self,
        caps: &ReportCapabilities,
    ) -> Result<MarginingLimits, Error> {
        let num_voltage_steps = if caps.voltage_supported {
            match self.report(ReportRequest::NumVoltageSteps)? {
                ReportResponse::NumVoltageSteps(steps) => {
                    let steps = Some(steps);
                    self.no_command()?;
                    steps
                }
                other => {
                    return Err(Error::Margin(format!(
                        "Unexpected response requesting NumVoltageSteps: {:?}",
                        other
                    )));
                }
            }
        } else {
            None
        };

        let num_timing_steps =
            match self.report(ReportRequest::NumTimingSteps)? {
                ReportResponse::NumTimingSteps(steps) => {
                    self.no_command()?;
                    steps
                }
                other => {
                    return Err(Error::Margin(format!(
                        "Unexpected response requesting NumTimingSteps: {:?}",
                        other
                    )));
                }
            };

        let max_timing_offset =
            match self.report(ReportRequest::MaxTimingOffset)? {
                ReportResponse::MaxTimingOffset(offset) => {
                    self.no_command()?;
                    offset
                }
                other => {
                    return Err(Error::Margin(format!(
                        "Unexpected response requesting MaxTimingOffset: {:?}",
                        other
                    )));
                }
            };

        let max_voltage_offset = if caps.voltage_supported {
            match self.report(ReportRequest::MaxVoltageOffset)? {
                ReportResponse::MaxVoltageOffset(offset) => {
                    let offset = Some(offset);
                    self.no_command()?;
                    offset
                }
                other => {
                    return Err(Error::Margin(format!(
                        "Unexpected response requesting MaxVoltageOffset: {:?}",
                        other
                    )));
                }
            }
        } else {
            None
        };

        let sampling_rate_voltage = if caps.voltage_supported {
            match self.report(ReportRequest::SamplingRateVoltage)? {
                ReportResponse::SamplingRateVoltage(rate) => {
                    let rate = Some(rate);
                    self.no_command()?;
                    rate
                }
                other => {
                    return Err(Error::Margin(format!(
                        "Unexpected response requesting SamplingRateVoltage: {:?}",
                        other
                    )));
                }
            }
        } else {
            None
        };

        let sampling_rate_timing =
            match self.report(ReportRequest::SamplingRateTiming)? {
                ReportResponse::SamplingRateTiming(rate) => {
                    self.no_command()?;
                    rate
                }
                other => {
                    return Err(Error::Margin(format!(
                    "Unexpected response requesting SamplingRateTiming: {:?}",
                    other
                )));
                }
            };

        Ok(MarginingLimits {
            num_voltage_steps,
            num_timing_steps,
            max_timing_offset,
            max_voltage_offset,
            sampling_rate_voltage,
            sampling_rate_timing,
        })
    }

    pub fn set_error_count_limit(&self, count: u8) -> Result<(), Error> {
        let limit = ErrorCount::from(count);
        let cmd = MarginCommand::SetErrorCountLimit(limit);
        self.write_command(cmd)?;
        let response = self.wait_for_response(cmd)?;
        match response {
            MarginResponse::ErrorCountLimit(actual_limit) => {
                if limit == actual_limit {
                    Ok(())
                } else {
                    Err(Error::Margin(format!(
                        "Failed to set error count limit"
                    )))
                }
            }
            _ => Err(Error::Margin(format!(
                "Expected set error limit response, found: {response:?})"
            ))),
        }
    }

    fn margin_at(
        &self,
        cmd: MarginCommand,
        duration: Duration,
    ) -> Result<(Duration, MarginResult), Error> {
        self.write_command(cmd)?;

        // Interval between checks when execution status is "setup"
        const INTERVAL: Duration = Duration::from_millis(1);

        // Total duration before setup must complete
        const TOTAL_DURATION: Duration = Duration::from_millis(200);

        let now = Instant::now();
        loop {
            let response = self.wait_for_response(cmd)?;
            match response {
                MarginResponse::StepExecutionStatus { status, error_count } => {
                    match status {
                        StepMarginExecutionStatus::Nak => {
                            return Err(Error::Margin(format!(
                            "Margin failed, step execution status returned NAK"
                        )));
                        }
                        StepMarginExecutionStatus::Setup => {
                            if now.elapsed() > TOTAL_DURATION {
                                return Err(Error::Margin(format!(
                                    "Failed to finish margin setup within {:?}",
                                    TOTAL_DURATION,
                                )));
                            }
                            sleep(INTERVAL);
                            continue;
                        }
                        StepMarginExecutionStatus::TooManyErrors => {
                            let result = MarginResult::Failed(error_count);
                            return Ok((now.elapsed(), result));
                        }
                        StepMarginExecutionStatus::InProgress => break,
                    }
                }
                _ => {
                    return Err(Error::Margin(format!(
                        "Expected step execution status response, found: {response:?})"
                    )));
                }
            }
        }

        // At this point, we're InProgress. Wait for the desired duration, then
        // read the error count
        sleep(duration);
        let response = self.wait_for_response(cmd)?;
        match response {
            MarginResponse::StepExecutionStatus {
                status: StepMarginExecutionStatus::InProgress,
                error_count,
            } => {
                self.no_command()?;
                self.clear_error_log()?;
                self.no_command()?;
                self.go_to_normal_settings()?;
                let result = MarginResult::Success(error_count);
                return Ok((now.elapsed(), result));
            }
            MarginResponse::StepExecutionStatus {
                status: StepMarginExecutionStatus::TooManyErrors,
                error_count,
            } => {
                let result = MarginResult::Failed(error_count);
                return Ok((now.elapsed(), result));
            }
            _ => {
                return Err(Error::Margin(format!(
                    "Margining failed, expected step margin execution status with InProgress, found {:?}",
                    response,
                )));
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MarginingLimits {
    // None if not supported
    pub num_voltage_steps: Option<u8>,
    pub num_timing_steps: u8,
    pub max_timing_offset: u8,
    pub max_voltage_offset: Option<u8>,
    pub sampling_rate_voltage: Option<u8>,
    pub sampling_rate_timing: u8,
}

#[derive(Debug)]
pub struct LaneMargin {
    inner: LaneMarginInner,
    capabilities: ReportCapabilities,
    limits: MarginingLimits,
}

impl LaneMargin {
    pub fn new(
        device: PcieDevice,
        receiver: Receiver,
        lane: Lane,
        verbosity: u64,
    ) -> Result<Self, Error> {
        // Find the size of the capability header, limited to the width of the
        // link.
        let link_status = device.link_status()?;
        let n_lanes = usize::from(u8::from(link_status.width));
        // 2 u16s for each lane (control / status) + 2 for the port capability / status
        let n_bytes = (n_lanes + 1) * std::mem::size_of::<u16>() * 2;
        let data = device.read_extended_capability_data::<u8>(
            &ExtendedCapabilityId::LaneMargining,
            n_bytes,
        )?;
        let header = LaneMarginingCapabilityHeader::try_from(data.as_slice())?;
        if !header.port_status.ready {
            return Err(Error::Margin(format!(
                "Margining appears unsupported on this device"
            )));
        }
        let header_offset = *device
            .find_extended_capability(&ExtendedCapabilityId::LaneMargining)
            .ok_or_else(|| {
                Error::Margin(format!(
                    "Failed to read offset of Lane Margining extended capability"
                ))
            })?
            .0;
        let cmd_offset = header_offset + header.lane_control_offset(lane)?;
        let sts_offset = header_offset + header.lane_status_offset(lane)?;
        let inner = LaneMarginInner {
            device,
            receiver,
            header,
            lane,
            cmd_offset,
            sts_offset,
            verbosity,
        };
        let capabilities = inner.report_capabilities()?;
        inner.no_command()?;
        let limits = inner.gather_limits(&capabilities)?;
        Ok(Self { inner, capabilities, limits })
    }

    pub fn vendor_id(&self) -> u16 {
        self.inner.device.vendor_id
    }

    pub fn device_id(&self) -> u16 {
        self.inner.device.device_id
    }

    pub fn header(&self) -> &LaneMarginingCapabilityHeader {
        &self.inner.header
    }

    pub fn lane(&self) -> Lane {
        self.inner.lane
    }

    pub fn receiver(&self) -> Receiver {
        self.inner.receiver
    }

    pub fn capabilities(&self) -> &ReportCapabilities {
        &self.capabilities
    }

    pub fn supports_voltage_margining(&self) -> bool {
        self.limits.num_voltage_steps.is_some()
    }

    pub fn limits(&self) -> &MarginingLimits {
        &self.limits
    }

    pub fn set_error_count_limit(&self, count: u8) -> Result<(), Error> {
        self.inner.set_error_count_limit(count)
    }

    pub fn no_command(&self) -> Result<(), Error> {
        self.inner.no_command()
    }

    pub fn clear_error_log(&self) -> Result<(), Error> {
        self.inner.clear_error_log()
    }

    pub fn go_to_normal_settings(&self) -> Result<(), Error> {
        self.inner.go_to_normal_settings()
    }

    pub fn margin_at_left_right(
        &self,
        step: StepLeftRight,
        duration: Duration,
    ) -> Result<(Duration, MarginResult), Error> {
        let cmd = MarginCommand::StepLeftRight(step);
        self.inner.margin_at(cmd, duration)
    }

    pub fn margin_at_up_down(
        &self,
        step: StepUpDown,
        duration: Duration,
    ) -> Result<(Duration, MarginResult), Error> {
        let cmd = MarginCommand::StepUpDown(step);
        self.inner.margin_at(cmd, duration)
    }

    pub fn iter_left_right_steps(&self) -> Vec<StepLeftRight> {
        let steps = self.limits().num_timing_steps;
        let base = 1..=steps;
        let right = base.clone().map(|pt| StepLeftRight {
            direction: Some(LeftRight::Right),
            steps: Steps::from(pt),
        });
        if self.capabilities().independent_left_right_sampling {
            base.rev()
                .map(|pt| StepLeftRight {
                    direction: Some(LeftRight::Left),
                    steps: Steps::from(pt),
                })
                .chain(right)
                .collect()
        } else {
            right.collect()
        }
    }

    pub fn iter_up_down_steps(&self) -> Vec<StepUpDown> {
        match self.limits().num_voltage_steps {
            None => vec![],
            Some(steps) => {
                let base = 1..=steps;
                let up = base.clone().map(|pt| StepUpDown {
                    direction: Some(UpDown::Up),
                    steps: Steps::from(pt),
                });
                if self.capabilities().independent_up_down_voltage {
                    base.rev()
                        .map(|pt| StepUpDown {
                            direction: Some(UpDown::Down),
                            steps: Steps::from(pt),
                        })
                        .into_iter()
                        .chain(up)
                        .collect()
                } else {
                    up.collect()
                }
            }
        }
    }
}

/// A PCIe Bus/Device/Function, representing a single PCIe receiver.
#[derive(Debug, Clone, Copy)]
pub struct Bdf {
    bus: u8,
    device: u8,
    function: u8,
}

impl std::str::FromStr for Bdf {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts = s.splitn(3, '/').collect::<Vec<_>>();
        if parts.len() == 3 {
            let converted = parts
                .iter()
                .map(|part| u8::from_str_radix(part, 16))
                .collect::<Result<Vec<u8>, _>>()
                .map_err(|_| Error::InvalidBdf(s.to_string()))?;
            Ok(Bdf {
                bus: converted[0],
                device: converted[1],
                function: converted[2],
            })
        } else {
            Err(Error::InvalidBdf(s.to_string()))
        }
    }
}

impl fmt::Display for Bdf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:x}/{:x}/{:x}", self.bus, self.device, self.function)
    }
}

/// The PCI Base Address Register
#[derive(Debug, Clone, Copy)]
enum Bar {
    Configuration,
    #[allow(dead_code)]
    Bar(NonZeroU8),
}

impl From<Bar> for u8 {
    fn from(bar: Bar) -> u8 {
        match bar {
            Bar::Configuration => 0,
            Bar::Bar(x) => x.get(),
        }
    }
}

/// Trait used to read fixed-size words from a `Bdf`.
pub trait DataWord: Sized + Copy + std::fmt::LowerHex + PartialEq {
    fn size_attribute() -> u32;
    fn from_raw(_: u64) -> Self;
    fn to_raw(self) -> u64;
    fn invalid_word() -> Self;
}

impl DataWord for u8 {
    fn size_attribute() -> u32 {
        0x0
    }

    fn from_raw(data: u64) -> Self {
        (data & 0xFF) as u8
    }

    fn to_raw(self) -> u64 {
        self as _
    }

    fn invalid_word() -> Self {
        !0
    }
}

impl DataWord for u16 {
    fn size_attribute() -> u32 {
        0x1
    }

    fn from_raw(data: u64) -> Self {
        (data & 0xFFFF) as u16
    }

    fn to_raw(self) -> u64 {
        self as _
    }

    fn invalid_word() -> Self {
        !0
    }
}

impl DataWord for u32 {
    fn size_attribute() -> u32 {
        0x2
    }

    fn from_raw(data: u64) -> Self {
        (data & 0xFFFF_FFFF) as u32
    }

    fn to_raw(self) -> u64 {
        self as _
    }

    fn invalid_word() -> Self {
        !0
    }
}

impl DataWord for u64 {
    fn size_attribute() -> u32 {
        0x3
    }

    fn from_raw(data: u64) -> Self {
        data
    }

    fn to_raw(self) -> u64 {
        self as _
    }

    fn invalid_word() -> Self {
        !0
    }
}

/// A handle to a PCIe device on the system.
#[derive(Debug)]
pub struct PcieDevice {
    pub vendor_id: u16,
    pub device_id: u16,
    pub bdf: Bdf,
    capabilities: BTreeMap<usize, CapabilityId>,
    extended_capabilities: BTreeMap<usize, ExtendedCapability>,
    file: File,
}

impl PcieDevice {
    /// Construct a PCIe device to operate on a single bus/device/function.
    pub fn new(bdf: Bdf) -> Result<Self, Error> {
        let file =
            File::options().read(true).write(true).open(PCI_REG_DEVICE_FILE)?;
        let vendor_id = read_configuration_space::<u16>(
            &file,
            &bdf,
            PCIE_VENDOR_ID_OFFSET,
        )?;
        let device_id = read_configuration_space::<u16>(
            &file,
            &bdf,
            PCIE_DEVICE_ID_OFFSET,
        )?;
        let capabilities = Self::read_capabilities(&file, &bdf)?;
        let extended_capabilities =
            Self::read_extended_capabilities(&file, &bdf)?;
        Ok(Self {
            vendor_id,
            device_id,
            bdf,
            capabilities,
            extended_capabilities,
            file,
        })
    }

    /// Try to clone `self`.
    ///
    /// This is fallible because it opens a new file for operating on the PCIe
    /// registers.
    pub fn try_clone(&self) -> Result<Self, Error> {
        let file =
            File::options().read(true).write(true).open(PCI_REG_DEVICE_FILE)?;
        Ok(Self {
            vendor_id: self.vendor_id,
            device_id: self.device_id,
            bdf: self.bdf,
            capabilities: self.capabilities.clone(),
            extended_capabilities: self.extended_capabilities.clone(),
            file,
        })
    }

    pub fn from_node(n: &devinfo::Node) -> Option<Self> {
        let mut pw = n.props();

        while let Some(p) = pw.next().transpose().ok()? {
            match p.name().as_str() {
                "reg" => {
                    if let Some(v) = p.as_i32() {
                        let uv = v as u32;
                        return Self::new(Bdf {
                            bus: ((uv & 0xff0000) >> 16) as u8,
                            device: ((uv & 0xf800) >> 11) as u8,
                            function: ((uv & 0x700) >> 8) as u8,
                        })
                        .ok();
                    }
                }
                _ => {}
            }
        }

        None
    }

    /// Return the PCIe capabilities for this device
    pub fn capabilities(&self) -> Vec<CapabilityId> {
        self.capabilities.values().copied().collect()
    }

    /// Return `true` if the capability `id` is supported
    pub fn supports_capability(&self, id: &CapabilityId) -> bool {
        self.find_capability(id).is_some()
    }

    /// Return the PCIe extended capabilities for this device
    pub fn extended_capabilities(&self) -> Vec<ExtendedCapability> {
        self.extended_capabilities.values().copied().collect()
    }

    /// Return `true` if the PCIe extended capability `id` is supported
    pub fn supports_extended_capability(
        &self,
        id: &ExtendedCapabilityId,
    ) -> bool {
        self.find_extended_capability(id).is_some()
    }

    // Return the register offset and capability ID of the requested
    // capability, or `None` if it's not supported by the device
    fn find_capability(
        &self,
        cap: &CapabilityId,
    ) -> Option<(&usize, &CapabilityId)> {
        self.capabilities.iter().find(|(_, capability)| capability == &cap)
    }

    // Return the register offset and capability ID of the requested
    // extended capability, or `None` if it's not supported by the device
    fn find_extended_capability(
        &self,
        cap: &ExtendedCapabilityId,
    ) -> Option<(&usize, &ExtendedCapability)> {
        self.extended_capabilities
            .iter()
            .find(|(_, capability)| capability.id == *cap)
    }

    // Read all the capabilities from configuration space for this device
    fn read_capabilities(
        file: &File,
        bdf: &Bdf,
    ) -> Result<BTreeMap<usize, CapabilityId>, Error> {
        let mut cap_start = usize::from(read_configuration_space::<u8>(
            file,
            bdf,
            PCIE_CAP_POINTER,
        )?);
        assert_ne!(
            cap_start, 0,
            "PCIe devices must implement at least 2 capabilities"
        );
        let mut caps = BTreeMap::new();
        loop {
            let word = read_configuration_space::<u16>(file, bdf, cap_start)?;
            let cap = PcieCapabilityHeader::from(word);
            caps.insert(cap_start, cap.id);
            if cap.next == 0 {
                return Ok(caps);
            }
            cap_start = usize::from(cap.next);
            assert!(cap_start < PCIE_CONFIGURATION_SPACE_SIZE);
        }
    }

    // Read all extended capabilities from configuration space for this device
    fn read_extended_capabilities(
        file: &File,
        bdf: &Bdf,
    ) -> Result<BTreeMap<usize, ExtendedCapability>, Error> {
        let mut cap_start = PCIE_EXTENDED_CAPABILITY_HEADER_OFFSET;
        let mut caps = BTreeMap::new();
        loop {
            let word = read_configuration_space::<u32>(file, bdf, cap_start)?;
            let cap = PcieExtendedCapabilityHeader::from(word);
            // There may be no extended capabilities
            if matches!(cap.capability.id, ExtendedCapabilityId::Other(0)) {
                return Ok(caps);
            }
            caps.insert(cap_start, cap.capability);
            if cap.next == 0 {
                return Ok(caps);
            }
            cap_start = usize::from(cap.next);
            assert!(cap_start < PCIE_CONFIGURATION_SPACE_SIZE);
        }
    }

    /// Return the Link Status Register for this device
    pub fn link_status(&self) -> Result<LinkStatus, Error> {
        let pcie_cap_start = self
            .find_capability(&CapabilityId::Pcie)
            .expect("All PCIe devices must implement the PCIe Capability")
            .0;
        LinkStatus::try_from(read_configuration_space::<u16>(
            &self.file,
            &self.bdf,
            pcie_cap_start + LinkStatus::REGISTER_OFFSET,
        )?)
    }

    // Read `len` items of data associated with the requested capability, if
    // exists. This skips the extended capability header itself.
    pub fn read_extended_capability_data<T>(
        &self,
        id: &ExtendedCapabilityId,
        len: usize,
    ) -> Result<Vec<T>, Error>
    where
        T: DataWord,
    {
        let (&cap_start, _) = self
            .find_extended_capability(id)
            .ok_or_else(|| Error::UnsupportedExtendedCap(*id))?;
        let data_start = cap_start + std::mem::size_of::<u32>(); // Skip the capability header itself
        let data_end = data_start + len;
        let mut data = Vec::with_capacity(len);
        for offset in data_start..data_end {
            data.push(read_configuration_space::<T>(
                &self.file, &self.bdf, offset,
            )?);
        }
        Ok(data)
    }
}

fn read_configuration_space<W>(
    file: &File,
    bdf: &Bdf,
    offset: usize,
) -> Result<W, Error>
where
    W: DataWord,
{
    let acc_attr = PCI_ATTR_ENDIANNESS | W::size_attribute();
    let mut register = PciRegister {
        user_version: 0x01,
        bus_no: bdf.bus,
        dev_no: bdf.device,
        func_no: bdf.function,
        barnum: u8::from(Bar::Configuration),
        offset: offset as u64,
        acc_attr,
        ..Default::default()
    };
    let ret = unsafe {
        libc::ioctl(
            file.as_raw_fd(),
            PCITOOL_DEVICE_GET_REG,
            &mut register as *mut _ as *mut libc::c_void,
        )
    };
    if ret == 0 {
        let data = W::from_raw(register.data);
        if data == W::invalid_word() {
            Err(Error::InvalidPcieRead(offset, register.data))
        } else {
            Ok(data)
        }
    } else {
        Err(io::Error::last_os_error().into())
    }
}

fn write_configuration_space<W>(
    file: &File,
    bdf: &Bdf,
    offset: usize,
    word: W,
) -> Result<(), Error>
where
    W: DataWord,
{
    let acc_attr = PCI_ATTR_ENDIANNESS | W::size_attribute();
    let mut register = PciRegister {
        user_version: 0x01,
        bus_no: bdf.bus,
        dev_no: bdf.device,
        func_no: bdf.function,
        barnum: u8::from(Bar::Configuration),
        offset: offset as u64,
        acc_attr,
        data: word.to_raw(),
        ..Default::default()
    };
    let ret = unsafe {
        libc::ioctl(
            file.as_raw_fd(),
            PCITOOL_DEVICE_SET_REG,
            &mut register as *mut _ as *mut libc::c_void,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error().into())
    }
}

#[derive(Debug)]
pub struct PcieNode {
    device: PcieDevice,
    driver: Option<String>,
    instance: Option<i32>,
    margin: bool,
}

#[derive(Debug)]
pub struct PcieBridge {
    pub bridge: PcieNode,
    pub children: Vec<PcieNode>,
}

impl PcieNode {
    pub fn new(
        device: PcieDevice,
        driver: Option<String>,
        instance: Option<i32>,
    ) -> Self {
        let margin = device
            .read_extended_capability_data::<u8>(
                &ExtendedCapabilityId::LaneMargining,
                8,
            )
            .ok()
            .and_then(|data| {
                LaneMarginingCapabilityHeader::try_from(data.as_slice()).ok()
            })
            .map(|h| h.port_status.ready)
            .unwrap_or(false);
        Self { device, driver, instance, margin }
    }
}

impl fmt::Display for PcieNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}{} -- {:#x} {:#x} -- {}{}",
            self.device.bdf,
            self.driver.clone().unwrap_or("??".to_string()),
            self.instance.unwrap_or(-1),
            self.device.vendor_id,
            self.device.device_id,
            self.device.link_status().expect("could not retrieve link status"),
            if self.margin { "  SUPPORTED" } else { "" }
        )
    }
}

impl PcieBridge {
    pub fn new(
        device: PcieDevice,
        driver: Option<String>,
        instance: Option<i32>,
    ) -> Self {
        Self {
            bridge: PcieNode::new(device, driver, instance),
            children: vec![],
        }
    }
}

fn enum_pcie_devices() -> Result<Vec<PcieBridge>> {
    let mut di = devinfo::DevInfo::new()?;
    let mut w = di.walk_driver("pcieb");
    let mut bridges: Vec<PcieBridge> = vec![];

    while let Some(n) = w.next().transpose()? {
        if let Some(dev) = PcieDevice::from_node(&n) {
            if !matches!(dev.link_status(), Ok(l) if l.active) {
                continue;
            }
            let mut bridge =
                PcieBridge::new(dev, n.driver_name(), n.instance());

            let mut cdi = devinfo::DevInfo::new()?;
            let mut childwalk = cdi.walk_node();
            if let Ok(path) = n.devfs_path() {
                while let Some(c) = childwalk.next().transpose()? {
                    match c.parent()? {
                        Some(parent)
                            if parent
                                .devfs_path()
                                .unwrap_or("".to_string())
                                == path =>
                        {
                            if let Some(cdev) = PcieDevice::from_node(&c) {
                                if cdev.vendor_id != 0x1022 {
                                    bridge.children.push(PcieNode::new(
                                        cdev,
                                        c.driver_name(),
                                        c.instance(),
                                    ));
                                }
                            }
                        }
                        _ => continue,
                    }
                }
            }

            if bridge.children.len() > 0 {
                bridges.push(bridge);
            }
        }
    }
    Ok(bridges)
}

/// A point at which the margining protocol was run.
#[derive(Debug, Clone, Copy)]
pub enum MarginPoint {
    /// Margined at a non-default position in time, with the percent of the unit
    /// interval (UI) as the argument.
    Time(f64),
    /// Margined at a non-default position in voltage, with the actual voltage
    /// as the argument.
    Voltage(f64),
}

impl MarginPoint {
    /// Return the time offset of the margin point as a float.
    pub fn time(&self) -> f64 {
        match self {
            MarginPoint::Time(t) => *t,
            _ => 0.0,
        }
    }

    /// Return the voltage offset of the margin point as a float.
    pub fn voltage(&self) -> f64 {
        match self {
            MarginPoint::Voltage(v) => *v,
            _ => 0.0,
        }
    }
}

/// The result of margining a single point.
#[derive(Debug, Clone, Copy)]
pub enum MarginResult {
    Success(ErrorCount),
    Failed(ErrorCount),
}

fn write_file_header(
    outfile: &mut File,
    device: &PcieDevice,
    lane: &Lane,
) -> std::io::Result<()> {
    writeln!(outfile, "Vendor ID: {:#x}", device.vendor_id)?;
    writeln!(outfile, "Device ID: {:#x}", device.device_id)?;
    writeln!(outfile, "Lane: {}", lane)?;
    writeln!(outfile, "Time (%UI)\tVoltage (V)\tDuration (s)\tCount\tPass")
}

fn open_margin_results_file(
    dir: Option<&str>,
    device: &PcieDevice,
    lane: &Lane,
) -> anyhow::Result<(File, String)> {
    let mut filename = format!(
        "margin-results-b{:x}-d{:x}-f{:x}-l{}",
        device.bdf.bus, device.bdf.device, device.bdf.function, lane,
    );

    if dir.is_none() {
        let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%S");
        filename.push_str(&format!("{timestamp}.txt"));
    }

    let path: PathBuf = match dir {
        Some(d) => Path::new(d).join(&filename),
        None => PathBuf::from(&filename),
    };

    let mut outfile = File::options()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| {
            format!("Failed to create output file at {}", path.display())
        })?;
    write_file_header(&mut outfile, device, lane)?;
    Ok((outfile, path.display().to_string()))
}

pub trait OutputExt {
    fn info(&self) -> String;
}

impl OutputExt for std::process::Output {
    fn info(&self) -> String {
        let mut out = String::new();

        if let Some(code) = self.status.code() {
            out.push_str(&format!("exit code {}", code));
        }

        let stderr = String::from_utf8_lossy(&self.stderr).trim().to_string();
        let extra = if stderr.is_empty() {
            String::from_utf8_lossy(&self.stdout).trim().to_string()
        } else {
            stderr
        };

        if !extra.is_empty() {
            if !out.is_empty() {
                out.push_str(": ");
            }
            out.push_str(&extra);
        }

        out
    }
}

fn margin_one(
    args: &Args,
    dir: &PathBuf,
    node: &PcieNode,
    port: Port,
) -> Result<()> {
    if !node.margin {
        return Ok(());
    }

    println!("Margining {}", node);

    let suffix = format!(
        "{}{}-b{:x}-d{:x}-f{:x}",
        node.driver.clone().unwrap_or("unknown".to_string()),
        node.instance.unwrap_or(-1),
        node.device.bdf.bus,
        node.device.bdf.device,
        node.device.bdf.function
    );

    // Save the config space
    let mut of = dir.clone();
    of.push(&format!("cfgspace-{suffix}"));
    let out = Command::new(PCIEADM)
        .env_clear()
        .arg("save-cfgspace")
        .arg("-d")
        .arg(&format!("{}", node.device.bdf))
        .arg(of)
        .output()?;
    if !out.status.success() {
        println!(
            "`pcieadm save-cfgspace` for {} failed with: {}",
            node.device.bdf,
            out.info()
        );
    }

    // Save information for NVMe devices
    if node.driver.as_deref() == Some("nvme") {
        let out = Command::new(NVMEADM)
            .env_clear()
            .arg("identify")
            .arg(&format!("nvme{}", node.instance.expect("instance not set")))
            .output()?;
        if !out.status.success() {
            println!(
                "`nvmeadm identify` for {} failed with: {}",
                node.device.bdf,
                out.info()
            );
        } else {
            let mut f = dir.clone();
            f.push(&format!("nvmeinfo-{suffix}"));
            let mut file = File::create(f)?;
            file.write_all(&out.stdout)?;
        }
    }

    let link_status =
        node.device.link_status().context("Failed to get link status")?;
    let lanes = LaneDescription::All.to_indices(&link_status)?;
    let receiver = match port {
        Port::Upstream => Receiver::upstream(),
        Port::Downstream => Receiver::downstream(),
    };

    run_margin(dir.to_str(), &node.device, lanes, receiver, args)
}

fn zip_dir(src_dir: &Path, dst_file: &Path) -> anyhow::Result<()> {
    let file = File::create(dst_file)?;
    let mut zip = ZipWriter::new(file);

    let options = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    let src_dir = src_dir.canonicalize()?;
    let top = src_dir.file_name().unwrap_or_default();

    for entry in WalkDir::new(&src_dir) {
        let entry = entry?;
        let path = entry.path();
        let rel_path = path.strip_prefix(&src_dir)?;
        let name = Path::new(top).join(rel_path);

        if path.is_file() {
            let mut f = File::open(path)?;
            let mut buffer = Vec::new();
            f.read_to_end(&mut buffer)?;

            zip.start_file(name.to_string_lossy(), options)?;
            zip.write_all(&buffer)?;
        } else if path.is_dir() {
            zip.add_directory(name.to_string_lossy(), options)?;
        }
    }

    zip.finish()?;
    Ok(())
}

fn margin_all(args: Args, bridges: Vec<PcieBridge>) -> Result<()> {
    let dirname =
        format!("margin-{}", chrono::Utc::now().format("%Y-%m-%dT%H-%M-%S"));
    fs::create_dir(&dirname).context("Failed to create output directory")?;

    let dir = PathBuf::from(&dirname);

    let devs = Command::new(PCIEADM).env_clear().arg("show-devs").output()?;
    let mut f = dir.clone();
    f.push("dev.list");
    let mut file = File::create(f)?;
    file.write_all(&devs.stdout)?;

    let devs = Command::new(PCIEADM)
        .env_clear()
        .arg("show-devs")
        .arg("-o")
        .arg("bdf,instance,vid,did,path")
        .arg("nvme")
        .output()?;
    let mut f = dir.clone();
    f.push("nvme.list");
    let mut file = File::create(f)?;
    file.write_all(&devs.stdout)?;

    for b in &bridges {
        for c in &b.children {
            margin_one(&args, &dir, c, Port::Upstream)?;
        }
        margin_one(&args, &dir, &b.bridge, Port::Downstream)?;
    }

    let zipfile = PathBuf::from(&dir).with_extension("zip");
    zip_dir(&dir, &zipfile)?;

    println!("Created {}", zipfile.display());

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.probe {
        let bridges = enum_pcie_devices()?;

        for b in &bridges {
            println!("{}", b.bridge);

            for c in &b.children {
                println!("    {}", c);
            }
        }
        if !args.report_only {
            println!("---");
            margin_all(args, bridges)?;
        }
        return Ok(());
    }

    let device = match args.bdf {
        Some(bdf) => {
            PcieDevice::new(bdf).context("Failed to create PCIe device")?
        }
        None => {
            return Err(ClapError::raw(
                ErrorKind::MissingRequiredArgument,
                "<BDF> is required in this context",
            )
            .into());
        }
    };

    let link_status =
        device.link_status().context("Failed to get link status")?;

    if args.verbose > verbosity::LINK_STATUS {
        println!("lmar: {:#x?}", link_status);
    }

    let receiver = match args.port {
        Some(Port::Upstream) => Receiver::upstream(),
        Some(Port::Downstream) => Receiver::downstream(),
        _ => {
            return Err(ClapError::raw(
                ErrorKind::MissingRequiredArgument,
                "<PORT> is required in this context",
            )
            .into());
        }
    };

    let lanes = args.lanes.to_indices(&link_status)?;
    if args.verbose >= verbosity::PROGRESS_SUMMARY {
        println!("lmar: margining lanes: {lanes:?}");
    }

    run_margin(None, &device, lanes, receiver, &args)?;

    Ok(())
}

fn run_margin(
    dir: Option<&str>,
    device: &PcieDevice,
    lanes: Vec<Lane>,
    receiver: Receiver,
    args: &Args,
) -> Result<()> {
    let duration = Duration::from_secs_f64(args.duration.max(0.200));
    if args.verbose >= verbosity::MARGIN_DURATION {
        println!("lmar: margin duration {:?}", duration);
    }

    // Each lane will be margined in a thread, and will send updates to the main
    // thread for (1) writing to a file and (2) printing progress.
    let (tx, rx) = mpsc::channel();

    // Create the multiprogress bar for printing the progress of each lane and
    // dimension, if the verbosity level calls for it.
    let progress = if args.verbose == verbosity::PROGRESS_SUMMARY {
        Some(MultiProgress::with_draw_target(ProgressDrawTarget::hidden()))
    } else {
        None
    };

    // Keep track of the state for each lane's margining protocol.
    let mut state = BTreeMap::new();
    let mut printed_caps = false;
    let mut margining_limits = None;
    for lane in lanes.iter().copied() {
        let device_ = device.try_clone()?;
        let tx_ = tx.clone();

        // Construct object for running the margining protocol.
        let margin = LaneMargin::new(device_, receiver, lane, args.verbose)
            .context("Could not initialize lane margining")?;

        // All margining threads will send us this report of
        // capabilities. Let's only print one of them.
        let limits = margin.limits();
        if !printed_caps
            && (args.verbose >= verbosity::CAPABILITIES || args.report_only)
        {
            let capabilities = margin.capabilities();
            println!("lmar: {capabilities:#?}");
            println!("lmar: {limits:#?}");
            printed_caps = true;
        }
        margining_limits.replace(limits.clone());

        // If we're just reporting the capabilities and limits of the device, we
        // do not need to do anything else at all. We've printed them, and
        // they're the same for all lanes. Exit successfully.
        if args.report_only {
            return Ok(());
        }

        // Spawn a thread for actually running the protocol.
        let (file, filename) = open_margin_results_file(dir, device, &lane)?;
        let error_count = args.error_count;
        let timing = args.timing;
        let voltage = args.voltage;
        let thr = thread::spawn(move || {
            margin_lane(margin, duration, error_count, timing, voltage, tx_)
        });
        if args.verbose >= verbosity::FILENAME {
            println!("lmar: saving lane {} to \"{}\"", lane, filename);
        }

        // Insert a small delay between each lane. This is not strictly
        // necessary, but may help improve throughput a bit.
        thread::sleep(duration / lanes.len() as u32);

        // Set up the progress bars for this lane, if the verbosity level
        // requires it.
        let bars = if args.verbose == verbosity::PROGRESS_SUMMARY {
            let bars =
                ProgressBars::new(&lane, margining_limits.as_ref().unwrap());
            let mp = progress.as_ref().expect("No progress bars!");
            mp.add(bars.time.clone());
            if let Some(vb) = bars.voltage.as_ref() {
                mp.add(vb.clone());
            }
            Some(bars)
        } else {
            None
        };

        // Store the state for this lane's margin worker.
        state.insert(lane, MarginState { thr, file, n_points: (0, 0), bars });
    }

    // Unhide the progress bars, and tick them all manually so that they draw to
    // the screen now, rather than when we update them. That's because the
    // voltage progress bars won't appear until we start margining in that
    // direction otherwise.
    if let Some(p) = &progress {
        p.set_draw_target(ProgressDrawTarget::stdout());
        for st in state.values() {
            if let Some(bars) = &st.bars {
                bars.time.tick();
                if let Some(vb) = &bars.voltage {
                    vb.tick();
                }
            }
        }
    }

    // Close our own sender, so that this call returns an error when all
    // per-thread senders are closed.
    drop(tx);
    while let Ok(update) = rx.recv() {
        let st =
            state.get_mut(&update.lane).expect("No margining state for lane");
        let limits = margining_limits
            .as_ref()
            .expect("Limits should be reported before margining starts");

        // Update the number of points we've margined for this lane, and
        // get the total we expect based on the reported capabilities.
        let (n_points, n_total_points) =
            if matches!(update.point, MarginPoint::Time(_)) {
                st.n_points.0 += 1;
                (st.n_points.0, limits.num_timing_steps * 2)
            } else {
                st.n_points.1 += 1;
                (
                    st.n_points.1,
                    limits.num_voltage_steps.expect(
                        "Received margin update for voltage for a \
                    device that doesn't appear to support \
                    voltage margining",
                    ) * 2,
                )
            };

        // Write the result to the output file.
        let (pass, count) = match update.result {
            MarginResult::Success(count) => (1, u8::from(count)),
            MarginResult::Failed(count) => (0, u8::from(count)),
        };
        writeln!(
            st.file,
            "{:0.3}\t{:0.3}\t{:0.9}\t{}\t{}",
            update.point.time(),
            update.point.voltage(),
            update.duration.as_secs_f64(),
            count,
            pass,
        )
        .unwrap();

        // Print the progress to the screen, if needed.
        if args.verbose > verbosity::PROGRESS_SUMMARY {
            println!("lmar: margined point {n_points} / {n_total_points}: {update:?}");
        } else if args.verbose == verbosity::PROGRESS_SUMMARY {
            let st = state.get_mut(&update.lane).unwrap();
            let bars = st.bars.as_ref().expect("No progress bars!");
            if matches!(update.point, MarginPoint::Time(_)) {
                bars.time.inc(1);
            } else {
                let voltage_bar = bars.voltage.as_ref().expect(
                    "Received voltage update for a \
                        device that doesn't appear to \
                        support voltage margining",
                );
                voltage_bar.inc(1);
            }
        }
    }

    // Finish the progress bars, leaving them on the screen.
    for st in state.values() {
        if let Some(bars) = &st.bars {
            bars.time.finish();
            if let Some(vb) = &bars.voltage {
                vb.finish();
            }
        }
    }

    // Print any error messages the threads hit.
    for (lane, state) in state.into_iter() {
        if let Err(e) = state.thr.join() {
            eprintln!("lmar: margining lane {lane} failed: {e:?}");
        }
    }

    Ok(())
}

// The state maintained for running the margining protocol in a thread.
#[derive(Debug)]
struct MarginState {
    // Handle to the thread running the protocol.
    thr: thread::JoinHandle<anyhow::Result<()>>,
    // File to which results are saved.
    file: File,
    // The number of points margined, for voltage and time.
    n_points: (u8, u8),
    // The progress bars, if the summary verbosity level was chosen.
    bars: Option<ProgressBars>,
}

// The progress bars for a single lane.
#[derive(Debug)]
struct ProgressBars {
    time: ProgressBar,
    voltage: Option<ProgressBar>,
}

impl ProgressBars {
    /// Create the progress bars for a lane, from the margining limits.
    pub fn new(lane: &Lane, limits: &MarginingLimits) -> Self {
        let time = ProgressBar::with_draw_target(
            Some(u64::from(limits.num_timing_steps) * 2),
            ProgressDrawTarget::hidden(),
        );
        time.set_style(
            ProgressStyle::with_template("{prefix:24}: {bar} {pos} / {len}")
                .unwrap(),
        );
        time.set_prefix(format!("lmar: lane {lane} (time)"));
        time.tick();
        let voltage = limits.num_voltage_steps.map(|n_steps| {
            let voltage = ProgressBar::with_draw_target(
                Some(u64::from(n_steps) * 2),
                ProgressDrawTarget::hidden(),
            );
            voltage.set_style(
                ProgressStyle::with_template(
                    "{prefix:24}: {bar} {pos} / {len}",
                )
                .unwrap(),
            );
            voltage.set_prefix(format!("lmar: lane {lane} (voltage)"));
            voltage.tick();
            voltage
        });
        Self { time, voltage }
    }
}

// An update from a thread about margining a single point.
#[derive(Debug)]
struct MarginUpdate {
    // The lane being margined.
    lane: Lane,
    // The duration of the margin process.
    duration: Duration,
    // The point at which the lane was margined, in either voltage or time.
    point: MarginPoint,
    // The result of the margining process.
    result: MarginResult,
}

fn margin_lane(
    margin: LaneMargin,
    duration: Duration,
    error_count: Option<u8>,
    timing: bool,
    voltage: bool,
    tx: mpsc::Sender<MarginUpdate>,
) -> anyhow::Result<()> {
    let lane = margin.lane();
    let capabilities = margin.capabilities();
    let limits = margin.limits();
    if let Some(_count) = error_count {
        /*
        margin.set_error_count_limit(4).unwrap();
        */
    }

    // Compute the resolution in both dimensions.
    let timing_resolution: f64 = f64::from(limits.max_timing_offset)
        / f64::from(limits.num_timing_steps);
    let voltage_resolution: f64 = if capabilities.voltage_supported {
        (f64::from(limits.max_voltage_offset.unwrap())
            / f64::from(limits.num_voltage_steps.unwrap()))
            / 100.0
    } else {
        0.0
    };

    // Iterate over the timing steps from left to right.
    if timing {
        let steps = margin.iter_left_right_steps();
        for step in steps.into_iter() {
            // Set up per the spec for margining a single point.
            margin.clear_error_log()?;
            margin.go_to_normal_settings()?;
            margin.no_command()?;

            // Compute the actual time as a percentage of UI that we're currently
            // margining.
            let sign = if matches!(step.direction, Some(LeftRight::Left)) {
                -1.0
            } else {
                1.0
            };
            let point = MarginPoint::Time(
                sign * timing_resolution * f64::from(step.steps.0),
            );
            let (margin_duration, result) = margin
                .margin_at_left_right(step, duration)
                .context(format!("Failed to margin point: {step:?}"))?;
            tx.send(MarginUpdate {
                lane,
                point,
                duration: margin_duration,
                result,
            })?;
        }
    }

    // Iterate over the voltage steps, if supported.
    if voltage {
        let steps = margin.iter_up_down_steps();
        for step in steps.into_iter() {
            // Set up per the spec for margining a single point.
            margin.clear_error_log()?;
            margin.go_to_normal_settings()?;
            margin.no_command()?;

            // Compute the actual voltage at which we're margining.
            let sign = if matches!(step.direction, Some(UpDown::Down)) {
                -1.0
            } else {
                1.0
            };
            let point = MarginPoint::Voltage(
                sign * voltage_resolution * f64::from(step.steps.0),
            );
            let (margin_duration, result) = margin
                .margin_at_up_down(step, duration)
                .context(format!("Failed to margin point: {step:?}"))?;
            tx.send(MarginUpdate {
                lane,
                point,
                duration: margin_duration,
                result,
            })?;
        }
    }

    Ok(())
}
