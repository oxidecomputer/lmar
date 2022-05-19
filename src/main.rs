// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! PCIe lane margining at the receiver prototype.

use clap::ArgEnum;
use clap::Parser;
use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fmt;
use std::fs::File;
use std::io;
use std::io::Write;
use std::num::NonZeroU8;
use std::os::unix::io::AsRawFd;
use std::thread::sleep;
use std::time::Duration;
use std::time::Instant;
use thiserror::Error;

#[derive(Copy, Clone, Debug, ArgEnum)]
enum Port {
    Upstream,
    Downstream,
}

#[derive(Parser, Debug)]
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
    bdf: Bdf,

    /// Which port to use on the chosen PCIe endpoint.
    ///
    /// For example, for the root component this should always be downstream.
    /// For an endpoint like a drive, this should be upstream.
    #[clap(arg_enum)]
    port: Port,

    /// Output file name, for saving the margining data.
    ///
    /// Default is a name based on the bus/device/function and timestamp, to
    /// avoid clobbering past results.
    #[clap(short, long)]
    output: Option<String>,

    /// The time to spend margining each point, in seconds.
    #[clap(short, long, default_value_t = 1.0)]
    duration: f64,

    /// The lane on the endpoint to be margined.
    #[clap(short, long, default_value_t = 0)]
    lane: u8,

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
}

/// Errors working with a PCIe device
#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] io::Error),

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

    #[error("Failed to decode response for command {command:?}, reason {reason}")]
    DecodeFailed {
        command: MarginCommand,
        reason: String,
    },

    #[error("Capability or feature not implemented yet: {0}")]
    Unimplemented(String),
}

// Ioctl command definitions
const PCITOOL_IOC: i32 = (('P' as i32) << 24) | (('C' as i32) << 16) | (('T' as i32) << 8);
const PCITOOL_DEVICE_GET_REG: i32 = PCITOOL_IOC | 1;
const PCITOOL_DEVICE_SET_REG: i32 = PCITOOL_IOC | 2;
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

impl ExtendedCapability {
    /// Return the size of the capability data blob associated with this cap ID.
    ///
    /// We're pretty heavily targeting the Lane Margining Cap, so everything
    /// else returns None at this point.
    pub fn data_size(&self) -> Option<usize> {
        match self.id {
            ExtendedCapabilityId::LaneMargining => Some(0x88),
            _ => None,
        }
    }
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
        Ok(Self {
            speed,
            width,
            training,
            active,
        })
    }
}

#[derive(Debug, Clone, Copy)]
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
        let port_capabilities = MarginingPortCapabilities::from(buf.get_u16_le());
        let port_status = MarginingPortStatus::from(buf.get_u16_le());

        // Each lane has a control + status register
        let n_lanes = buf.len() / (2 * std::mem::size_of::<u16>());
        let mut lanes = Vec::with_capacity(n_lanes);
        for _ in 0..n_lanes {
            let control = MarginingLaneControl::from(buf.get_u16_le());
            let status = MarginingLaneStatus::decode_for_cmd(control.cmd, buf.get_u16_le())?;
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
        Self {
            uses_driver_software: (word & 0x1) != 0,
        }
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
        Self {
            ready,
            software_ready,
        }
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
        let bits = match self {
            MarginCommand::NoCommand => 0b111,
            MarginCommand::Report(_) => 0b001,
            MarginCommand::SetErrorCountLimit(_)
            | MarginCommand::GoToNormalSettings
            | MarginCommand::ClearErrorLog => 0b010,
            MarginCommand::StepLeftRight(_) => 0b011,
            MarginCommand::StepUpDown(_) => 0b100,
        };
        bits << 3
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
            (MarginCommand::SetErrorCountLimit(_), MarginResponse::ErrorCountLimit(_)) => true,
            (MarginCommand::GoToNormalSettings, MarginResponse::GoToNormalSettings) => true,
            (MarginCommand::ClearErrorLog, MarginResponse::ClearErrorLog) => true,
            (MarginCommand::StepLeftRight(_), MarginResponse::StepExecutionStatus { .. }) => true,
            (MarginCommand::StepUpDown(_), MarginResponse::StepExecutionStatus { .. }) => true,
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
                _ => MarginCommand::SetErrorCountLimit(ErrorCount::from(payload)),
            },
            0b011 => MarginCommand::StepLeftRight(StepLeftRight::from(payload)),
            0b100 => MarginCommand::StepLeftRight(StepLeftRight::from(payload)),
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
            steps: Steps::from(word),
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
            Some(dir) => u8::from(dir) << 6,
        };
        bits | u8::from(step.steps)
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

#[derive(Debug, Clone, Copy)]
pub struct Steps(u8);

impl From<u8> for Steps {
    fn from(word: u8) -> Steps {
        Self(word & 0b1111111)
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
    pub fn decode_for_cmd(cmd: MarginCommand, word: u16) -> Result<Self, Error> {
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
                        reason: format!("Expected payload 0x9C, found: {:#x}", payload),
                    });
                }
                if receiver != 0 {
                    return Err(Error::DecodeFailed {
                        command: cmd,
                        reason: format!("Expected receiver 0x0, found: {:#x}", receiver),
                    });
                }
                Ok(MarginResponse::NoCommand)
            }
            0b001 => {
                if let MarginCommand::Report(report) = cmd {
                    let report_response = match report {
                        ReportRequest::Capabilities => {
                            ReportResponse::Capabilities(ReportCapabilities::from(payload))
                        }
                        ReportRequest::NumVoltageSteps => ReportResponse::NumVoltageSteps(payload),
                        ReportRequest::NumTimingSteps => ReportResponse::NumTimingSteps(payload),
                        ReportRequest::MaxTimingOffset => ReportResponse::MaxTimingOffset(payload),
                        ReportRequest::MaxVoltageOffset => {
                            ReportResponse::MaxVoltageOffset(payload)
                        }
                        ReportRequest::SamplingRateVoltage => {
                            ReportResponse::SamplingRateVoltage(payload)
                        }
                        ReportRequest::SamplingRateTiming => {
                            ReportResponse::SamplingRateTiming(payload)
                        }
                        ReportRequest::SampleCount => ReportResponse::SampleCount(payload),
                        ReportRequest::MaxLanes => ReportResponse::MaxLanes(payload),
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
                        Ok(MarginResponse::ErrorCountLimit(ErrorCount::from(payload)))
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
    pub fn decode_for_cmd(cmd: MarginCommand, word: u16) -> Result<Self, Error> {
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
}

impl LaneMarginInner {
    fn report(&self, request: ReportRequest) -> Result<ReportResponse, Error> {
        let cmd = MarginCommand::Report(request);
        let word = self.build_command(cmd);
        write_configuration_space(&self.device.file, &self.device.bdf, self.cmd_offset, word)?;
        match self.wait_for_response(cmd)? {
            MarginResponse::Report(report) => Ok(report),
            other => Err(Error::Margin(format!(
                "Expected response for {:?}, found: {:?}",
                request, other
            ))),
        }
    }

    fn report_capabilities(&self) -> Result<ReportCapabilities, Error> {
        if let ReportResponse::Capabilities(caps) = self.report(ReportRequest::Capabilities)? {
            Ok(caps)
        } else {
            Err(Error::Margin(format!(
                "Expected ReportResponse::ReportCapabilities"
            )))
        }
    }

    fn no_command(&self) -> Result<(), Error> {
        let cmd = MarginCommand::NoCommand;
        let word = self.build_command(cmd);
        write_configuration_space(&self.device.file, &self.device.bdf, self.cmd_offset, word)?;
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
        let word = self.build_command(cmd);
        write_configuration_space(&self.device.file, &self.device.bdf, self.cmd_offset, word)?;
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
        let word = self.build_command(cmd);
        write_configuration_space(&self.device.file, &self.device.bdf, self.cmd_offset, word)?;
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
        let words = [margin_type | receiver, payload];
        u16::from_le_bytes(words)
    }

    fn wait_for_response(&self, cmd: MarginCommand) -> Result<MarginResponse, Error> {
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
                "for command {:?} within time limit of {:?}: {:?}",
            ),
            cmd, MAX_WAIT_TIME, response
        )))
    }

    fn read_command_response(&self, cmd: MarginCommand) -> Result<MarginResponse, Error> {
        let word = read_configuration_space(&self.device.file, &self.device.bdf, self.sts_offset)?;
        MarginResponse::decode_for_cmd(cmd, word)
    }

    fn gather_limits(&self) -> Result<MarginingLimits, Error> {
        let num_voltage_steps = if let ReportResponse::NumVoltageSteps(steps) =
            self.report(ReportRequest::NumVoltageSteps)?
        {
            steps
        } else {
            101
        };
        self.no_command()?;
        let num_timing_steps = if let ReportResponse::NumTimingSteps(steps) =
            self.report(ReportRequest::NumTimingSteps)?
        {
            steps
        } else {
            101
        };
        self.no_command()?;
        let max_timing_offset = if let ReportResponse::MaxTimingOffset(steps) =
            self.report(ReportRequest::MaxTimingOffset)?
        {
            steps
        } else {
            101
        };
        self.no_command()?;
        let max_voltage_offset = if let ReportResponse::MaxVoltageOffset(steps) =
            self.report(ReportRequest::MaxVoltageOffset)?
        {
            steps
        } else {
            101
        };
        self.no_command()?;
        let sampling_rate_voltage = if let ReportResponse::SamplingRateVoltage(steps) =
            self.report(ReportRequest::SamplingRateVoltage)?
        {
            steps
        } else {
            101
        };
        self.no_command()?;
        let sampling_rate_timing = if let ReportResponse::SamplingRateTiming(steps) =
            self.report(ReportRequest::SamplingRateTiming)?
        {
            steps
        } else {
            101
        };
        self.no_command()?;
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
        let word = self.build_command(cmd);
        write_configuration_space(&self.device.file, &self.device.bdf, self.cmd_offset, word)?;
        let response = self.wait_for_response(cmd)?;
        match response {
            MarginResponse::ErrorCountLimit(actual_limit) => {
                if limit == actual_limit {
                    Ok(())
                } else {
                    Err(Error::Margin(format!("Failed to set error count limit")))
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
        let word = self.build_command(cmd);
        write_configuration_space(&self.device.file, &self.device.bdf, self.cmd_offset, word)?;

        // Interval between checks when execution status is "setup"
        const INTERVAL: Duration = Duration::from_millis(1);

        // Total duration before setup must complete
        const TOTAL_DURATION: Duration = Duration::from_millis(200);

        let now = Instant::now();
        loop {
            let response = self.wait_for_response(cmd)?;
            match response {
                MarginResponse::StepExecutionStatus {
                    status,
                    error_count,
                } => match status {
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
                },
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
    pub num_voltage_steps: u8,
    pub num_timing_steps: u8,
    pub max_timing_offset: u8,
    pub max_voltage_offset: u8,
    pub sampling_rate_voltage: u8,
    pub sampling_rate_timing: u8,
}

#[derive(Debug)]
pub struct LaneMargin {
    inner: LaneMarginInner,
    capabilities: ReportCapabilities,
    limits: MarginingLimits,
}

impl LaneMargin {
    pub fn new(device: PcieDevice, receiver: Receiver, lane: Lane) -> Result<Self, Error> {
        // Find the size of the capability header, limited to the width of the
        // link.
        let link_status = device.link_status()?;
        let n_lanes = usize::from(u8::from(link_status.width));
        // 2 u16s for each lane (control / status) + 2 for the port capability / status
        let n_bytes = (n_lanes + 1) * std::mem::size_of::<u16>() * 2;
        let data =
            device.read_extended_capability_data::<u8>(&ExtendedCapabilityId::LaneMargining)?;
        let header = LaneMarginingCapabilityHeader::try_from(&data[..n_bytes])?;
        if !header.port_status.ready {
            return Err(Error::Margin(format!(
                "Margining appears unsupported on this device"
            )));
        }
        let header_offset = device
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
        };
        let capabilities = inner.report_capabilities()?;
        inner.no_command()?;
        let limits = inner.gather_limits()?;
        Ok(Self {
            inner,
            capabilities,
            limits,
        })
    }

    /// Consume the lane margin controller, and return the device. This is
    /// useful for running the margining protocol on a new lane or receiver.
    pub fn device(self) -> PcieDevice {
        self.inner.device
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

    pub fn iter_left_right_steps(&self) -> (usize, Box<dyn Iterator<Item = StepLeftRight>>) {
        let steps = self.limits().num_timing_steps;
        let base = 1..=steps;
        let right = base.clone().map(|pt| StepLeftRight {
            direction: Some(LeftRight::Right),
            steps: Steps::from(pt),
        });
        if self.capabilities().independent_left_right_sampling {
            (
                base.len() * 2,
                Box::new(
                    base.rev()
                        .map(|pt| StepLeftRight {
                            direction: Some(LeftRight::Left),
                            steps: Steps::from(pt),
                        })
                        .chain(right),
                ),
            )
        } else {
            (base.len(), Box::new(right.into_iter()))
        }
    }

    pub fn iter_up_down_steps(&self) -> (usize, Box<dyn Iterator<Item = StepUpDown>>) {
        let steps = self.limits().num_voltage_steps;
        let base = 1..=steps;
        let up = base.clone().map(|pt| StepUpDown {
            direction: Some(UpDown::Up),
            steps: Steps::from(pt),
        });
        if self.capabilities().independent_up_down_voltage {
            (
                base.len() * 2,
                Box::new(
                    base.rev()
                        .map(|pt| StepUpDown {
                            direction: Some(UpDown::Down),
                            steps: Steps::from(pt),
                        })
                        .chain(up),
                ),
            )
        } else {
            (base.len(), Box::new(up.into_iter()))
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
            let bus = parts[0]
                .parse()
                .map_err(|_| Error::InvalidBdf(s.to_string()))?;
            let device = parts[1]
                .parse()
                .map_err(|_| Error::InvalidBdf(s.to_string()))?;
            let function = parts[2]
                .parse()
                .map_err(|_| Error::InvalidBdf(s.to_string()))?;
            Ok(Bdf {
                bus,
                device,
                function,
            })
        } else {
            Err(Error::InvalidBdf(s.to_string()))
        }
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
pub trait DataWord: Sized + Copy + std::fmt::LowerHex {
    fn size_attribute() -> u32;
    fn from_raw(_: u64) -> Self;
    fn to_raw(self) -> u64;
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
}

/// A handle to a PCIe device on the system.
#[derive(Debug)]
pub struct PcieDevice {
    pub vendor_id: u16,
    pub device_id: u16,
    bdf: Bdf,
    capabilities: BTreeMap<usize, CapabilityId>,
    extended_capabilities: BTreeMap<usize, ExtendedCapability>,
    file: File,
}

impl PcieDevice {
    /// Construct a PCIe device to operate on a single bus/device/function.
    pub fn new(bdf: Bdf) -> Result<Self, Error> {
        let file = File::options()
            .read(true)
            .write(true)
            .open(PCI_REG_DEVICE_FILE)?;
        let vendor_id = read_configuration_space::<u16>(&file, &bdf, PCIE_VENDOR_ID_OFFSET)?;
        let device_id = read_configuration_space::<u16>(&file, &bdf, PCIE_DEVICE_ID_OFFSET)?;
        let capabilities = Self::read_capabilities(&file, &bdf)?;
        let extended_capabilities = Self::read_extended_capabilities(&file, &bdf)?;
        Ok(Self {
            vendor_id,
            device_id,
            bdf,
            capabilities,
            extended_capabilities,
            file,
        })
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
    pub fn supports_extended_capability(&self, id: &ExtendedCapabilityId) -> bool {
        self.find_extended_capability(id).is_some()
    }

    // Return the register offset and capability ID of the requested
    // capability, or `None` if it's not supported by the device
    fn find_capability(&self, cap: &CapabilityId) -> Option<(&usize, &CapabilityId)> {
        self.capabilities
            .iter()
            .find(|(_, capability)| capability == &cap)
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
    fn read_capabilities(file: &File, bdf: &Bdf) -> Result<BTreeMap<usize, CapabilityId>, Error> {
        let mut cap_start =
            usize::from(read_configuration_space::<u8>(file, bdf, PCIE_CAP_POINTER)?);
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

    // Read the data associated with the requested capability, if it exists
    pub fn read_extended_capability_data<T>(
        &self,
        id: &ExtendedCapabilityId,
    ) -> Result<Vec<T>, Error>
    where
        T: DataWord,
    {
        let (&cap_start, cap) = self
            .find_extended_capability(id)
            .ok_or_else(|| Error::UnsupportedExtendedCap(*id))?;
        let size = cap.data_size().ok_or_else(|| {
            Error::Unimplemented(format!("Extended capability parsing for cap: {:?}", id))
        })?;
        let data_start = cap_start + std::mem::size_of::<u32>(); // Skip the capability header itself
        let data_end = cap_start + size;
        let mut data = Vec::with_capacity(data_end - data_start);
        for offset in data_start..data_end {
            data.push(read_configuration_space::<T>(
                &self.file, &self.bdf, offset,
            )?);
        }
        Ok(data)
    }
}

/*
struct LinkMargin {
    device: PcieDevice,
    limits: MarginLimits,
}
*/

// Should always be able to issue no command
// Should always have the ability to wait for a command success/failure
// Should always be able to check if margining is supported

/// State when we've verified that the link supports margining and is in the
/// correct power state, the link is up, and the speed is supported.
///
/// From here we can go to an error or LimitsReported.
//trait Initialized;

// Here we can report limits

/// State after we've reported the voltage/timing limits, the number of steps in
/// each (if they're both supported), the sampling rate in voltage/timing steps.
/// This basically means we know the grid of the margining process.
///
/// From here we can go to an error or take margin steps.
//trait Limits;

// Here we can take a step in either direction, verified to be within the limits
// as previously learned.
//
// Not sure the state business is worth it. We should just check that margining
// is supported and that we know the limits in the constructor. After that
// we just take some steps, get the result, check if we've failed, etc.

fn read_configuration_space<W>(file: &File, bdf: &Bdf, offset: usize) -> Result<W, Error>
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
        Ok(W::from_raw(register.data))
    } else {
        Err(io::Error::last_os_error().into())
    }
}

fn write_configuration_space<W>(file: &File, bdf: &Bdf, offset: usize, word: W) -> Result<(), Error>
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

#[derive(Debug, Clone, Copy)]
pub enum MarginResult {
    Success(ErrorCount),
    Failed(ErrorCount),
}

fn main() {
    let args = Args::parse();

    let device = PcieDevice::new(args.bdf).expect("Failed to create PCIe device");
    let link_status = device.link_status().expect("Failed to get link status");

    if args.verbose > 1 {
        println!("{:#x?}", link_status);
    }

    let receiver = match args.port {
        Port::Upstream => Receiver::upstream(),
        Port::Downstream => Receiver::downstream(),
    };

    let lane = Lane::new(args.lane).expect("Invalid lane");
    assert!(
        link_status.valid_lane(lane),
        "Lane {} is invalid for a device with link width {}",
        lane,
        link_status.width,
    );

    let margin =
        LaneMargin::new(device, receiver, lane).expect("Could not initialize lane margining");
    let caps = margin.capabilities();
    let limits = margin.limits();

    if args.verbose > 1 {
        println!("{:#?}", caps);
        println!("{:#?}", limits);
    }

    if let Some(_count) = args.error_count {
        /*
        margin.set_error_count_limit(4).unwrap();
        */
    }

    let duration = Duration::from_secs_f64(args.duration.max(0.200));
    if args.verbose > 1 {
        println!("Margin duration {:?}", duration);
    }

    let filename = args.output.unwrap_or_else(|| {
        format!(
            "margin-results-b{}-d{}-f{}-{}.txt",
            args.bdf.bus,
            args.bdf.device,
            args.bdf.function,
            chrono::Utc::now().format("%Y-%m-%dT%H-%M-%S"),
        )
    });
    let mut outfile = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&filename)
        .expect("Failed to create output file");
    if args.verbose > 0 {
        println!("Writing results to: \"{}\"", filename);
    }

    let mut left_right: Vec<(StepLeftRight, Duration, MarginResult)> = vec![];
    let (n_steps, iter) = margin.iter_left_right_steps();
    for (i, step) in iter.enumerate() {
        margin.clear_error_log().unwrap();
        margin.go_to_normal_settings().unwrap();
        margin.no_command().unwrap();
        let (margin_duration, result) = margin.margin_at_left_right(step, duration).unwrap();
        left_right.push((step, margin_duration, result));
        if args.verbose > 1 {
            println!(
                "{}/{n_steps}: {step:?}, Duration: {duration:?}, {result:?}",
                i + 1
            );
        } else if args.verbose == 1 {
            if i < n_steps - 1 {
                print!("\rMargining time: {}/{n_steps}", i + 1);
                std::io::stdout().flush().unwrap();
            } else {
                println!("\rMargining time: {}/{n_steps}", i + 1);
            }
        }
    }
    if args.verbose == 1 {
        println!("");
    }

    let mut up_down: Vec<(StepUpDown, Duration, MarginResult)> = vec![];
    let (n_steps, iter) = margin.iter_up_down_steps();
    for (i, step) in iter.enumerate() {
        margin.clear_error_log().unwrap();
        margin.go_to_normal_settings().unwrap();
        margin.no_command().unwrap();

        let (margin_duration, result) = margin.margin_at_up_down(step, duration).unwrap();
        up_down.push((step, margin_duration, result));
        if args.verbose > 1 {
            println!(
                "{}/{n_steps}: {step:?}, Duration: {duration:?}, {result:?}",
                i + 1
            );
        } else if args.verbose > 0 {
            if i < n_steps - 1 {
                print!("\rMargining voltage: {}/{n_steps}", i + 1);
                std::io::stdout().flush().unwrap();
            } else {
                println!("\rMargining voltage: {}/{n_steps}", i + 1);
            }
        }
    }

    let timing_resolution: f64 =
        f64::from(limits.max_timing_offset) / f64::from(limits.num_timing_steps);
    let voltage_resolution: f64 =
        (f64::from(limits.max_voltage_offset) / f64::from(limits.num_voltage_steps)) / 100.0;

    writeln!(
        outfile,
        "Time (%UI)\tVoltage (V)\tDuration (s)\tCount\tPass"
    )
    .unwrap();
    for (step, margin_duration, result) in left_right.iter() {
        let sign = if matches!(step.direction, Some(LeftRight::Left)) {
            -1.0
        } else {
            1.0
        };
        let (pass, count) = match result {
            MarginResult::Success(count) => (1, u8::from(*count)),
            MarginResult::Failed(count) => (0, u8::from(*count)),
        };
        writeln!(
            outfile,
            "{:0.3}\t0.00\t{}\t{:0.9}\t{}",
            sign * timing_resolution * f64::from(u8::from(step.steps)),
            margin_duration.as_secs_f64(),
            count,
            pass,
        )
        .unwrap();
    }

    for (step, margin_duration, result) in up_down.iter() {
        let sign = if matches!(step.direction, Some(UpDown::Down)) {
            -1.0
        } else {
            1.0
        };
        let (pass, count) = match result {
            MarginResult::Success(count) => (1, u8::from(*count)),
            MarginResult::Failed(count) => (0, u8::from(*count)),
        };
        writeln!(
            outfile,
            "0.00\t{:0.3}\t{}\t{:0.9}\t{}",
            sign * voltage_resolution * f64::from(u8::from(step.steps)),
            margin_duration.as_secs_f64(),
            count,
            pass,
        )
        .unwrap();
    }
}
