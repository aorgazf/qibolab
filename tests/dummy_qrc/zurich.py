import itertools
import pathlib

import laboneq.simple as lo
import numpy as np
import pytest

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.channels import ChannelMap
from qibolab.instruments.dummy_oscillator import DummyLocalOscillator as LocalOscillator
from qibolab.instruments.zhinst import ZhPulse, ZhSweeperLine, Zurich
from qibolab.platform import Platform
from qibolab.pulses import (
    Drag,
    FluxPulse,
    Gaussian,
    Pulse,
    PulseSequence,
    ReadoutPulse,
    Rectangular,
)
from qibolab.sweeper import Parameter, Sweeper

RUNCARD = pathlib.Path(__file__).parent / "zurich.yml"


# Function returning a calibrated device setup
def create_offline_device_setup():
    """
    Function returning a device setup
    """

    # Instantiate Zh set of instruments[They work as one]
    instruments = {
        "SHFQC": [{"address": "DEV12146", "uid": "device_shfqc"}],
        "HDAWG": [
            {"address": "DEV8660", "uid": "device_hdawg"},
            {"address": "DEV8673", "uid": "device_hdawg2"},
        ],
        "PQSC": [{"address": "DEV10055", "uid": "device_pqsc"}],
    }

    shfqc = []
    for i in range(5):
        shfqc.append({"iq_signal": f"q{i}/drive_line", "ports": f"SGCHANNELS/{i}/OUTPUT"})
        shfqc.append({"iq_signal": f"q{i}/measure_line", "ports": ["QACHANNELS/0/OUTPUT"]})
        shfqc.append({"acquire_signal": f"q{i}/acquire_line", "ports": ["QACHANNELS/0/INPUT"]})

    hdawg = []
    for i in range(5):
        hdawg.append({"rf_signal": f"q{i}/flux_line", "ports": f"SIGOUTS/{i}"})
    for c, i in zip(itertools.chain(range(0, 2), range(3, 4)), range(5, 8)):
        hdawg.append({"rf_signal": f"qc{c}/flux_line", "ports": f"SIGOUTS/{i}"})

    hdawg2 = [{"rf_signal": "qc4/flux_line", "ports": f"SIGOUTS/0"}]

    pqsc = [
        "internal_clock_signal",
        {"to": "device_hdawg2", "port": "ZSYNCS/4"},
        {"to": "device_hdawg", "port": "ZSYNCS/2"},
        {"to": "device_shfqc", "port": "ZSYNCS/0"},
    ]

    connections = {
        "device_shfqc": shfqc,
        "device_hdawg": hdawg,
        "device_hdawg2": hdawg2,
        "device_pqsc": pqsc,
    }

    descriptor = {
        "instruments": instruments,
        "connections": connections,
    }

    device_setup = lo.DeviceSetup.from_dict(
        descriptor,
        server_host="my_ip_address",
        server_port="8004",
        setup_name="test_setup",
    )

    return device_setup, descriptor


def create(runcard=RUNCARD):
    """Create platform using Zurich Instrumetns (Zh) SHFQC, HDAWGs and PQSC.

    Instrument related parameters are hardcoded in ``__init__`` and ``setup``.

    Args:
        runcard (str): Path to the runcard file.
    """
    # Create channel objects
    channels = ChannelMap()
    # readout
    channels |= "L3-31"
    # feedback
    channels |= "L2-7"
    # drive
    channels |= (f"L4-{i}" for i in range(15, 20))
    # flux qubits
    channels |= (f"L4-{i}" for i in range(6, 11))
    # flux couplers
    channels |= (f"L4-{i}" for i in range(11, 15))

    # Map controllers to qubit channels
    # feedback
    channels["L3-31"].ports = [("device_shfqc", "[QACHANNELS/0/INPUT]")]
    channels["L3-31"].power_range = 10
    # readout
    channels["L2-7"].ports = [("device_shfqc", "[QACHANNELS/0/OUTPUT]")]
    channels["L2-7"].power_range = -25  # -5 for punchout
    # drive
    for i in range(5, 10):
        channels[f"L4-1{i}"].ports = [("device_shfqc", f"SGCHANNELS/{i-5}/OUTPUT")]
        channels[f"L4-1{i}"].power_range = -10

    # flux qubits (CAREFUL WITH THIS !!!)
    for i in range(6, 11):
        channels[f"L4-{i}"].ports = [("device_hdawg", f"SIGOUTS/{i-6}")]
        # channels[f"L4-{i}"].power_range = 0 #This may not be the default value find it

    # flux couplers (CAREFUL WITH THIS !!!)
    for i in range(11, 14):
        channels[f"L4-{i}"].ports = [("device_hdawg", f"SIGOUTS/{i-11+5}")]
        # channels[f"L4-{i}"].power_range = 0 #This may not be the default value find it

    channels[f"L4-14"].ports = [("device_hdawg2", f"SIGOUTS/0")]
    # channels["L4-14"].power_range = 0 #This may not be the default value find it

    device_setup, descriptor = create_offline_device_setup()

    controller = Zurich("EL_ZURO", descriptor, use_emulation=False)

    # set time of flight for readout integration (HARDCODED)
    controller.time_of_flight = 280
    controller.smearing = 100

    # Instantiate local oscillators
    local_oscillators = [LocalOscillator(f"lo_{kind}", None) for kind in ["readout"] + [f"drive_{n}" for n in range(4)]]

    # Set Dummy LO parameters (Map only the two by two oscillators)
    local_oscillators[0].frequency = 5_500_000_000  # For SG0 (Readout)
    local_oscillators[1].frequency = 4_200_000_000  # For SG1 and SG2 (Drive)
    local_oscillators[2].frequency = 4_600_000_000  # For SG3 and SG4 (Drive)
    local_oscillators[3].frequency = 4_800_000_000  # For SG5 and SG6 (Drive)

    # Map LOs to channels
    ch_to_lo = {"L2-7": 0, "L4-15": 1, "L4-16": 1, "L4-17": 2, "L4-18": 2, "L4-19": 3}
    for ch, lo in ch_to_lo.items():
        channels[ch].local_oscillator = local_oscillators[lo]

    instruments = [controller] + local_oscillators
    platform = Platform("IQM5q", runcard, instruments, channels)
    platform.resonator_type = "2D"

    # assign channels to qubits and sweetspots(operating points)
    qubits = platform.qubits
    for q in range(0, 5):
        qubits[q].feedback = channels["L3-31"]
        qubits[q].readout = channels["L2-7"]

    for q in range(0, 5):
        qubits[q].drive = channels[f"L4-{15 + q}"]
        qubits[q].flux = channels[f"L4-{6 + q}"]
        channels[f"L4-{6 + q}"].qubit = qubits[q]

    # assign channels to couplers and sweetspots(operating points)
    for c in range(0, 2):
        qubits[f"c{c}"].flux = channels[f"L4-{11 + c}"]
        channels[f"L4-{11 + c}"].qubit = qubits[f"c{c}"]
    for c in range(3, 5):
        qubits[f"c{c}"].flux = channels[f"L4-{10 + c}"]
        channels[f"L4-{10 + c}"].qubit = qubits[f"c{c}"]

    # assign qubits to couplers
    for c in itertools.chain(range(0, 2), range(3, 5)):
        qubits[f"c{c}"].flux_coupler = [qubits[c]]
        qubits[f"c{c}"].flux_coupler.append(qubits[2])

    return platform


def test_random_functions():
    platform = create(RUNCARD)
    IQM5q = platform.instruments[0]
    IQM5q.start()
    IQM5q.stop()
    IQM5q.disconnect()


# def test_connections():
#     platform = create(RUNCARD)
#     IQM5q = platform.design.instruments[0]
#     # IQM5q.connect()


@pytest.mark.parametrize("shape", ["Rectangular", "Gaussian", "GaussianSquare", "Drag"])
def test_zhpulse(shape):
    if shape == "Rectangular":
        pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    if shape == "Gaussian":
        pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Gaussian(5), "ch0", qubit=0)
    if shape == "GaussianSquare":
        pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Gaussian(5), "ch0", qubit=0)
    if shape == "Drag":
        pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Drag(5, 0.4), "ch0", qubit=0)

    zhpulse = ZhPulse(pulse)
    assert zhpulse.pulse.serial == pulse.serial
    assert zhpulse.zhpulse.length == 40e-9


@pytest.mark.parametrize("parameter", [Parameter.bias, Parameter.delay])
def test_select_sweeper(parameter):
    swept_points = 5
    platform = create(RUNCARD)
    qubits = {0: platform.qubits[0]}
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits.values():
        q = qubit.name
        qd_pulses[q] = platform.create_RX_pulse(q, start=0)
        sequence.add(qd_pulses[q])
        ro_pulses[q] = platform.create_qubit_readout_pulse(q, start=qd_pulses[q].finish)
        sequence.add(ro_pulses[q])

        parameter_range = np.random.randint(swept_points, size=swept_points)
        if parameter is Parameter.delay:
            sweeper = Sweeper(parameter, parameter_range, pulses=[qd_pulses[q]])
        if parameter is Parameter.bias:
            sweeper = Sweeper(parameter, parameter_range, qubits=q)

        ZhSweeper = ZhSweeperLine(sweeper, qubit, sequence)
        assert ZhSweeper.sweeper == sweeper


def test_zhinst_setup():
    platform = create(RUNCARD)
    platform.setup()
    IQM5q = platform.instruments[0]
    assert IQM5q.time_of_flight == 280


def test_zhsequence():
    qd_pulse = Pulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch0", qubit=0)
    ro_pulse = ReadoutPulse(0, 40, 0.05, int(3e9), 0.0, Rectangular(), "ch1", qubit=0)
    sequence = PulseSequence()
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)
    IQM5q = create(RUNCARD)

    IQM5q.instruments[0].sequence_zh(sequence, IQM5q.qubits, sweepers=[])
    zhsequence = IQM5q.instruments[0].sequence

    with pytest.raises(AttributeError):
        IQM5q.instruments[0].sequence_zh("sequence", IQM5q.qubits, sweepers=[])
        zhsequence = IQM5q.instruments[0].sequence

    assert len(zhsequence) == 2
    assert len(zhsequence["readout0"]) == 1


def test_zhinst_register_readout_line():
    platform = create(RUNCARD)
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup, descriptor = create_offline_device_setup()
    IQM5q.register_readout_line(platform.qubits[0], intermediate_frequency=int(1e6))

    assert "measure0" in IQM5q.signal_map
    assert "acquire0" in IQM5q.signal_map
    assert "/logical_signal_groups/q0/measure_line" in IQM5q.calibration.calibration_items


def test_zhinst_register_drive_line():
    platform = create(RUNCARD)
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup, descriptor = create_offline_device_setup()
    IQM5q.register_drive_line(platform.qubits[0], intermediate_frequency=int(1e6))

    assert "drive0" in IQM5q.signal_map
    assert "/logical_signal_groups/q0/drive_line" in IQM5q.calibration.calibration_items


def test_zhinst_register_flux_line():
    platform = create(RUNCARD)
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup, descriptor = create_offline_device_setup()
    IQM5q.register_flux_line(platform.qubits[0])

    assert "flux0" in IQM5q.signal_map
    assert "/logical_signal_groups/q0/flux_line" in IQM5q.calibration.calibration_items


def test_experiment_execute_pulse_sequence():
    platform = create(RUNCARD)
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup, descriptor = create_offline_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0], "c0": platform.qubits["c0"]}
    platform.qubits = qubits

    ro_pulses = {}
    qf_pulses = {}
    for qubit in qubits.values():
        q = qubit.name
        qf_pulses[q] = FluxPulse(
            start=0,
            duration=500,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[q].flux.name,
            qubit=q,
        )
        sequence.add(qf_pulses[q])
        if qubit.flux_coupler:
            continue
        ro_pulses[q] = platform.create_qubit_readout_pulse(q, start=qf_pulses[q].finish)
        sequence.add(ro_pulses[q])

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    IQM5q.experiment_flow(qubits, sequence, options)

    assert "flux0" in IQM5q.experiment.signals
    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals


@pytest.mark.parametrize("fast_reset", [True, False])
def test_experiment_execute_pulse_sequence(fast_reset):
    platform = create(RUNCARD)
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup, descriptor = create_offline_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}
    platform.qubits = qubits

    ro_pulses = {}
    qd_pulses = {}
    qf_pulses = {}
    fr_pulses = {}
    for qubit in qubits:
        if fast_reset:
            fr_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(ro_pulses[qubit])
        qf_pulses[qubit] = FluxPulse(
            start=0,
            duration=ro_pulses[qubit].se_start,
            amplitude=1,
            shape=Rectangular(),
            channel=platform.qubits[qubit].flux.name,
            qubit=qubit,
        )
        sequence.add(qf_pulses[qubit])

    if fast_reset:
        fast_reset = fr_pulses

    options = ExecutionParameters(
        relaxation_time=300e-6,
        fast_reset=fast_reset,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    IQM5q.experiment_flow(qubits, sequence, options)

    assert "drive0" in IQM5q.experiment.signals
    assert "flux0" in IQM5q.experiment.signals
    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals


@pytest.mark.parametrize("parameter1", [Parameter.delay, Parameter.duration])
def test_experiment_sweep_single(parameter1):
    platform = create(RUNCARD)
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup, descriptor = create_offline_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(ro_pulses[qubit])

    parameter_range_1 = (
        np.random.rand(swept_points)
        if parameter1 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    sweepers = []
    sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=[qd_pulses[qubit]]))

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    IQM5q.sweepers = sweepers

    IQM5q.experiment_flow(qubits, sequence, options, sweepers)

    # assert
    # AcquisitionType.SPECTROSCOPY
    # AveragingMode.CYCLIC
    # I'm using dumb IW

    assert 1 == 1


SweeperParameter = {
    Parameter.frequency,
    Parameter.amplitude,
    Parameter.duration,
    Parameter.delay,
    Parameter.relative_phase,
}


@pytest.mark.parametrize("parameter1", Parameter)
@pytest.mark.parametrize("parameter2", Parameter)
def test_experiment_sweep_2d_general(parameter1, parameter2):
    platform = create(RUNCARD)
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup, descriptor = create_offline_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(ro_pulses[qubit])

    parameter_range_1 = (
        np.random.rand(swept_points)
        if parameter1 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    parameter_range_2 = (
        np.random.rand(swept_points)
        if parameter2 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    sweepers = []
    if parameter1 in SweeperParameter:
        if parameter1 is not Parameter.delay:
            sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=[ro_pulses[qubit]]))
    if parameter2 in SweeperParameter:
        if parameter2 is Parameter.amplitude:
            if parameter1 is not Parameter.amplitude:
                sweepers.append(Sweeper(parameter2, parameter_range_2, pulses=[qd_pulses[qubit]]))

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    IQM5q.sweepers = sweepers
    rearranging_axes, sweepers = IQM5q.rearrange_sweepers(sweepers)
    IQM5q.experiment_flow(qubits, sequence, options, sweepers)

    assert "drive0" in IQM5q.experiment.signals
    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals


def test_experiment_sweep_2d_specific():
    platform = create(RUNCARD)
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup, descriptor = create_offline_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        sequence.add(qd_pulses[qubit])
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
        sequence.add(ro_pulses[qubit])

    parameter1 = Parameter.relative_phase
    parameter2 = Parameter.frequency

    parameter_range_1 = (
        np.random.rand(swept_points)
        if parameter1 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    parameter_range_2 = (
        np.random.rand(swept_points)
        if parameter2 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    sweepers = []
    sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=[qd_pulses[qubit]]))
    sweepers.append(Sweeper(parameter2, parameter_range_2, pulses=[qd_pulses[qubit]]))

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    IQM5q.sweepers = sweepers
    rearranging_axes, sweepers = IQM5q.rearrange_sweepers(sweepers)
    IQM5q.experiment_flow(qubits, sequence, options, sweepers)

    assert "drive0" in IQM5q.experiment.signals
    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals


@pytest.mark.parametrize("parameter", [Parameter.frequency, Parameter.amplitude, Parameter.bias])
def test_experiment_sweep_punchouts(parameter):
    platform = create(RUNCARD)
    platform.setup()
    IQM5q = platform.instruments[0]
    IQM5q.device_setup, descriptor = create_offline_device_setup()

    sequence = PulseSequence()
    qubits = {0: platform.qubits[0]}

    if parameter is Parameter.frequency:
        parameter1 = Parameter.frequency
        parameter2 = Parameter.amplitude
    if parameter is Parameter.amplitude:
        parameter1 = Parameter.amplitude
        parameter2 = Parameter.frequency
    if parameter is Parameter.bias:
        parameter1 = Parameter.bias
        parameter2 = Parameter.frequency

    swept_points = 5
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    parameter_range_1 = (
        np.random.rand(swept_points)
        if parameter1 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    parameter_range_2 = (
        np.random.rand(swept_points)
        if parameter2 is Parameter.amplitude
        else np.random.randint(swept_points, size=swept_points)
    )

    sweepers = []
    if parameter1 is Parameter.bias:
        sweepers.append(Sweeper(parameter1, parameter_range_1, qubits=[qubits[qubit]]))
    else:
        sweepers.append(Sweeper(parameter1, parameter_range_1, pulses=[ro_pulses[qubit]]))
    sweepers.append(Sweeper(parameter2, parameter_range_2, pulses=[ro_pulses[qubit]]))

    options = ExecutionParameters(
        relaxation_time=300e-6, acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )

    IQM5q.sweepers = sweepers
    rearranging_axes, sweepers = IQM5q.rearrange_sweepers(sweepers)
    IQM5q.experiment_flow(qubits, sequence, options, sweepers)

    assert "measure0" in IQM5q.experiment.signals
    assert "acquire0" in IQM5q.experiment.signals


# TODO: SIM NOT WORKING
# def test_sim():
#     platform = create(RUNCARD)
#     platform.setup()
#     IQM5q = platform.design.instruments[0]
#     IQM5q.device_setup, descriptor = create_offline_device_setup()

#     sequence = PulseSequence()
#     qubits = {0: platform.qubits[0]}
#     platform.qubits = qubits

#     ro_pulses = {}
#     qd_pulses = {}
#     qf_pulses = {}
#     for qubit in qubits:
#         qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
#         sequence.add(qd_pulses[qubit])
#         ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=qd_pulses[qubit].finish)
#         sequence.add(ro_pulses[qubit])
#         qf_pulses[qubit] = FluxPulse(
#             start=0,
#             duration=500,
#             amplitude=1,
#             shape=Rectangular(),
#             channel=platform.qubits[qubit].flux.name,
#             qubit=qubit,
#         )
#         sequence.add(qf_pulses[qubit])
