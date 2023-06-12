"""Tests for RFSoC driver"""

import numpy as np
import pytest
import qibosoq.components as rfsoc

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters, create_platform
from qibolab.instruments.rfsoc import (
    convert_frequency_sweeper,
    convert_pulse,
    convert_qubit,
    convert_sweep,
)
from qibolab.platform import Qubit
from qibolab.pulses import Drag, Gaussian, Pulse, PulseSequence, PulseType, Rectangular
from qibolab.result import (
    AveragedIntegratedResults,
    AveragedSampleResults,
    IntegratedResults,
    SampleResults,
)
from qibolab.sweeper import Parameter, Sweeper


def test_convert_qubit(dummy_qrc):
    """Tests conversion from `qibolab.platforms.abstract.Qubit` to `rfsoc.Qubit`"""

    platform = create_platform("rfsoc")
    qubit = platform.qubits[0]
    qubit.flux.bias = 0.05
    qubit.flux.ports = [("name", 4)]
    qubit = convert_qubit(qubit)
    targ = rfsoc.Qubit(0.05, 4)

    assert qubit == targ

    platform = create_platform("rfsoc")
    qubit = platform.qubits[0]
    qubit.flux = None
    qubit = convert_qubit(qubit)
    targ = rfsoc.Qubit(0.0, None)

    assert qubit == targ


def test_convert_pulse(dummy_qrc):
    """Tests conversion from `qibolab.pulses.Pulse` to `rfsoc.Pulse`"""

    platform = create_platform("rfsoc")
    qubit = platform.qubits[0]
    qubit.drive.ports = [("name", 4)]
    qubit.readout.ports = [("name", 2)]
    qubit.feedback.ports = [("name", 1)]
    qubit.readout.local_oscillator._frequency = 1e6

    pulse = Pulse(0, 40, 0.9, 50e6, 0, Drag(5, 2), 0, PulseType.DRIVE, 0)
    targ = rfsoc.Pulse(50, 0.9, 0, 0, 0.04, pulse.serial, "drive", 4, None, "drag", 5, 2)
    assert convert_pulse(pulse, platform.qubits) == targ

    pulse = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(2), 0, PulseType.DRIVE, 0)
    targ = rfsoc.Pulse(50, 0.9, 0, 0, 0.04, pulse.serial, "drive", 4, None, "gaussian", 2)
    assert convert_pulse(pulse, platform.qubits) == targ

    pulse = Pulse(0, 40, 0.9, 50e6, 0, Rectangular(), 0, PulseType.READOUT, 0)
    targ = rfsoc.Pulse(49, 0.9, 0, 0, 0.04, pulse.serial, "readout", 2, 1, "rectangular")
    assert convert_pulse(pulse, platform.qubits) == targ


def test_convert_frequency_sweeper(dummy_qrc):
    """Tests frequency conversion for `rfsoc.Sweeper` objects"""
    platform = create_platform("rfsoc")
    qubit = platform.qubits[0]
    qubit.drive.ports = [("name", 4)]
    qubit.readout.ports = [("name", 2)]
    qubit.feedback.ports = [("name", 1)]
    qubit.readout.local_oscillator._frequency = 1e6

    seq = PulseSequence()
    pulse0 = Pulse(0, 40, 0.9, 50e6, 0, Gaussian(2), 0, PulseType.DRIVE, 0)
    pulse1 = Pulse(0, 40, 0.9, 50e6, 0, Rectangular(), 0, PulseType.READOUT, 0)
    seq.add(pulse0)
    seq.add(pulse1)

    sweeper = rfsoc.Sweeper(10, [rfsoc.Parameter.FREQUENCY], [0], [10e6], [1])
    convert_frequency_sweeper(sweeper, seq, platform.qubits)

    assert sweeper.starts == [-1]
    assert sweeper.stops == [9]


def test_rfsoc_init(dummy_qrc):
    """Tests instrument can initilize and its attribute are assigned"""
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    assert instrument.host == "0.0.0.0"
    assert instrument.port == 0
    assert isinstance(instrument.cfg, rfsoc.Config)


def test_classify_shots(dummy_qrc):
    """Creates fake IQ values and check classification works as expected"""
    qubit0 = Qubit(name="q0", threshold=1, iq_angle=np.pi / 2)
    qubit1 = Qubit(
        name="q1",
    )
    i_val = [0] * 7
    q_val = [-5, -1.5, -0.5, 0, 0.5, 1.5, 5]

    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    shots = instrument.classify_shots(i_val, q_val, qubit0)
    target_shots = np.array([1, 1, 0, 0, 0, 0, 0])

    assert (target_shots == shots).all()
    assert instrument.classify_shots(i_val, q_val, qubit1) is None


def test_merge_sweep_results(dummy_qrc):
    """Creates fake dictionary of results and check merging works as expected"""
    dict_a = {"serial1": AveragedIntegratedResults(np.array([0 + 1j * 1]))}
    dict_b = {
        "serial1": AveragedIntegratedResults(np.array([4 + 1j * 4])),
        "serial2": AveragedIntegratedResults(np.array([5 + 1j * 5])),
    }
    dict_c = {}
    targ_dict = {
        "serial1": AveragedIntegratedResults(np.array([0 + 1j * 1, 4 + 1j * 4])),
        "serial2": AveragedIntegratedResults(np.array([5 + 1j * 5])),
    }

    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    out_dict1 = instrument.merge_sweep_results(dict_a, dict_b)
    out_dict2 = instrument.merge_sweep_results(dict_c, dict_a)

    assert targ_dict.keys() == out_dict1.keys()
    assert (out_dict1["serial1"].serialize["MSR[V]"] == targ_dict["serial1"].serialize["MSR[V]"]).all()
    assert (out_dict1["serial1"].serialize["MSR[V]"] == targ_dict["serial1"].serialize["MSR[V]"]).all()

    assert dict_a.keys() == out_dict2.keys()
    assert (out_dict2["serial1"].serialize["MSR[V]"] == dict_a["serial1"].serialize["MSR[V]"]).all()
    assert (out_dict2["serial1"].serialize["MSR[V]"] == dict_a["serial1"].serialize["MSR[V]"]).all()


def test_get_if_python_sweep(dummy_qrc):
    """Creates pulse sequences and check if they can be swept by the firmware.

    Qibosoq does not support sweep on readout frequency, more than one sweep
    at the same time, sweep on channels where multiple pulses are sent.
    If Qibosoq does not support the sweep, the driver will use a python loop
    """

    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence_1 = PulseSequence()
    sequence_1.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence_1.add(platform.create_MZ_pulse(qubit=0, start=100))

    sweep1 = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 100, 10), pulses=[sequence_1[0]])
    sweep2 = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 100, 10), pulses=[sequence_1[1]])
    sweep3 = Sweeper(parameter=Parameter.amplitude, values=np.arange(0.01, 0.5, 0.1), pulses=[sequence_1[1]])
    sweep1 = convert_sweep(sweep1, sequence_1, platform.qubits)
    sweep2 = convert_sweep(sweep2, sequence_1, platform.qubits)
    sweep3 = convert_sweep(sweep3, sequence_1, platform.qubits)

    assert instrument.get_if_python_sweep(sequence_1, platform.qubits, sweep2)
    assert not instrument.get_if_python_sweep(sequence_1, platform.qubits, sweep1)
    assert not instrument.get_if_python_sweep(sequence_1, platform.qubits, sweep3)

    sequence_2 = PulseSequence()
    sequence_2.add(platform.create_RX_pulse(qubit=0, start=0))

    sweep1 = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 100, 10), pulses=[sequence_2[0]])
    sweep2 = Sweeper(parameter=Parameter.amplitude, values=np.arange(0.01, 0.5, 0.1), pulses=[sequence_2[0]])
    sweep1 = convert_sweep(sweep1, sequence_2, platform.qubits)
    sweep2 = convert_sweep(sweep2, sequence_2, platform.qubits)

    assert not instrument.get_if_python_sweep(sequence_2, platform.qubits, sweep1)
    assert not instrument.get_if_python_sweep(sequence_2, platform.qubits, sweep1, sweep2)

    # TODO repetition
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence_1 = PulseSequence()
    sequence_1.add(platform.create_RX_pulse(qubit=0, start=0))
    sweep1 = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 100, 10), pulses=[sequence_1[0]])
    sweep2 = Sweeper(parameter=Parameter.relative_phase, values=np.arange(0, 1, 0.01), pulses=[sequence_1[0]])
    sweep3 = Sweeper(parameter=Parameter.bias, values=np.arange(-0.1, 0.1, 0.001), qubits=[platform.qubits[0]])
    sweep1 = convert_sweep(sweep1, sequence_1, platform.qubits)
    sweep2 = convert_sweep(sweep2, sequence_1, platform.qubits)
    sweep3 = convert_sweep(sweep3, sequence_1, platform.qubits)
    assert not instrument.get_if_python_sweep(sequence_1, platform.qubits, sweep1, sweep2, sweep3)

    platform.qubits[0].flux.bias = 0.5
    sweep1 = Sweeper(parameter=Parameter.bias, values=np.arange(-1, 1, 0.1), qubits=[0])
    with pytest.raises(ValueError):
        sweep1 = convert_sweep(sweep1, sequence_1, platform.qubits)


def test_convert_av_sweep_results(dummy_qrc):
    """Qibosoq sends results using nested lists, check if the conversion
    to dictionary of AveragedResults, for averaged sweep, works as expected
    """

    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=200))
    sweep1 = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 35, 10), pulses=[sequence[0]])
    sweep1 = convert_sweep(sweep1, sequence, platform.qubits)
    serial1 = sequence[1].serial
    serial2 = sequence[2].serial

    avgi = [[[1, 2, 3], [4, 1, 2]]]
    avgq = [[[7, 8, 9], [-1, -2, -3]]]

    execution_parameters = ExecutionParameters(
        acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )
    out_dict = instrument.convert_sweep_results(sequence.ro_pulses, platform.qubits, avgi, avgq, execution_parameters)
    targ_dict = {
        serial1: AveragedIntegratedResults(np.array([1, 2, 3]) + 1j * np.array([7, 8, 9])),
        serial2: AveragedIntegratedResults(np.array([4, 1, 2]) + 1j * np.array([-1, -2, -3])),
    }

    assert (out_dict[serial1].serialize["i[V]"] == targ_dict[serial1].serialize["i[V]"]).all()
    assert (out_dict[serial1].serialize["q[V]"] == targ_dict[serial1].serialize["q[V]"]).all()
    assert (out_dict[serial2].serialize["i[V]"] == targ_dict[serial2].serialize["i[V]"]).all()
    assert (out_dict[serial2].serialize["q[V]"] == targ_dict[serial2].serialize["q[V]"]).all()


def test_convert_nav_sweep_results(dummy_qrc):
    """Qibosoq sends results using nested lists, check if the conversion
    to dictionary of ExecutionResults, for not averaged sweep, works as expected
    """
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=200))
    sweep1 = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 35, 10), pulses=[sequence[0]])
    sweep1 = convert_sweep(sweep1, sequence, platform.qubits)
    serial1 = sequence[1].serial
    serial2 = sequence[2].serial

    avgi = [[[[1, 1], [2, 2], [3, 3]], [[4, 4], [1, 1], [2, 2]]]]
    avgq = [[[[7, 7], [8, 8], [9, 9]], [[-1, -1], [-2, -2], [-3, -3]]]]

    execution_parameters = ExecutionParameters(
        acquisition_type=AcquisitionType.INTEGRATION, averaging_mode=AveragingMode.CYCLIC
    )
    out_dict = instrument.convert_sweep_results(sequence.ro_pulses, platform.qubits, avgi, avgq, execution_parameters)
    targ_dict = {
        serial1: AveragedIntegratedResults(np.array([1, 1, 2, 2, 3, 3]) + 1j * np.array([7, 7, 8, 8, 9, 9])),
        serial2: AveragedIntegratedResults(np.array([4, 4, 1, 1, 2, 2]) + 1j * np.array([-1, -1, -2, -2, -3, -3])),
    }

    assert (out_dict[serial1].serialize["i[V]"] == targ_dict[serial1].serialize["i[V]"]).all()
    assert (out_dict[serial1].serialize["q[V]"] == targ_dict[serial1].serialize["q[V]"]).all()
    assert (out_dict[serial2].serialize["i[V]"] == targ_dict[serial2].serialize["i[V]"]).all()
    assert (out_dict[serial2].serialize["q[V]"] == targ_dict[serial2].serialize["q[V]"]).all()


@pytest.mark.qpu
def test_call_executepulsesequence():
    """Executes a PulseSequence and check if result shape is as expected.
    Both for averaged results and not averaged results.
    """
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))

    i_vals_nav, q_vals_nav = instrument._execute_pulse_sequence(instrument.cfg, sequence, platform.qubits, 1, False)
    i_vals_av, q_vals_av = instrument._execute_pulse_sequence(instrument.cfg, sequence, platform.qubits, 1, True)

    assert np.shape(i_vals_nav) == (1, 1, 1000)
    assert np.shape(q_vals_nav) == (1, 1, 1000)
    assert np.shape(i_vals_av) == (1, 1)
    assert np.shape(q_vals_av) == (1, 1)


@pytest.mark.qpu
def test_call_execute_sweeps():
    """Executes a firmware sweep and check if result shape is as expected.
    Both for averaged results and not averaged results.
    """
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))
    sweep = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 35, 10), pulses=[sequence[0]])
    expts = len(sweep.values)

    sweep = [convert_sweep(sweep, sequence, platform.qubits)]
    i_vals_nav, q_vals_nav = instrument._execute_sweeps(instrument.cfg, sequence, platform.qubits, sweep, 1, False)
    i_vals_av, q_vals_av = instrument._execute_sweeps(instrument.cfg, sequence, platform.qubits, sweep, 1, True)

    assert np.shape(i_vals_nav) == (1, 1, expts, 1000)
    assert np.shape(q_vals_nav) == (1, 1, expts, 1000)
    assert np.shape(i_vals_av) == (1, 1, expts)
    assert np.shape(q_vals_av) == (1, 1, expts)


@pytest.mark.qpu
def test_play():
    """Sends a PulseSequence using `play` and check results are what expected"""
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))

    out_dict = instrument.play(
        platform.qubits, sequence, ExecutionParameters(acquisition_type=AcquisitionType.INTEGRATION)
    )

    assert sequence[1].serial in out_dict
    assert isinstance(out_dict[sequence[1].serial], IntegratedResults)
    assert np.shape(out_dict[sequence[1].serial].voltage_i) == (1000,)


@pytest.mark.qpu
def test_sweep():
    """Sends a PulseSequence using `sweep` and check results are what expected"""
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))
    sweep = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 35, 10), pulses=[sequence[0]])

    out_dict1 = instrument.sweep(
        platform.qubits,
        sequence,
        ExecutionParameters(relaxation_time=100_000, averaging_mode=AveragingMode.CYCLIC),
        sweep,
    )
    out_dict2 = instrument.sweep(
        platform.qubits,
        sequence,
        ExecutionParameters(
            relaxation_time=100_000,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.SINGLESHOT,
        ),
        sweep,
    )

    assert sequence[1].serial in out_dict1
    assert sequence[1].serial in out_dict2
    assert isinstance(out_dict1[sequence[1].serial], AveragedSampleResults)
    assert isinstance(out_dict2[sequence[1].serial], IntegratedResults)
    assert np.shape(out_dict2[sequence[1].serial].voltage_i) == (1000, len(sweep.values))
    assert np.shape(out_dict1[sequence[1].serial].statistical_frequency) == (len(sweep.values),)


@pytest.mark.qpu
def test_python_reqursive_sweep():
    """Sends a PulseSequence directly to `python_reqursive_sweep` and check results are what expected"""
    platform = create_platform("rfsoc")
    instrument = platform.instruments[0]

    sequence = PulseSequence()
    sequence.add(platform.create_RX_pulse(qubit=0, start=0))
    sequence.add(platform.create_MZ_pulse(qubit=0, start=100))
    sweep1 = Sweeper(parameter=Parameter.amplitude, values=np.arange(0.01, 0.03, 10), pulses=[sequence[0]])
    sweep2 = Sweeper(parameter=Parameter.frequency, values=np.arange(10, 35, 10), pulses=[sequence[0]])

    out_dict = instrument.sweep(
        platform.qubits,
        sequence,
        ExecutionParameters(relaxation_time=100_000, averaging_mode=AveragingMode.CYCLIC),
        sweep1,
        sweep2,
    )

    assert sequence[1].serial in out_dict
