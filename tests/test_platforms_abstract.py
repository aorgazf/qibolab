"""Tests :class:`qibolab.platforms.multiqubit.MultiqubitPlatform` and
:class:`qibolab.platforms.platform.DesignPlatform`.
"""
import pickle
import warnings

import numpy as np
import pytest
from qibo.models import Circuit
from qibo.states import CircuitResult

from qibolab import create_platform
from qibolab.backends import QibolabBackend
from qibolab.execution_parameters import ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence

nshots = 1024


def test_create_platform(platform):
    assert isinstance(platform, Platform)


def test_create_platform_error():
    with pytest.raises(ValueError):
        platform = create_platform("nonexistent")


def test_abstractplatform_pickle(platform):
    serial = pickle.dumps(platform)
    new_platform = pickle.loads(serial)
    assert new_platform.name == platform.name
    assert new_platform.runcard == platform.runcard
    assert new_platform.settings == platform.settings
    assert new_platform.is_connected == platform.is_connected


# TODO: this test should be improved
@pytest.mark.parametrize(
    "par",
    [
        "readout_frequency",
        "sweetspot",
        "threshold",
        "bare_resonator_frequency",
        "drive_frequency",
        "iq_angle",
        "mean_gnd_states",
        "mean_exc_states",
        "classifiers_hpars",
    ],
)
@pytest.mark.qpu
def test_update(platform, par):
    new_values = np.ones(platform.nqubits)
    if "states" in par:
        updates = {par: {q: [new_values[i], new_values[i]] for i, q in enumerate(platform.qubits)}}
    else:
        updates = {par: {q: new_values[i] for i, q in enumerate(platform.qubits)}}
    platform.update(updates)
    for i, qubit in platform.qubits.items():
        value = updates[par][i]
        if "frequency" in par:
            value *= 1e9
        if "states" in par:
            assert value == platform.settings["characterization"]["single_qubit"][i][par]
            assert value == getattr(qubit, par)
        else:
            assert value == float(platform.settings["characterization"]["single_qubit"][i][par])
            assert value == float(getattr(qubit, par))


@pytest.fixture(scope="module")
def connected_platform(platform):
    platform.connect()
    platform.setup()
    platform.start()
    yield platform
    platform.stop()
    platform.disconnect()


@pytest.mark.qpu
def test_abstractplatform_setup_start_stop(connected_platform):
    pass


@pytest.mark.qpu
def test_platform_execute_empty(connected_platform):
    # an empty pulse sequence
    platform = connected_platform
    sequence = PulseSequence()
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_platform_execute_one_drive_pulse(connected_platform):
    # One drive pulse
    platform = connected_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_multiqubitplatform_execute_one_long_drive_pulse(connected_platform):
    # Long duration
    platform = connected_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=8192 + 200))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_multiqubitplatform_execute_one_extralong_drive_pulse(connected_platform):
    # Extra Long duration
    platform = connected_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=2 * 8192 + 200))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_multiqubitplatform_execute_one_drive_one_readout(connected_platform):
    # One drive pulse and one readout pulse
    platform = connected_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.add(platform.create_qubit_readout_pulse(qubit, start=200))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_multiqubitplatform_execute_multiple_drive_pulses_one_readout(connected_platform):
    # Multiple qubit drive pulses and one readout pulse
    platform = connected_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=204, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=408, duration=400))
    sequence.add(platform.create_qubit_readout_pulse(qubit, start=808))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_multiqubitplatform_execute_multiple_drive_pulses_one_readout_no_spacing(
    connected_platform,
):
    # Multiple qubit drive pulses and one readout pulse with no spacing between them
    platform = connected_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=200, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=400, duration=400))
    sequence.add(platform.create_qubit_readout_pulse(qubit, start=800))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_multiqubitplatform_execute_multiple_overlaping_drive_pulses_one_readout(
    connected_platform,
):
    # Multiple overlapping qubit drive pulses and one readout pulse
    platform = connected_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=0, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=200, duration=200))
    sequence.add(platform.create_qubit_drive_pulse(qubit, start=50, duration=400))
    sequence.add(platform.create_qubit_readout_pulse(qubit, start=800))
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.mark.qpu
def test_multiqubitplatform_execute_multiple_readout_pulses(connected_platform):
    # Multiple readout pulses
    platform = connected_platform
    qubit = next(iter(platform.qubits))
    sequence = PulseSequence()
    qd_pulse1 = platform.create_qubit_drive_pulse(qubit, start=0, duration=200)
    ro_pulse1 = platform.create_qubit_readout_pulse(qubit, start=200)
    qd_pulse2 = platform.create_qubit_drive_pulse(qubit, start=(ro_pulse1.start + ro_pulse1.duration), duration=400)
    ro_pulse2 = platform.create_qubit_readout_pulse(qubit, start=(ro_pulse1.start + ro_pulse1.duration + 400))
    sequence.add(qd_pulse1)
    sequence.add(ro_pulse1)
    sequence.add(qd_pulse2)
    sequence.add(ro_pulse2)
    platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=nshots))


@pytest.fixture
def qubits(connected_platform):
    for qubit in connected_platform.qubits:
        yield connected_platform, qubit


@pytest.mark.qpu
@pytest.mark.xfail(raises=AssertionError, reason="Probabilities are not well calibrated")
def test_excited_state_probabilities_pulses(qubits):
    platform, qubit = qubits
    backend = QibolabBackend(platform)
    qd_pulse = platform.create_RX_pulse(qubit)
    ro_pulse = platform.create_MZ_pulse(qubit, start=qd_pulse.duration)
    sequence = PulseSequence()
    sequence.add(qd_pulse)
    sequence.add(ro_pulse)
    result = platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=5000))

    cr = CircuitResult(backend, Circuit(platform.nqubits), result, nshots=5000)
    probs = backend.circuit_result_probabilities(cr, qubits=[qubit])
    warnings.warn(f"Excited state probabilities: {probs}")
    np.testing.assert_allclose(probs, [0, 1], atol=0.05)


@pytest.mark.qpu
@pytest.mark.parametrize("start_zero", [False, True])
@pytest.mark.xfail(raises=AssertionError, reason="Probabilities are not well calibrated")
def test_ground_state_probabilities_pulses(qubits, start_zero):
    platform, qubit = qubits
    backend = QibolabBackend(platform)
    if start_zero:
        ro_pulse = platform.create_MZ_pulse(qubit, start=0)
    else:
        qd_pulse = platform.create_RX_pulse(qubit)
        ro_pulse = platform.create_MZ_pulse(qubit, start=qd_pulse.duration)
    sequence = PulseSequence()
    sequence.add(ro_pulse)
    result = platform.execute_pulse_sequence(sequence, ExecutionParameters(nshots=5000))

    cr = CircuitResult(backend, Circuit(platform.nqubits), result, nshots=5000)
    probs = backend.circuit_result_probabilities(cr, qubits=[qubit])
    warnings.warn(f"Ground state probabilities: {probs}")
    np.testing.assert_allclose(probs, [1, 0], atol=0.05)
