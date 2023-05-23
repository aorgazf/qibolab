import copy

import numpy as np
import yaml
from qibo.config import log, raise_error

from qibolab.platforms.abstract import AbstractPlatform, Qubit
from qibolab.pulses import PulseSequence, PulseType
from qibolab.result import ExecutionResults
from qibolab.sweeper import Parameter, Sweeper


class QbloxController(AbstractPlatform):
    def __init__(self, name, runcard):
        super().__init__(name, runcard)
        self.instruments = {}
        # Instantiate instruments
        for name in self.settings["instruments"]:
            lib = self.settings["instruments"][name]["lib"]
            i_class = self.settings["instruments"][name]["class"]
            address = self.settings["instruments"][name]["address"]
            from importlib import import_module

            InstrumentClass = getattr(import_module(f"qibolab.instruments.{lib}"), i_class)
            instance = InstrumentClass(name, address)
            self.instruments[name] = instance

        # Generate qubit_instrument_map from runcard
        self.qubit_instrument_map = {}
        for qubit in self.qubit_channel_map:
            self.qubit_instrument_map[qubit] = [None, None, None, None]
            for name in self.instruments:
                if self.settings["instruments"][name]["class"] in ["ClusterQRM_RF", "ClusterQCM_RF", "ClusterQCM"]:
                    for port in self.settings["instruments"][name]["settings"]["ports"]:
                        channel = self.settings["instruments"][name]["settings"]["ports"][port]["channel"]
                        if channel in self.qubit_channel_map[qubit]:
                            self.qubit_instrument_map[qubit][self.qubit_channel_map[qubit].index(channel)] = name
                if "s4g_modules" in self.settings["instruments"][name]["settings"]:
                    for channel in self.settings["instruments"][name]["settings"]["s4g_modules"]:
                        if channel in self.qubit_channel_map[qubit]:
                            self.qubit_instrument_map[qubit][self.qubit_channel_map[qubit].index(channel)] = name

        from qibolab.designs import Channel, ChannelMap

        # Create channel objects
        self.channels = ChannelMap.from_names(*self.settings["channels"])

    def reload_settings(self):
        super().reload_settings()
        self.characterization = self.settings["characterization"]
        self.qubit_channel_map = self.settings["qubit_channel_map"]
        self.hardware_avg = self.settings["settings"]["hardware_avg"]
        self.relaxation_time = self.settings["settings"]["relaxation_time"]

        # FIX: Set attenuation again to the original value after sweep attenuation in punchout
        if hasattr(self, "qubit_instrument_map"):
            for qubit in range(self.nqubits):
                instrument_name = self.qubit_instrument_map[qubit][0]
                port = self.qrm[qubit]._channel_port_map[self.qubit_channel_map[qubit][0]]
                att = self.settings["instruments"][instrument_name]["settings"]["ports"][port]["attenuation"]
                self.ro_port[qubit].attenuation = att

    def update(self, updates: dict):
        r"""Updates platform dependent runcard parameters and set up platform instruments if needed.

        Args:

            updates (dict): Dictionary containing the parameters to update the runcard.
        """
        for par, values in updates.items():
            for qubit, value in values.items():
                # resonator_punchout_attenuation
                if par == "readout_attenuation":
                    attenuation = int(value)
                    # save settings
                    instrument_name = self.qubit_instrument_map[qubit][0]
                    port = self.qrm[qubit]._channel_port_map[self.qubit_channel_map[qubit][0]]
                    self.settings["instruments"][instrument_name]["settings"]["ports"][port][
                        "attenuation"
                    ] = attenuation
                    # configure RO attenuation
                    self.ro_port[qubit].attenuation = attenuation

                # resonator_spectroscopy_flux / qubit_spectroscopy_flux
                if par == "sweetspot":
                    sweetspot = float(value)
                    # save settings
                    instrument_name = self.qubit_instrument_map[qubit][2]
                    port = self.qrm[qubit]._channel_port_map[self.qubit_channel_map[qubit][2]]
                    self.settings["instruments"][instrument_name]["settings"]["ports"][port]["offset"] = sweetspot
                    # configure instrument qcm_bb offset
                    self.qb_port[qubit].current = sweetspot

                # qubit_spectroscopy / qubit_spectroscopy_flux / ramsey
                if par == "drive_frequency":
                    freq = int(value * 1e9)

                    # update Qblox qubit LO drive frequency config
                    instrument_name = self.qubit_instrument_map[qubit][1]
                    port = self.qdm[qubit]._channel_port_map[self.qubit_channel_map[qubit][1]]
                    drive_if = self.single_qubit_natives[qubit]["RX"]["if_frequency"]
                    self.settings["instruments"][instrument_name]["settings"]["ports"][port]["lo_frequency"] = (
                        freq - drive_if
                    )

                    # set Qblox qubit LO drive frequency
                    self.qd_port[qubit].lo_frequency = freq - drive_if

                # classification
                if par == "threshold":
                    threshold = float(value)
                    # update Qblox qubit classification threshold
                    instrument_name = self.qubit_instrument_map[qubit][0]
                    self.settings["instruments"][instrument_name]["settings"]["classification_parameters"][qubit][
                        "threshold"
                    ] = threshold

                    self.instruments[instrument_name].setup(
                        **self.settings["settings"],
                        **self.settings["instruments"][instrument_name]["settings"],
                    )

                # classification
                if par == "iq_angle":
                    rotation_angle = float(value)
                    rotation_angle = (
                        rotation_angle * 360 / (2 * np.pi)
                    ) % 360  # save rotation angle in degrees for qblox
                    # update Qblox qubit classification iq angle
                    instrument_name = self.qubit_instrument_map[qubit][0]
                    self.settings["instruments"][instrument_name]["settings"]["classification_parameters"][qubit][
                        "rotation_angle"
                    ] = rotation_angle

                    self.instruments[instrument_name].setup(
                        **self.settings["settings"],
                        **self.settings["instruments"][instrument_name]["settings"],
                    )

                super().update(updates)

    def set_lo_drive_frequency(self, qubit, freq):
        self.qd_port[qubit].lo_frequency = freq

    def get_lo_drive_frequency(self, qubit):
        return self.qd_port[qubit].lo_frequency

    def set_lo_readout_frequency(self, qubit, freq):
        self.ro_port[qubit].lo_frequency = freq

    def get_lo_readout_frequency(self, qubit):
        return self.ro_port[qubit].lo_frequency

    def set_lo_twpa_frequency(self, qubit, freq):
        for instrument in self.instruments:
            if "twpa" in instrument:
                self.instruments[instrument].frequency = freq
                return None
        raise_error(NotImplementedError, "No twpa instrument found in the platform. ")

    def get_lo_twpa_frequency(self, qubit):
        for instrument in self.instruments:
            if "twpa" in instrument:
                return self.instruments[instrument].frequency
        raise_error(NotImplementedError, "No twpa instrument found in the platform. ")

    def set_lo_twpa_power(self, qubit, power):
        for instrument in self.instruments:
            if "twpa" in instrument:
                self.instruments[instrument].power = power
                return None
        raise_error(NotImplementedError, "No twpa instrument found in the platform. ")

    def get_lo_twpa_power(self, qubit):
        for instrument in self.instruments:
            if "twpa" in instrument:
                return self.instruments[instrument].power
        raise_error(NotImplementedError, "No twpa instrument found in the platform. ")

    def set_attenuation(self, qubit: Qubit, att):
        self.ro_port[qubit.name].attenuation = att
    
    def set_gain(self, qubit, gain):
        self.qd_port[qubit].gain = gain

    def set_bias(self, qubit: Qubit, bias):
        if qubit.name in self.qbm:
            self.qb_port[qubit.name].current = bias
        elif qubit.name in self.qfm:
            self.qf_port[qubit.name].offset = bias

    def get_attenuation(self, qubit: Qubit):
        return self.ro_port[qubit.name].attenuation      

    def get_bias(self, qubit: Qubit):
        if qubit.name in self.qbm:
            return self.qb_port[qubit.name].current
        elif qubit.name in self.qfm:
            return self.qf_port[qubit.name].offset

    def get_gain(self, qubit):
        return self.qd_port[qubit].gain

    def connect(self):
        """Connects to lab instruments using the details specified in the calibration settings."""
        if not self.is_connected:
            try:
                for name in self.instruments:
                    self.instruments[name].connect()
                self.is_connected = True
            except Exception as exception:
                raise_error(
                    RuntimeError,
                    "Cannot establish connection to " f"{self.name} instruments. " f"Error captured: '{exception}'",
                )
                # TODO: check for exception 'The instrument qrm_rf0 does not have parameters in0_att' and reboot the cluster

            else:
                log.info(f"All platform instruments connected.")

    def setup(self):
        if not self.is_connected:
            raise_error(
                RuntimeError,
                "There is no connection to the instruments, the setup cannot be completed",
            )

        for name in self.instruments:
            # Set up every with the platform settings and the instrument settings
            self.instruments[name].setup(
                **self.settings["settings"],
                **self.settings["instruments"][name]["settings"],
            )

        # Generate ro_channel[qubit], qd_channel[qubit], qf_channel[qubit], qrm[qubit], qcm[qubit], lo_qrm[qubit], lo_qcm[qubit]
        self.ro_channel = {}  # readout
        self.qd_channel = {}  # qubit drive
        self.qf_channel = {}  # qubit flux
        self.qb_channel = {}  # qubit flux biassing
        self.qrm = {}  # qubit readout module
        self.qdm = {}  # qubit drive module
        self.qfm = {}  # qubit flux module
        self.qbm = {}  # qubit flux biassing module
        self.ro_port = {}
        self.qd_port = {}
        self.qf_port = {}
        self.qb_port = {}

        for qubit in self.qubit_channel_map:
            self.ro_channel[qubit] = self.qubit_channel_map[qubit][0]
            self.qd_channel[qubit] = self.qubit_channel_map[qubit][1]
            self.qb_channel[qubit] = self.qubit_channel_map[qubit][2]
            self.qf_channel[qubit] = self.qubit_channel_map[qubit][3]

            if not self.qubit_instrument_map[qubit][0] is None:
                self.qrm[qubit] = self.instruments[self.qubit_instrument_map[qubit][0]]
                self.ro_port[qubit] = self.qrm[qubit].ports[
                    self.qrm[qubit]._channel_port_map[self.qubit_channel_map[qubit][0]]
                ]
                self.qubits[qubit].readout = self.channels[self.qubit_channel_map[qubit][0]]
            if not self.qubit_instrument_map[qubit][1] is None:
                self.qdm[qubit] = self.instruments[self.qubit_instrument_map[qubit][1]]
                self.qd_port[qubit] = self.qdm[qubit].ports[
                    self.qdm[qubit]._channel_port_map[self.qubit_channel_map[qubit][1]]
                ]
                self.qubits[qubit].drive = self.channels[self.qubit_channel_map[qubit][1]]
            if not self.qubit_instrument_map[qubit][2] is None:
                self.qfm[qubit] = self.instruments[self.qubit_instrument_map[qubit][2]]
                self.qf_port[qubit] = self.qfm[qubit].ports[
                    self.qfm[qubit]._channel_port_map[self.qubit_channel_map[qubit][2]]
                ]
                self.qubits[qubit].flux = self.channels[self.qubit_channel_map[qubit][2]]
            if not self.qubit_instrument_map[qubit][3] is None:
                self.qbm[qubit] = self.instruments[self.qubit_instrument_map[qubit][3]]
                self.qb_port[qubit] = self.qbm[qubit].dacs[self.qubit_channel_map[qubit][3]]

    def start(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].start()

    def stop(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].stop()

    def disconnect(self):
        if self.is_connected:
            for name in self.instruments:
                self.instruments[name].disconnect()
            self.is_connected = False

    def execute_pulse_sequence(
        self,
        sequence: PulseSequence,
        nshots=None,
        navgs=None,
        relaxation_time=None,
        sweepers: list() = [],  # list(Sweeper) = []
    ):
        if not self.is_connected:
            raise_error(RuntimeError, "Execution failed because instruments are not connected.")
        if nshots is None and navgs is None:
            nshots = 1
            navgs = self.hardware_avg
        elif nshots and navgs is None:
            navgs = 1
        elif navgs and nshots is None:
            nshots = 1

        if relaxation_time is None:
            relaxation_time = self.relaxation_time
        repetition_duration = sequence.finish + relaxation_time

        num_bins = nshots
        for sweeper in sweepers:
            num_bins *= len(sweeper.values)

        # DEBUG: Plot Pulse Sequence
        # sequence.plot('plot.png')
        # DEBUG: sync_en
        # from qblox_instruments.qcodes_drivers.cluster import Cluster
        # cluster:Cluster = self.instruments['cluster'].device
        # for module in cluster.modules:
        #     if module.get("present"):
        #         for sequencer in module.sequencers:
        #             if sequencer.get('sync_en'):
        #                 print(f"type: {module.module_type}, sequencer: {sequencer.name}, sync_en: True")

        # Process Pulse Sequence. Assign pulses to instruments and generate waveforms & program
        instrument_pulses = {}
        roles = {}
        data = {}
        for name in self.instruments:
            roles[name] = self.settings["instruments"][name]["roles"]
            if "control" in roles[name] or "readout" in roles[name]:
                instrument_pulses[name] = sequence.get_channel_pulses(*self.instruments[name].channels)

                # until we have frequency planning use the ifs stored in the runcard to change the los
                if self.instruments[name].__class__.__name__.split(".")[-1] in [
                    "ClusterQRM_RF",
                    "ClusterQCM_RF",
                    "ClusterQCM",
                ]:
                    for port in self.instruments[name].ports:
                        _los = []
                        _ifs = []
                        port_pulses = instrument_pulses[name].get_channel_pulses(
                            self.instruments[name]._port_channel_map[port]
                        )
                        for pulse in port_pulses:
                            if pulse.type == PulseType.READOUT:
                                _if = int(self.native_gates["single_qubit"][pulse.qubit]["MZ"]["if_frequency"])
                                pulse._if = _if
                                _los.append(int(pulse.frequency - _if))
                                _ifs.append(int(_if))
                            elif pulse.type == PulseType.DRIVE:
                                _if = int(self.native_gates["single_qubit"][pulse.qubit]["RX"]["if_frequency"])
                                pulse._if = _if
                                _los.append(int(pulse.frequency - _if))
                                _ifs.append(int(_if))

                        if len(_los) > 1:
                            for _ in range(1, len(_los)):
                                if _los[0] != _los[_]:
                                    raise ValueError(
                                        f"Pulses:\n{instrument_pulses[name]}\nsharing the lo at device: {name} - port: {port}\ncannot be synthesised with intermediate frequencies:\n{_ifs}"
                                    )
                        if len(_los) > 0:
                            self.instruments[name].ports[port].lo_frequency = _los[0]

                

                self.instruments[name].process_pulse_sequence(
                    instrument_pulses[name], navgs, nshots, repetition_duration, sweepers
                )

                # log.info(f"{self.instruments[name]}: Uploading pulse sequence")
                self.instruments[name].upload()

        for name in self.instruments:
            if "control" in roles[name] or "readout" in roles[name]:
                if True:  # not instrument_pulses[name].is_empty:
                    # log.info(f"{self.instruments[name]}: Playing pulse sequence")
                    self.instruments[name].play_sequence()

        acquisition_results = {}
        for name in self.instruments:
            if "readout" in roles[name]:
                if not instrument_pulses[name].is_empty:
                    if not instrument_pulses[name].ro_pulses.is_empty:
                        results = self.instruments[name].acquire()
                        existing_keys = set(acquisition_results.keys()) & set(results.keys())
                        for key, value in results.items():
                            if key in existing_keys:
                                acquisition_results[key].update(value)
                            else:
                                acquisition_results[key] = value

        for ro_pulse in sequence.ro_pulses:
            data[ro_pulse.serial] = ExecutionResults.from_components(*acquisition_results[ro_pulse.serial])
            data[ro_pulse.qubit] = copy.copy(data[ro_pulse.serial])
        return data

    def sweep(self, qubits, sequence, *sweepers, nshots=None, average=True, relaxation_time=None):
        id_results = {}
        map_id_serial = {}
        sequence_copy = sequence.copy()

        if nshots is None:
            nshots = self.hardware_avg
        navgs = nshots
        if average:
            nshots = 1
        else:
            navgs = 1

        if relaxation_time is None:
            relaxation_time = self.relaxation_time

        sweepers_copy = []
        for sweeper in sweepers:
            if sweeper.pulses:
                ps = []
                for pulse in sweeper.pulses:
                    if pulse in sequence_copy:
                        ps.append(sequence_copy[sequence_copy.index(pulse)])
            else:
                ps = None
            sweepers_copy.append(
                Sweeper(
                    parameter=sweeper.parameter,
                    values=sweeper.values,
                    pulses=ps,
                    qubits=sweeper.qubits,
                )
            )

        #reverse sweepers exept for res punchout att
        contains_attenuation_frequency = any(
            sweepers_copy[i].parameter == Parameter.attenuation and
            sweepers_copy[i + 1].parameter == Parameter.frequency
            for i in range(len(sweepers_copy) - 1)
        )

        if not contains_attenuation_frequency:
            sweepers_copy.reverse()


        for pulse in sequence_copy.ro_pulses:
            map_id_serial[pulse.id] = pulse.serial
            id_results[pulse.id] = ExecutionResults.from_components(np.array([]), np.array([]))
            id_results[pulse.qubit] = id_results[pulse.id]

        self._sweep_recursion(
            sequence_copy,
            *tuple(sweepers_copy),
            results=id_results,
            nshots=nshots,
            navgs=navgs,
            average=average,
            relaxation_time=relaxation_time,
        )

        serial_results = {}
        for pulse in sequence_copy.ro_pulses:
            serial_results[map_id_serial[pulse.id]] = id_results[pulse.id]
            serial_results[pulse.qubit] = id_results[pulse.id]
        return serial_results

    def _sweep_recursion(
        self,
        sequence,
        *sweepers,
        results,
        nshots,
        navgs,
        relaxation_time,
        average,
    ):
        sweeper = sweepers[0]

        initial = {}
        if sweeper.parameter is Parameter.attenuation:
            for qubit in sweeper.qubits:
                initial[qubit.name] = self.get_attenuation(qubit)

        # elif sweeper.parameter is Parameter.relative_phase:
        #     initial = {}
        #     for pulse in sweeper.pulses:
        #         initial[pulse.id] = pulse.relative_phase

        elif sweeper.parameter is Parameter.lo_frequency:
            initial = {}
            for pulse in sweeper.pulses:
                if pulse.type == PulseType.READOUT:
                    initial[pulse.id] = self.get_lo_readout_frequency(pulse.qubit)
                elif pulse.type == PulseType.DRIVE:
                    initial[pulse.id] = self.get_lo_readout_frequency(pulse.qubit)

        # elif sweeper.parameter is Parameter.frequency:
        #     initial = {}
        #     for pulse in sweeper.pulses:
        #         initial[pulse.id] = pulse.frequency
        
        # elif sweeper.parameter is Parameter.bias:
        #     initial = {}
        #     for qubit in sweeper.qubits:
        #         initial[qubit] = self.get_bias(qubit)

        elif sweeper.parameter is Parameter.gain:
            for pulse in sweeper.pulses:
                self.set_gain(pulse.qubit, 1)
        elif sweeper.parameter is Parameter.amplitude:
            for pulse in sweeper.pulses:
                pulse.amplitude = 1

        for_loop_sweepers = [Parameter.attenuation, Parameter.lo_frequency]
        rt_sweepers = [
            Parameter.frequency,
            Parameter.gain,
            Parameter.bias,
            Parameter.amplitude,
            Parameter.start,
            Parameter.duration,
            Parameter.relative_phase,
        ]

        if sweeper.parameter in for_loop_sweepers:
            # perform sweep recursively
            for value in sweeper.values:
                if sweeper.parameter is Parameter.attenuation:
                    for qubit in sweeper.qubits:
                        # self.set_attenuation(qubit, initial[qubit] + value)
                        self.set_attenuation(qubit, value)  # make att absolute
                # if sweeper.parameter is Parameter.relative_phase:
                #     for pulse in sweeper.pulses:
                #         pulse.relative_phase = initial[pulse.id] + value
                elif sweeper.parameter is Parameter.lo_frequency:
                    for pulse in sweeper.pulses:
                        if pulse.type == PulseType.READOUT:
                            self.set_lo_readout_frequency(initial[pulse.id] + value)
                        elif pulse.type == PulseType.DRIVE:
                            self.set_lo_readout_frequency(initial[pulse.id] + value)

                if len(sweepers) > 1:
                    self._sweep_recursion(
                        sequence,
                        *sweepers[1:],
                        results=results,
                        nshots=nshots,
                        navgs=navgs,
                        average=average,
                        relaxation_time=relaxation_time,
                    )
                else:
                    result = self.execute_pulse_sequence(sequence, nshots, navgs, relaxation_time)
                    for pulse in sequence.ro_pulses:
                        results[pulse.id] += result[pulse.serial].average if average else result[pulse.serial]
                        results[pulse.qubit] = results[pulse.id]
        else:
            split_relative_phase = False
            if sweeper.parameter == Parameter.relative_phase:
                from qibolab.instruments.qblox_q1asm import convert_phase

                c_values = np.array([convert_phase(v) for v in sweeper.values])
                if any(np.diff(c_values) < 0):
                    split_relative_phase = True
                    _from = 0
                    for idx in np.append(np.where(np.diff(c_values) < 0), len(c_values) - 1):
                        _to = idx + 1
                        _values = sweeper.values[_from:_to]
                        split_sweeper = Sweeper(
                            parameter=sweeper.parameter,
                            values=_values,
                            pulses=sweeper.pulses,
                            qubits=sweeper.qubits,
                        )
                        self._sweep_recursion(
                            sequence,
                            *(tuple([split_sweeper]) + sweepers[1:]),
                            results=results,
                            nshots=nshots,
                            navgs=navgs,
                            average=average,
                            relaxation_time=relaxation_time,
                        )
                        _from = _to

            if not split_relative_phase:
                if all(s.parameter in rt_sweepers for s in sweepers):
                    # rt-based sweepers
                    num_bins = nshots
                    for sweeper in sweepers:
                        num_bins *= len(sweeper.values)

                    if num_bins < 2**17:
                        repetition_duration = sequence.finish + relaxation_time
                        execution_time = navgs * num_bins * ((repetition_duration + 1000 * len(sweepers)) * 1e-9)
                        log.info(
                            f"Real time sweeper execution time: {int(execution_time)//60}m {int(execution_time) % 60}s"
                        )

                        result = self.execute_pulse_sequence(sequence, nshots, navgs, relaxation_time, sweepers)
                        for pulse in sequence.ro_pulses:
                            results[pulse.id] += result[pulse.serial]
                            results[pulse.qubit] = results[pulse.id]
                    else:
                        sweepers_repetitions = 1
                        for sweeper in sweepers:
                            sweepers_repetitions *= len(sweeper.values)
                        if sweepers_repetitions < 2**17:
                            # split nshots
                            max_rt_nshots = (2**17) // sweepers_repetitions
                            num_full_sft_iterations = nshots // max_rt_nshots
                            num_bins = max_rt_nshots * sweepers_repetitions

                            for sft_iteration in range(num_full_sft_iterations + 1):
                                _nshots = min(max_rt_nshots, nshots - sft_iteration * max_rt_nshots)
                                self._sweep_recursion(
                                    sequence,
                                    *sweepers,
                                    results=results,
                                    nshots=_nshots,
                                    navgs=navgs,
                                    average=average,
                                    relaxation_time=relaxation_time,
                                )
                        else:
                            for shot in range(nshots):
                                num_bins = 1
                                for sweeper in sweepers[1:]:
                                    num_bins *= len(sweeper.values)
                                sweeper = sweepers[0]
                                max_rt_iterations = (2**17) // num_bins
                                num_full_sft_iterations = len(sweeper.values) // max_rt_iterations
                                num_bins = nshots * max_rt_iterations
                                for sft_iteration in range(num_full_sft_iterations + 1):
                                    _from = sft_iteration * max_rt_iterations
                                    _to = min((sft_iteration + 1) * max_rt_iterations, len(sweeper.values))
                                    _values = sweeper.values[_from:_to]
                                    split_sweeper = Sweeper(
                                        parameter=sweeper.parameter,
                                        values=_values,
                                        pulses=sweeper.pulses,
                                        qubits=sweeper.qubits,
                                    )

                                    self._sweep_recursion(
                                        sequence,
                                        *(tuple([split_sweeper]) + sweepers[1:]),
                                        results=results,
                                        nshots=nshots,
                                        navgs=navgs,
                                        average=average,
                                        relaxation_time=relaxation_time,
                                    )
                else:
                    raise Exception("cannot execute a for-loop sweeper nested inside of a rt sweeper")

    def measure_fidelity(self, qubits=None, nshots=None):
        self.reload_settings()
        if not qubits:
            qubits = self.qubits
        results = {}
        for qubit in qubits:
            self.qrm[qubit].ports["i1"].hardware_demod_en = True  # required for binning
            # create exc sequence
            sequence_exc = PulseSequence()
            RX_pulse = self.create_RX_pulse(qubit, start=0)
            ro_pulse = self.create_qubit_readout_pulse(qubit, start=RX_pulse.duration)
            sequence_exc.add(RX_pulse)
            sequence_exc.add(ro_pulse)
            amplitude, phase, i, q = self.execute_pulse_sequence(sequence_exc, nshots=nshots)[
                "demodulated_integrated_binned"
            ][ro_pulse.serial]

            iq_exc = i + 1j * q

            sequence_gnd = PulseSequence()
            ro_pulse = self.create_qubit_readout_pulse(qubit, start=0)
            sequence_gnd.add(ro_pulse)

            amplitude, phase, i, q = self.execute_pulse_sequence(sequence_gnd, nshots=nshots)[
                "demodulated_integrated_binned"
            ][ro_pulse.serial]
            iq_gnd = i + 1j * q

            iq_mean_exc = np.mean(iq_exc)
            iq_mean_gnd = np.mean(iq_gnd)
            origin = iq_mean_gnd

            iq_gnd_translated = iq_gnd - origin
            iq_exc_translated = iq_exc - origin
            rotation_angle = np.angle(np.mean(iq_exc_translated))
            # rotation_angle = np.angle(iq_mean_exc - origin)
            iq_exc_rotated = iq_exc_translated * np.exp(-1j * rotation_angle) + origin
            iq_gnd_rotated = iq_gnd_translated * np.exp(-1j * rotation_angle) + origin

            # sort both lists of complex numbers by their real components
            # combine all real number values into one list
            # for each item in that list calculate the cumulative distribution
            # (how many items above that value)
            # the real value that renders the biggest difference between the two distributions is the threshold
            # that is the one that maximises fidelity

            real_values_exc = iq_exc_rotated.real
            real_values_gnd = iq_gnd_rotated.real
            real_values_combined = np.concatenate((real_values_exc, real_values_gnd))
            real_values_combined.sort()

            cum_distribution_exc = [
                sum(map(lambda x: x.real >= real_value, real_values_exc)) for real_value in real_values_combined
            ]
            cum_distribution_gnd = [
                sum(map(lambda x: x.real >= real_value, real_values_gnd)) for real_value in real_values_combined
            ]
            cum_distribution_diff = np.abs(np.array(cum_distribution_exc) - np.array(cum_distribution_gnd))
            argmax = np.argmax(cum_distribution_diff)
            threshold = real_values_combined[argmax]
            errors_exc = nshots - cum_distribution_exc[argmax]
            errors_gnd = cum_distribution_gnd[argmax]
            fidelity = cum_distribution_diff[argmax] / nshots
            assignment_fidelity = 1 - (errors_exc + errors_gnd) / nshots / 2
            # assignment_fidelity = 1/2 + (cum_distribution_exc[argmax] - cum_distribution_gnd[argmax])/nshots/2
            results[qubit] = ((rotation_angle * 360 / (2 * np.pi)) % 360, threshold, fidelity, assignment_fidelity)
        return results
