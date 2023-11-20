"""ERAsynth drivers.

Supports the ERAsynth ++.

https://qcodes.github.io/Qcodes_contrib_drivers/_modules/qcodes_contrib_drivers/drivers/ERAInstruments/erasynth.html#ERASynthBase.clear_read_buffer
"""

import json

import requests
from qcodes_contrib_drivers.drivers.ERAInstruments import ERASynthPlusPlus
from qibo.config import log

from qibolab.instruments.oscillator import DummyDevice, LocalOscillator


class ERASynthEthernet(DummyDevice):
    """ERA ethernet driver that follows the QCoDeS interface."""

    MAX_RECONNECTION_ATTEMPTS = 10
    TIMEOUT = 10

    def __init__(self, name, address):
        self.name = name
        self.address = address
        self.post("readAll", 1)
        self.post("readDiagnostic", 0)
        self.post("rfoutput", 0)

    @property
    def url(self):
        return f"http://{self.address}/"

    def post(self, name, value):
        """Post a value to the instrument's web server.

        Try to post multiple times, waiting for 0.1 seconds between each attempt.

        Args:
            name: str = The name of the value to post.
            value: str = The value to post.
        """
        value = str(value)
        for _ in range(self.MAX_RECONNECTION_ATTEMPTS):
            try:
                response = requests.post(self.url, data={name: value}, timeout=self.TIMEOUT)
                if response.status_code == 200:
                    return True
                break
            except (ConnectionError, TimeoutError, requests.exceptions.ReadTimeout):
                log.info("ERAsynth connection timed out, retrying...")
        raise ConnectionError(f"Unable to post {name}={value} to {self.name}")

    def get(self, name):
        """Get a value from the instrument's web server.

        Try to get multiple times, waiting for 0.1 seconds between each attempt.

        Args:
            name: str = The name of the value to get.
        """
        if name == "ref_osc_source":
            value = self.get("reference_int_ext")
            if value == 1:
                return "EXT"
            else:
                return "INT"

        for _ in range(self.MAX_RECONNECTION_ATTEMPTS):
            try:
                response = requests.post(self.url, params={"readAll": 1}, timeout=self.TIMEOUT)
                if response.status_code == 200:
                    # reponse.text is a dictonary in string format, convert it to a dictonary
                    return json.loads(response.text)[name]
                break
            except (ConnectionError, TimeoutError, requests.exceptions.ReadTimeout):
                log.info("ERAsynth connection timed out, retrying...")

        raise ConnectionError(f"Unable to get {name} from {self.name}")

    def set(self, name, value):
        if name == "ref_osc_source":
            if value in ("int", "internal", "INT", "INTERNAL"):
                self.post("reference_int_ext", 0)
            elif value in ("ext", "external", "EXT", "EXTERNAL"):
                self.post("reference_int_ext", 1)
            else:
                raise ValueError(f"Invalid reference clock source {value}")

        elif name == "frequency":
            self.post(name, int(value))

        elif name == "power":
            self.post(name, float(value))

        else:
            self.post(name, value)

    def on(self):
        self.post("rfoutput", 1)

    def off(self):
        self.post("rfoutput", 0)

    def close(self):
        self.off()


class ERA(LocalOscillator):
    def __init__(self, name, address, ethernet=True, reference_clock_source=None):
        super().__init__(name, address, reference_clock_source)
        self.ethernet = ethernet

    def create(self):
        if self.ethernet:
            return ERASynthEthernet(self.name, self.address)
        else:
            return ERASynthPlusPlus(f"{self.name}", f"TCPIP::{self.address}::INSTR")

    def __del__(self):
        self.disconnect()
