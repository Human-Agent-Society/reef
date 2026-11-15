"""Serving backend doubles and waits for the inference worker's published routes."""

import time

from reef.dispatcher import Dispatcher
from reef.inference.interfaces import InferenceBackend


class Engines(InferenceBackend):
    """Serving engines that record each control call.

    ``misreport`` names a version they report after a load; ``dead`` is the
    number of engines the next ``recover`` replaces.
    """

    def __init__(self) -> None:
        self.events: list = []
        self.versions = ["boot"]
        self.misreport: str | None = None
        self.dead = 0

    def load_weight_files(self, files, *, adapter_name, runtime_load_id):
        self.events.append(("load", files.layout, adapter_name, runtime_load_id, files.path))
        self.versions = [runtime_load_id if self.misreport is None else self.misreport]

    def initialize_version(self, runtime_load_id):
        self.events.append(("initialize", runtime_load_id))
        self.versions = [runtime_load_id]

    def inference_url(self):
        return "http://component.example"

    def runtime_load_ids(self):
        return list(self.versions)

    def pause(self):
        self.events.append("pause")

    def resume(self):
        self.events.append("resume")

    def recover(self):
        replaced, self.dead = self.dead, 0
        self.events.append(("recover", replaced))
        return replaced

    def abort(self):
        self.events.append("abort")
        self.dead = 2

    def unload_adapter(self, name):
        self.events.append(("unload", name))

    def loads(self) -> list:
        return [event for event in self.events if event[0] == "load"]


def wait_for_serving(dispatcher: Dispatcher, scenario: str = "math") -> None:
    """Wait for the current selection's published routes, or report its startup failure."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        item = dispatcher.coordination.served_models.current_request(scenario)
        selected = dispatcher.get_or_create_scenario(scenario).artifact_selection().to_dict()
        if item is not None and item.payload.get("selection") == selected:
            if item.state == "finished":
                return
            if item.state == "failed":
                raise AssertionError(f"inference startup failed: {item.error}")
        time.sleep(0.01)
    raise AssertionError(f"inference did not publish the selection for {scenario}")
