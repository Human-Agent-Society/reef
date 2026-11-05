"""Wait for the inference worker when a test needs the selected snapshot to serve."""

import time

from reef.dispatcher import Dispatcher


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
