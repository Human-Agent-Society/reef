"""``python3 -m reef.harness.harness_wrapper``, the line every installed wrapper script carries; the code is ``reef.harness.client.wrapper``.

The install script bakes this module path into ``reef-<adapter>`` on a
user's machine, and those scripts outlive any one Reef release, so this
name stays as long as such wrappers exist.
"""

from reef.harness.client.wrapper import main

if __name__ == "__main__":
    main()
