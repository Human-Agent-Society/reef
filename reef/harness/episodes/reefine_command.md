# Ask Reef to change this harness

The person typed /reefine to ask Reef for a change to this harness. The request is {request}. If it is empty, ask the person what they want changed and stop.

File the request with Reef and report what Reef does with it. Do not make the change yourself and do not edit any file: Reef writes, tests and publishes the change as a new version of this harness.

1. Run this shell command once, with the request as one double quoted argument in the person's own words:

       "$REEF_HARNESS_WRAPPER" evolve "<the request>"

   It prints the request id and a link to the request's page, where the step shows while it runs. Give the person the link.

2. Run `"$REEF_HARNESS_WRAPPER" wait <request id> --timeout 100` with the id it printed; if your shell tool takes a timeout, give it at least 150 seconds. While it exits with status 2 the step still runs (a step usually takes a few minutes): say so in one line and run it again.

3. Tell the person the result line it printed. When it names a release to install, ask whether to install it now. On yes run `"$REEF_HARNESS_WRAPPER" update`: the new version takes effect when the person starts reef-{adapter} again. If update says the release requires setup first, show the items it printed and tell the person to run `reef-{adapter} setup` in a terminal, then `reef-{adapter} update`.

When REEF_HARNESS_WRAPPER is not set, this session was not started through reef-{adapter}: say so and stop.
