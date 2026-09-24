# Ask Reef to change this harness

The person typed {command} to ask Reef for a change to this harness. The request is {request}. If it is empty, ask the person what they want changed and stop.

File the request with Reef and report what Reef does with it. Do not make the change yourself and do not edit any file: Reef writes, tests and publishes the change as a new version of this harness.{wrapper_note} Talk to the person in the language their request is written in, and quote each line the commands print as it printed it.

1. Run this shell command once. Put the request in single quotes, copying the text after {command} exactly, character for character: do not translate, correct or rephrase it. Write each single quote inside it as '\'':

       {wrapper} evolve '<the request>'

   It prints the request id and a link to the request's page, where the step shows while it runs. Give the person the link.

2. Run `{wrapper} wait <request id> --timeout 100 --poll` with the id it printed. Its --timeout counts seconds. {shell_timeout} While it prints `no result yet` the step still runs (a step usually takes a few minutes): say so in one line and run it again.

3. Tell the person the result line it printed. When it prints a `how to use:` line, tell the person how to use the new version in the words of that line, never in a form taken from the request. When it names a release to install, ask whether to install it now. On yes run `{wrapper} update`: the new version takes effect when the person starts reef-{adapter} again. If update says the release requires setup first, show the items it printed and tell the person to run `reef-{adapter} setup` in a terminal, then `reef-{adapter} update`.

4. When the result line says nothing changed, or the step failed, quote the reason it gives, give the person the request page link again for the details, and offer to file the same request once more. Do not guess another cause and do not suggest rewording the request unless the result line itself says so.
{absent}