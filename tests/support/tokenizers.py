"""A tokenizer fake for the chat handlers that render a prompt and decode sampled tokens."""


class FakeTokenizer:
    """A tokenizer with one fixed prompt: the text ``<user></user><assistant>`` or the token IDs ``[10, 11]``.

    ``rendered_messages`` is the (messages, template arguments) pair of the last call that asked for token IDs,
    or None before the first one. ``decode`` writes each token ID ``n`` as ``<n>``.
    """

    def __init__(self) -> None:
        self.rendered_messages = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert add_generation_prompt is True
        if not tokenize:
            return "<user></user><assistant>"
        self.rendered_messages = (messages, kwargs)
        return [10, 11]

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "".join(f"<{token_id}>" for token_id in token_ids)
