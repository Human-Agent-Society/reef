"""The mailboxes of one team stage run: a queue per member, and the messages sent to the agent that started the team.

A message is queued when it is sent and read at the receiver's next step; a member that has ended receives nothing
more, and a message to it is reported undelivered. The inbox lasts one stage run.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass

from reef.harness.runners.native.control import EpisodeStop
from reef.harness.tree.nodes import redact_secret_shaped

#: Characters one message may carry, and messages one member may send in a stage run.
TEAM_MESSAGE_MAX_CHARS = 8000
TEAM_MAX_SENDS_PER_MEMBER = 256
#: The longest a wait sleeps before it reads the stop flag again.
WAIT_SLICE_SECONDS = 1.0


@dataclass(frozen=True)
class TeamMessage:
    message_id: str
    sender: str
    text: str


class Inbox:
    """One stage run's mailboxes; safe to use from every member's thread."""

    def __init__(self, members: Mapping[str, str], caller: str, stop: EpisodeStop) -> None:
        #: Each member's instance name and the role (the agent) it runs.
        self.members = dict(members)
        self.caller = caller
        self.stop = stop
        self.condition = threading.Condition()
        self.queues: dict[str, list[TeamMessage]] = {member: [] for member in self.members}
        self.closed: set[str] = set()
        self.sent_counts: dict[str, int] = dict.fromkeys(self.members, 0)
        self.to_caller: list[TeamMessage] = []

    def recipients(self, sender: str, to: str) -> list[str]:
        """Who ``to`` names: a member, ``all`` other members, the caller, or every other member of a role."""
        roles = sorted(set(self.members.values()))
        if to == sender:
            raise ValueError("a member cannot send a message to itself")
        if to in self.members or to == self.caller:
            return [to]
        if to == "all" or to in roles:
            return [member for member, role in self.members.items() if member != sender and to in ("all", role)]
        raise ValueError(
            f"no member, role or caller is named {to!r}; members: {', '.join(self.members)}; "
            f"roles: {', '.join(roles)}; all; caller: {self.caller}"
        )

    def send(self, sender: str, to: str, text: str) -> tuple[TeamMessage, list[str], list[str]]:
        """Queue one message; (the message as queued, who got it, who has ended and did not)."""
        if len(text) > TEAM_MESSAGE_MAX_CHARS:
            raise ValueError(
                f"a message carries at most {TEAM_MESSAGE_MAX_CHARS} characters; this one has {len(text)}"
            )
        recipients = self.recipients(sender, to)
        with self.condition:
            if self.sent_counts[sender] >= TEAM_MAX_SENDS_PER_MEMBER:
                raise ValueError(f"a member sends at most {TEAM_MAX_SENDS_PER_MEMBER} messages in a stage run")
            self.sent_counts[sender] += 1
            # Redacted before it is queued, so neither file ever holds the credential.
            message = TeamMessage(f"{sender}-{self.sent_counts[sender]}", sender, redact_secret_shaped(text))
            delivered, undelivered = [], []
            for recipient in recipients:
                if recipient in self.closed:
                    undelivered.append(recipient)
                elif recipient == self.caller:
                    self.to_caller.append(message)
                    delivered.append(recipient)
                else:
                    self.queues[recipient].append(message)
                    delivered.append(recipient)
            self.condition.notify_all()
        return message, delivered, undelivered

    def take(self, member: str) -> list[TeamMessage]:
        """The messages waiting for ``member``, in the order they were sent; they are no longer waiting after."""
        with self.condition:
            waiting, self.queues[member] = self.queues[member], []
        return waiting

    def is_alone(self, member: str) -> bool:
        """Whether every other member has ended, so no message can come; the caller is waiting and sends none."""
        with self.condition:
            return all(other in self.closed for other in self.members if other != member)

    def wait(self, member: str, timeout_seconds: float) -> int:
        """Wait until a message waits for ``member``, the stop flag is set, every other member has ended, or the time
        passes; the number of messages waiting then."""
        deadline = time.monotonic() + timeout_seconds
        with self.condition:
            while True:
                waiting = len(self.queues[member])
                remaining = deadline - time.monotonic()
                if waiting or self.stop.is_set or self.is_alone(member) or remaining <= 0:
                    return waiting
                self.condition.wait(min(remaining, WAIT_SLICE_SECONDS))

    def close(self, member: str) -> list[TeamMessage]:
        """End ``member``'s mailbox; the messages it never read."""
        with self.condition:
            self.closed.add(member)
            unread, self.queues[member] = self.queues[member], []
            self.condition.notify_all()
        return unread

    def caller_messages(self) -> list[TeamMessage]:
        with self.condition:
            return list(self.to_caller)


@dataclass(frozen=True)
class TeamMember:
    """What a member's run reads of its team: who it is and the stage run's inbox."""

    instance: str
    role: str
    inbox: Inbox
