"""Exploitable-poker judge: one deterministic hand behind a sidecar.

Replays CL-Bench's exploitable_poker mechanics (Apache-2.0) for a single
hand. The deck and button are dealt exactly as upstream (global-RNG seed
plus burn, verified against the upstream harness), the opponent is the
vendored deterministic policy, and the reward is chip profit divided by
the big blind. The deck and both hands live in this container only; the
agent gets state as text over HTTP.

Endpoints: ``GET /state`` — ``POST /act`` {"action", "amount"} (invalid
actions leave state unchanged and cost nothing, as upstream) —
``GET /final``, finishing an unfinished hand check/fold — ``GET /health``.
"""

import json
import os
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from texasholdem import TexasHoldEm
from texasholdem.game.action_type import ActionType
from texasholdem.game.player_state import PlayerState

sys.path.insert(0, str(Path(__file__).parent))
import poker_opponents  # noqa: E402 — lives beside this file in the judge image

STARTING_CHIPS, SMALL_BLIND, BIG_BLIND = 1000, 5, 10

_lock = threading.Lock()


def deal_hand(config: dict) -> TexasHoldEm:
    """Deal the hand exactly as upstream: seed the stage, burn earlier hands."""
    random.seed(int(config["stage_seed"]))
    for _ in range(int(config["burn"])):
        TexasHoldEm(
            buyin=STARTING_CHIPS,
            big_blind=BIG_BLIND,
            small_blind=SMALL_BLIND,
            max_players=2,
        ).start_hand()
    game = TexasHoldEm(
        buyin=STARTING_CHIPS,
        big_blind=BIG_BLIND,
        small_blind=SMALL_BLIND,
        max_players=2,
    )
    game.start_hand()
    return game


class Hand:
    def __init__(self, config: dict):
        self.config = config
        self.policy = poker_opponents.get_opponent_policy(config["variant"])
        self.game = deal_hand(config)
        self.bot_actions: list[str] = []
        self.agent_bet_this_street = 0
        self.finalized = False
        self.run_bot()

    # ---- upstream mechanics ----

    def run_bot(self) -> None:
        while self.game.is_hand_running() and self.game.current_player == 1:
            action, total = self.policy(self.game)
            self.bot_actions.append(action.name)
            if total is not None:
                self.game.take_action(action, total=total)
            else:
                self.game.take_action(action)

    @property
    def done(self) -> bool:
        return not self.game.is_hand_running()

    def act(self, verb: str, amount: int | None) -> str | None:
        """Apply the agent's action; returns an error message if invalid."""
        try:
            action = ActionType[verb.upper()]
        except KeyError:
            return f"Invalid poker action: unknown action '{verb}'"
        if action not in (
            ActionType.FOLD,
            ActionType.CALL,
            ActionType.CHECK,
            ActionType.RAISE,
        ):
            return f"Invalid poker action: '{verb}' is not available"
        try:
            if action == ActionType.RAISE:
                if amount is None:
                    return (
                        "Invalid poker action: RAISE needs an amount (raise TO total)"
                    )
                self.game.validate_move(0, action, total=int(amount), throws=True)
            else:
                self.game.validate_move(0, action, throws=True)
        except Exception as error:  # noqa: BLE001 — engine's reason, verbatim
            return (
                f"Invalid poker action: {error}\n"
                "The hand state was not changed. Choose one of the legal "
                "actions for the current betting state."
            )
        street = self.game.hand_phase
        if action == ActionType.RAISE:
            self.game.take_action(action, total=int(amount))
        else:
            self.game.take_action(action)
        self.agent_bet_this_street = (
            0 if self.game.hand_phase != street else self.agent_bet_this_street
        )
        self.bot_actions = []
        self.run_bot()
        return None

    def finish_check_fold(self) -> None:
        while self.game.is_hand_running():
            if self.game.current_player == 0:
                try:
                    self.game.validate_move(0, ActionType.CHECK, throws=True)
                    self.game.take_action(ActionType.CHECK)
                except Exception:  # noqa: BLE001
                    self.game.take_action(ActionType.FOLD)
            else:
                self.run_bot()

    # ---- rendering, following the upstream prompt ----

    @staticmethod
    def _cards(cards) -> str:
        parts = []
        for card in cards:
            pretty = getattr(card, "pretty_string", None)
            parts.append(pretty if isinstance(pretty, str) else str(card))
        return " ".join(parts)

    def pot_total(self) -> int:
        return sum(pot.get_total_amount() for pot in self.game.pots)

    def state_text(self) -> str:
        game = self.game
        phase = game.hand_phase.name
        position = "Button / small blind" if game.btn_loc == 0 else "Big blind"
        hand_str = self._cards(game.hands[0])
        board = self._cards(game.board) if game.board else "No cards yet"
        chips_to_call = game.chips_to_call(0)
        min_raise_total = game.value_to_total(game.min_raise(), 0)
        player = game.players[0]

        if not self.bot_actions:
            bot_str = ""
        elif len(self.bot_actions) == 1:
            bot_str = f"\nOpponent's action: {self.bot_actions[0]}"
        else:
            bot_str = f"\nOpponent's actions: {' -> '.join(self.bot_actions)}"

        bot_bet = game.player_bet_amount(1)
        if player.state == PlayerState.TO_CALL:
            verb = "raised to" if game.player_bet_amount(0) > 0 else "bet"
            situation = f"Opponent {verb} {bot_bet} (you need {chips_to_call} to call)"
        elif self.bot_actions and self.bot_actions[-1] == "CALL":
            situation = (
                f"Opponent called your bet of {game.player_bet_amount(1)} "
                "(you can check or raise)"
            )
        else:
            situation = "Action to you (you can check or raise)"

        return f"""Hand #{self.config["hand_number"]} - {phase}
Table: Heads-up Texas Hold'em (2 players)
Opponent: {self.config["opponent_name"]}
Your position: {position}
Your hand: {hand_str}
Board: {board}
Pot: {self.pot_total()} chips
Your chips: {player.chips}{bot_str}

Situation: {situation}

What's your action?
  FOLD - fold your hand
  CALL - call the bet (costs {chips_to_call} chips)
  CHECK - check (if no bet to call)
  RAISE X - raise to X total (min raise to {min_raise_total})"""

    def result(self) -> dict:
        delta = self.game.players[0].chips - STARTING_CHIPS
        outcome = "WON" if delta > 0 else ("LOST" if delta < 0 else "TIED")
        showdown = ""
        folded = any(p.state == PlayerState.OUT for p in self.game.players)
        if not folded:
            showdown = (
                f"\nShowdown — Board: {self._cards(self.game.board)}"
                f"\nYour hand: {self._cards(self.game.hands[0])}"
                f"\nOpponent's hand: {self._cards(self.game.hands[1])}"
            )
        return {
            "report": (
                f"Hand {self.config['hand_number']} complete: You {outcome}!{showdown}\n\n"
                f"Ending stack: {self.game.players[0].chips} chips\n"
                f"Net chip change this hand: {delta:+d} chips"
            ),
            "net_chips": delta,
            "reward": round(delta / BIG_BLIND, 6),
        }


class Handler(BaseHTTPRequestHandler):
    def _reply(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._reply({"ok": True})
        with _lock:
            if self.path == "/state":
                if HAND.done:
                    return self._reply({"done": True, **HAND.result()})
                return self._reply({"done": False, "prompt": HAND.state_text()})
            if self.path == "/final":
                if not HAND.done:
                    HAND.finish_check_fold()
                HAND.finalized = True
                return self._reply(HAND.result())
        return self._reply({"error": "unknown endpoint"}, 404)

    def do_POST(self):
        if self.path != "/act":
            return self._reply({"error": "unknown endpoint"}, 404)
        try:
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
            )
        except (json.JSONDecodeError, ValueError):
            return self._reply({"error": "body must be JSON"}, 400)
        with _lock:
            if HAND.finalized or HAND.done:
                return self._reply({"done": True, **HAND.result()})
            error = HAND.act(str(body.get("action", "")), body.get("amount"))
            if error:
                return self._reply({"done": False, "error": error})
            if HAND.done:
                return self._reply({"done": True, **HAND.result()})
            return self._reply({"done": False, "prompt": HAND.state_text()})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    judge_dir = Path(os.environ.get("JUDGE_DIR", Path(__file__).parent))
    port = int(os.environ.get("PORT", "8082"))
    HAND = Hand(json.loads((judge_dir / "hand_config.json").read_text()))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
