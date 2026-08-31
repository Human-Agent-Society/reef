## NetHack Dungeon Adventure Agent

Write `agent.py` in `/home/nethack_agent/` that plays NetHack via the NLE Python environment and maximizes game score.

---

## Problem

NetHack is a classic hardcore Roguelike dungeon crawler (1987). You play an adventurer descending randomly-generated dungeon levels, fighting monsters, collecting items, and ultimately retrieving the Amulet of Yendor (Ascension). Every death permanently ends the game.

**Goal:** play effectively by surviving, exploring, fighting monsters, collecting useful items, and descending deeper into the dungeon.

---

## Interface

Your `agent.py` **must** export a function:

```python
def get_action(obs: dict, info: dict, step: int) -> int:
    """Return an integer action for the current game state."""
    ...
```

The evaluation system calls `get_action(obs, info, step)` each step and feeds the returned action into the NLE environment. **Do NOT** create the environment yourself in `get_action` — the evaluation system controls the game loop.

You may also include an `if __name__ == "__main__":` block for local testing (see skeleton below).

---

## Environment

- Python 3.10
- `nle==1.2.0` (NetHack Learning Environment, Gymnasium-style)
- `gymnasium`
- **Note:** `nle-language-wrapper` is NOT available. Use the `nle` native API with integer actions and numeric observations.

### Observation Space

`obs` is a dict with the following keys (all numpy arrays):
- `glyphs` — (21, 79) integer array of tile glyph IDs (0–5991)
- `chars` — (21, 79) byte array of ASCII characters displayed on screen
- `colors` — (21, 79) integer array of colors (0–15)
- `specials` — (21, 79) integer array of highlight flags
- `blstats` — (27,) integer array of bottom-line stats such as position, attributes, health, energy, armor, experience, dungeon depth, inventory load, hunger, alignment, and conditions.
- `message` — (256,) byte array of the last game message
- `inv_glyphs`, `inv_strs`, `inv_letters`, `inv_oclasses` — inventory info

### Action Space

Actions are integers in [0, 120]. The NLE action space maps action indices to NetHack key codes. **Use the `nle.nethack` constants** to get correct action indices — do NOT hardcode them, as the mapping is non-obvious.

```python
from nle.nethack import (
    ACTIONS, CompassDirection, CompassDirectionLonger,
    MiscDirection, MiscAction, Command,
)

ACTIONS_LIST = list(ACTIONS)
def action_index(key_code):
    return ACTIONS_LIST.index(key_code)
```

Common action indices (verified for nle==1.2.0):

| Action | Constant | Index |
|--------|----------|-------|
| north | CompassDirection.N | 0 |
| east | CompassDirection.E | 1 |
| south | CompassDirection.S | 2 |
| west | CompassDirection.W | 3 |
| northeast | CompassDirection.NE | 4 |
| southeast | CompassDirection.SE | 5 |
| southwest | CompassDirection.SW | 6 |
| northwest | CompassDirection.NW | 7 |
| run north | CompassDirectionLonger.N | 8 |
| run east | CompassDirectionLonger.E | 9 |
| run south | CompassDirectionLonger.S | 10 |
| run west | CompassDirectionLonger.W | 11 |
| go up stairs | MiscDirection.UP | 16 |
| go down stairs | MiscDirection.DOWN | 17 |
| wait | MiscDirection.WAIT | 18 |
| more/continue | MiscAction.MORE | 19 |
| kick | Command.KICK | 48 |
| pickup | Command.PICKUP | 61 |
| open | Command.OPEN | 57 |
| close | Command.CLOSE | 30 |
| eat | Command.EAT | 35 |
| search | Command.SEARCH | 75 |
| pray | Command.PRAY | 62 |
| apply | Command.APPLY | 24 |
| zap | Command.ZAP | 104 |
| throw | Command.THROW | 91 |
| fire | Command.FIRE | 40 |
| drop | Command.DROP | 33 |
| wield | Command.WIELD | 102 |
| wear | Command.WEAR | 99 |
| read | Command.READ | 67 |
| quaff | Command.QUAFF | 64 |
| esc | Command.ESC | 38 |

**Important:** The action indices are NOT sequential or intuitive. Always use `action_index(Command.XXX)` to get the correct index.

### Skeleton Code

```python
from nle.nethack import (
    ACTIONS, CompassDirection, CompassDirectionLonger,
    MiscDirection, MiscAction, Command,
)

ACTIONS_LIST = list(ACTIONS)
def action_index(key_code):
    return ACTIONS_LIST.index(key_code)

N = action_index(CompassDirection.N)
E = action_index(CompassDirection.E)
S = action_index(CompassDirection.S)
W = action_index(CompassDirection.W)
WAIT = action_index(MiscDirection.WAIT)
MORE = action_index(MiscAction.MORE)
UP = action_index(MiscDirection.UP)
DOWN = action_index(MiscDirection.DOWN)
PICKUP = action_index(Command.PICKUP)
OPEN = action_index(Command.OPEN)
CLOSE = action_index(Command.CLOSE)
EAT = action_index(Command.EAT)
SEARCH = action_index(Command.SEARCH)
PRAY = action_index(Command.PRAY)

def get_action(obs, info, step):
    return WAIT

if __name__ == "__main__":
    import argparse, gymnasium as gym, nle, numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5000)
    args = parser.parse_args()
    env = gym.make("NetHackChallenge-v0")
    try:
        env.seed(args.seed)
    except Exception:
        pass
    obs, info = env.reset(), {}
    if isinstance(obs, tuple):
        obs, info = obs[0], obs[1] if len(obs) > 1 else {}
    for step in range(args.max_steps):
        action = get_action(obs, info, step)
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        if done:
            break
    env.close()
```

---

## Evaluation

Your agent is evaluated by running it over multiple NetHack games under the official judge. Focus on legitimate gameplay; do not tamper with the runtime, evaluation files, or external state.

---

## Tips

- Start simple: random walk → explore aggressively → fight monsters → manage inventory
- Use `blstats`, `chars`, `message`, and inventory observations to infer the game state.
- `chars` array gives you the ASCII screen — parse it for terrain and monster info
- `message` tells you what just happened (e.g., "You kill the jackal!")
- Common strategy: explore rooms, pick up items, descend stairs to go deeper
- The game is extremely punishing — a simple survival strategy already scores well
- **Always use `nle.nethack` constants** for action indices, never hardcode them
