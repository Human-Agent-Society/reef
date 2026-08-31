# Generals Agent Skeleton

Edit `bot.py` and implement `FrontierAgent`.

The judge-side arena is black-box. This workspace contains the bot skeleton and
the public `generals-bots` package, but no Frontier-CS evaluation harness,
baseline ensemble, seeds, or match runner.

## Goal

Protect your own general and capture the opponent's general as often and as
quickly as possible. The final score combines win rate against several fixed
baseline bot families with a speed tiebreaker for faster wins.

## Game Rules

The task uses the local `generals-bots` simulator, not the online generals.io
service.

- The game is played on a square grid. The default Frontier-CS setting is 10x10
  with a 180-turn truncation limit.
- The grid contains passable empty cells, impassable mountains, neutral cities,
  your general, and the opponent's general.
- You observe every cell in the 3x3 neighborhood around your owned cells.
  Other cells are fogged. Fogged cities/mountains may appear as
  `structures_in_fog` obstacles.
- Each turn both players choose one action. Invalid actions become no-ops.
- Moving into your own cell reinforces it. Moving into a neutral or enemy cell
  captures it only when the moving army is strictly larger than the defending
  army. The remaining army is the absolute difference.
- Capturing the enemy general immediately wins. If neither general is captured
  before truncation, the game is a draw for win-rate scoring.
- Army growth is deterministic: every owned cell gains one army when
  `timestep % 50 == 0`, and owned generals/cities gain one army when
  `timestep % 2 == 1`.

Useful imports:

```python
import jax.numpy as jnp
from generals.core.action import create_action, compute_valid_move_mask_obs
from generals.agents import Agent
```

Action format:

```text
[pass, row, col, direction, split]
```

Directions:

```text
0 up, 1 down, 2 left, 3 right
```

If `pass` is true, the bot does nothing. Otherwise `(row, col)` is the owned
source cell. If `split == 0`, the move sends `source_army - 1` armies and leaves
one behind. If `split == 1`, the move sends `source_army // 2` armies.

The `observation` passed to `act(observation, key)` has these fields:

```text
armies, generals, cities, mountains, neutral_cells, owned_cells,
opponent_cells, fog_cells, structures_in_fog, owned_land_count,
owned_army_count, opponent_land_count, opponent_army_count, timestep
```

All spatial fields have shape `(H, W)`. Boolean masks use `True` for presence;
`armies` is zero in fog.

Create a patch submission:

```bash
bash /app/make_submission.sh
bash /app/submit.sh
```

Submit the baseline skeleton once before running long local experiments, then
submit every meaningful improvement. Local simulations are useful only for tiny
sanity checks; the black-box judge is the scoring feedback for this task, and
submissions run asynchronously while you keep improving.

The judge accepts patches touching only `bot.py`, `strategy.py`, and `utils.py`.
Do not read files, launch subprocesses, open network sockets, or inspect
environment variables; the evaluator rejects these patterns.

The simulator is JAX-based. Compact array logic usually runs faster than large
Python-heavy policies.
