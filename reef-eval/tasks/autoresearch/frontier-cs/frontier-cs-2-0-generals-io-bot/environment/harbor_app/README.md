# Generals.io Bot Arena

![Generals.io gameplay](https://raw.githubusercontent.com/strakam/generals-bots/master/generals/assets/gifs/wider_gameplay.gif)

Image credit: `strakam/generals-bots`, MIT license.

Implement a bot for a local Generals.io-style two-player arena. The online
generals.io service is not used.

## Game Setting

Each game is a square-grid game with fog of war. The default Frontier-CS setting
uses 10x10 maps, a 180-turn truncation limit, and one game per baseline matchup
so that submissions return feedback quickly.

Your goal is to protect your own general and capture the opponent's general as
often and as quickly as possible. If neither general is captured by the
truncation limit, the game is a draw for win-rate scoring.

The simulator rules are:

- The grid contains empty passable cells, impassable mountains, neutral cities,
  and one general per player.
- You observe every cell in the 3x3 neighborhood around your owned cells.
  Everything else is fogged, except fogged cities/mountains may appear as
  `structures_in_fog` obstacles.
- Every turn, each player returns one action:

```text
[pass, row, col, direction, split]
```

- Directions are `0=up`, `1=down`, `2=left`, `3=right`.
- If `pass` is true, the bot does nothing. Otherwise `(row, col)` must be an
  owned source cell and the destination must be adjacent and passable.
- If `split == 0`, the move sends `source_army - 1` armies and leaves one
  behind. If `split == 1`, the move sends `source_army // 2` armies.
- Moving into your own cell reinforces it. Moving into a neutral or enemy cell
  captures it only when the moving army is strictly larger than the defending
  army. The remaining army is the absolute difference.
- Capturing the enemy general immediately wins.
- Army growth is deterministic: every owned cell gains one army when
  `timestep % 50 == 0`, and owned generals/cities gain one army when
  `timestep % 2 == 1`.

Your `act(observation, key)` receives a `generals.core.observation.Observation`
with these fields:

```text
armies, generals, cities, mountains, neutral_cells, owned_cells,
opponent_cells, fog_cells, structures_in_fog, owned_land_count,
owned_army_count, opponent_land_count, opponent_army_count, timestep
```

All spatial fields have shape `(H, W)`. Boolean masks use `True` for presence;
`armies` is zero in fog.

## Workflow

Work in `/app/generals_agent`, then run:

```bash
bash /app/make_submission.sh
bash /app/submit.sh
```

Submit the baseline skeleton once before running long local experiments, then
submit every meaningful improvement. Local simulations are useful only for tiny
sanity checks; the black-box judge is the scoring feedback for this task, and
submissions run asynchronously while you keep improving.

Submission is asynchronous. Use:

```bash
bash /app/submissions.sh
bash /app/wait_submission.sh <submission_uuid>
```

to inspect judge results while continuing to improve the bot.
