## DCSS Dungeon Crawl Stone Soup AI

Write a Lua bot for DCSS (Dungeon Crawl Stone Soup, version 0.28) that maximizes game score, playing a Minotaur Berserker.

---

## How you're scored

- Holdout: 5 hidden seeds, **900s game budget per game**, mean score across all 5 games. The 5 games run **in parallel**, so a full submission is about 15-18min plus overhead, and may finish earlier if games die or quit early.
- Start combo is fixed for determinism: Minotaur Berserker with `weapon=mace`. The evaluator forces this with Crawl `-extra-opt-last weapon=mace`, so repeated unchanged submissions are not supposed to sample a different starting weapon.
- **There is no turn limit.** Wall-time is the only constraint.
- **You don't need to design death/termination.** At the 900s game-budget deadline, the eval automatically sends `Ctrl+G` + `Esc`x2 + `Ctrl+Q` + `yes` + `Enter` via the screen session, which makes crawl write a morgue regardless of game state (it forces an interrupt + abandon-character prompt). Your character ends with whatever XP/depth/items they accumulated; that's your score. Just play to maximize what you get in 900s. `crawl.millis()` returns a monotonic millisecond timestamp; if you use it, compare `crawl.millis() - start_ms`. The evaluator already handles end-of-game termination, so use `you.turns()` for turn-based bot logic unless you specifically need wall-clock self-pacing.
- A game's score = the integer on line 5 of `morgue-EvalN-*.txt`, which is DCSS's own score formula (XP + depth + items + kills + ...).
- Total eval is roughly 15-18min/submission because the five 900s games run in parallel; it is not `5 games x 3min`.

## Two levers, both matter

```
score ≈ turn_count × score_per_turn
      = (turn_rate × 900s) × score_per_turn
```

- **turn_rate** = how many game-turns you advance per second of wall-time. Bounded by PTY/screen overhead and which keys you send.
- **score_per_turn** = how much score each turn produces on average. Bounded by strategy: depth reached, monsters killed, items used, neutralized risks.

Neither alone is enough. High turn-rate while resting safely on D:1 = score 0. Perfect combat at 0.5 turn/s = ~450 turns in 900s, still shallow.

## High-turn-rate operations (free time-skipping)

DCSS folds multiple game-turns into a single Lua hook invocation when you use batch commands:

| Key / command | Game-turns per call | Use case |
|---|---|---|
| `crawl.sendkeys("h")` | 1 | Single-step move (slow) |
| `crawl.do_commands({"CMD_WAIT"})` | 1 | Skip 1 turn |
| `crawl.sendkeys("o")` | ~10-50 | Autoexplore current level |
| `crawl.sendkeys("G>")` | ~50-200 when it works | Autotravel toward deeper levels; in console it can leave a `Where to?` prompt if used from the wrong state |
| `crawl.do_commands({"CMD_REST"})` or `crawl.sendkeys("5")` | up to 100 | Long-rest until full HP/MP or interrupted |
| `crawl.sendkeys("G7")` etc. | hundreds | Autotravel to a specific level |

Stair handling is a common bottleneck. `view.feature_at(0, 0)` only describes the tile you are standing on; downstairs features are typically named like `stone_stairs_down_i`, `stone_stairs_down_ii`, `stone_stairs_down_iii`, or `escape_hatch_down`. When already on a downstairs tile, `crawl.do_commands({"CMD_GO_DOWNSTAIRS"})` or `crawl.sendkeys(">")` is usually the direct descent. When stairs are visible but not under you, walk onto them first or use carefully tested travel. If a morgue shows repeated `Where to?`, `What level of the Dungeon?`, `Okay, then`, or `You're already here!`, your travel command is stuck in a prompt loop and is not advancing gameplay.

Baseline uses `o` + single-step combat → ~1-2 turn/s. A bot that chains autotravel + long-rest + autofight can hit 10-50 turn/s.

## Files you edit in `/home/dcss_agent/`

1. `my_bot.rc` — DCSS options + `include = my_bot_include.rc`.
2. `my_bot_include.rc` — Lua block. Defines `ready()` (called every turn) and `c_message(text, channel)` (catches prompts).

RC/Lua syntax pitfall: the outer `{ ... }` in `my_bot_include.rc` is a DCSS rc Lua block delimiter. Avoid putting a Lua table-constructor closing brace `}` alone on its own line inside that block; the rc parser can treat it as the end of the Lua block, causing confusing `near '<eof>'` Lua errors. Prefer compact table endings like `["foo"] = 1 }` or otherwise keep nested table braces away from standalone `}` lines.

Both have a baseline committed to git. If you wreck things: `git checkout HEAD -- my_bot.rc my_bot_include.rc`.

Before holdout submissions, make sure the current bot still runs locally. If `dcss-eval` returns all-zero scores, a turn-1 quit, repeated `Unknown command` / `Where to?` / `Okay, then`, or Lua/RC errors, keep debugging locally before submitting.

## Iteration loop

Required cadence for long runs:

1. First, inspect the two bot files and run one short local `dcss-eval` sanity check with any timeout you choose.
2. If the current files run without syntax errors or immediate turn-1 quit, start a holdout submission in the background early, even if it is only the baseline. Use a command like `sforge-submit --details > /tmp/holdout_round1.log 2>&1 &` and keep debugging locally while it runs. Do not spend 20+ minutes local-only without any holdout submission in flight.
3. When the holdout result returns, inspect `/tmp/holdout_round1.log` or run `sforge-submit --list`, then decide whether the current local idea is actually better before submitting it. For the second and later holdout submissions, require a real gameplay change plus clear local evidence across several public/random seeds. If local score collapses, cases only quit on turn 1, or the morgue shows prompt loops / repeated unknown commands, keep debugging locally instead of submitting.

- **`dcss-eval`** at `/usr/local/bin/dcss-eval` — local evaluator, your seeds, your params, same suicide-mechanism as the judge:
  ```bash
  dcss-eval --n 1 --timeout 30      # very short sanity check
  dcss-eval --n 5 --parallel 5 --timeout SECONDS # broader local check; choose any debug timeout
  dcss-eval --seeds 7,42,99
  dcss-eval --random --n 5          # 5 random seeds from 1..100
  dcss-eval --show-morgue           # dump death cause + morgue tail
  ```
  Local seed pool is 1..100, disjoint from the holdout. The holdout seed set and start weapon are fixed but hidden/controlled; repeated unchanged submissions are broadly comparable and cannot sample new maps or weapons. Wall-clock/PTTY scheduling can still create small score noise, so use submissions to measure real gameplay changes, not to gamble on unchanged or comment-only reruns. The local `--timeout` value is arbitrary and only controls your debugging runtime; it can be short or long as needed and does not reveal hidden seeds or change the judge. The holdout evaluator uses 900s per game. Optimizing on a few specific seeds gives **no signal** about holdout — rotate seeds, target robustness.

- **`sforge-submit`** — submit to the holdout. About 15-18min round-trip because 5 games run in parallel at 900s/game. Run it in the background when possible and keep using `dcss-eval` plus code edits while it runs; do not block the whole session on one holdout unless you specifically need that result before choosing the next change. The submit UI reports the current submission `Score`, `Pass rate`, `Passed`, and metrics with each hidden case's morgue score; `--details` also shows hidden case IDs (`case_0000`..`case_0004`), pass/fail status, and each hidden case's morgue score, but never seed values. `Pass rate`/`Passed` comes from the judge CASE OK count: it only means how many games produced accepted morgues, so for this score task it is a validity check, not the quality metric. Optimize `Score`, which comes from `TOTAL_SCORE`. Do not spend time resubmitting unchanged or comment-only code: the holdout is fixed, so identical gameplay should score nearly the same apart from small scheduling noise, and only DCSS's morgue score is counted. Comments, `crawl.mpr`/`crawl.take_note` notes, API probes, and textual claims do not add score. Remove temporary probe/debug code such as `INVPROBE` before submitting; it can only add noise or slow the bot down.

## DCSS Lua API quick reference

- `you.turns()`, `you.hp()`, `you.where()`, `you.xl()`, `you.god()`
- `crawl.do_commands({"CMD_AUTOFIGHT", "CMD_REST", "CMD_WAIT", "CMD_GO_DOWNSTAIRS"})` — named commands
- `crawl.sendkeys("o")` — raw keys (`o`=autoexplore, `>`=descend when standing on downstairs, `<`=ascend, `5`=long-rest, `G>`/`G7`=autotravel but can enter travel prompts if misused, `Ctrl+F`=find)
- `crawl.millis()` — monotonic millisecond timestamp; compare deltas from a captured value; not needed for termination
- `view.feature_at(dx, dy)` — terrain string at offset
- `monster.get_monster_at(x, y)` — adjacent/visible monster; `:is_firewood()`, `:attitude()`, and `:name()` are useful. Avoid assuming every apparent monster method exists; verify with `dcss-eval --show-morgue` because a missing Lua method stops progress.
- Trog abilities can be driven with `crawl.do_commands({"CMD_USE_ABILITY"})` plus `chk_ability_choice = "a"` for the first ability, but ability-menu APIs vary in Crawl 0.28. Test any ability code with `dcss-eval --show-morgue`; a Lua error in the morgue means that game stopped making progress.
- `crawl.mpr("msg")` — log to messages, captured in morgue Notes

Docs in container: `/usr/share/crawl/docs/options_guide.txt`, `/usr/share/crawl/docs/keybind.txt`, `/usr/share/crawl/dat/clua/*.lua`, `/usr/share/crawl/dat/dlua/*.lua`.

## Banned (auto-zero)

- **Wizard mode / explore mode**: pressing `&` or `Ctrl+X` opens god-mode. Judge greps morgue for `Entered (wizard|explore) mode|WIZARD|EX-WIZARD` and zeroes the game. Local dcss-eval enforces the same.
- Modifying `crawl` binary or eval scripts (not in `submit_paths` anyway).

Death-by-monster is the intended risk, not a thing to engineer around.