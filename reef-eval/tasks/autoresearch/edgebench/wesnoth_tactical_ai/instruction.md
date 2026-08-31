## Wesnoth Tactical AI

Write a Lua AI for The Battle for Wesnoth 1.18 that wins turn-based multiplayer matches against the default AI.

---

## Goal

Maximize **total_score = 80 x win_rate + 20 x avg_efficiency**, evaluated over 3 maps x N seeds (default 10) x N repeats (default 2) as side 1 (your AI) vs side 2 (default Wesnoth AI).

- `win_rate` = wins / total_games
- `avg_efficiency` = mean(min(1, baseline_turns / actual_turns)) over wins only

Baseline turn counts (default vs default): `Weldyn_Channel=16`, `The_Freelands=15`, `Fallenstar_Lake=20`.

Note: The baseline AI (default C++ CAs) typically scores 0-20. Focus on incremental improvements.

---

## API Restrictions (CRITICAL — violation = score=0 and hack flag)

Forbidden APIs (not for AI decisions):
- `wesnoth.wml_actions.*` — scene design API (WML actions like kill, message, etc.), NOT for AI decisions
- `wesnoth.game_events.*` — can fire arbitrary WML events, NOT for AI decisions
- `os.execute` — system command execution
- `io.open` — file system access
- `debug.*` — can bypass sandbox restrictions

Allowed APIs (AI decision APIs):
- `wesnoth.units.*` — unit queries (find_on_map, etc.)
- `wesnoth.current.*` — current game state
- `wesnoth.sides.*` — side information (gold, recruit list, villages, etc.)
- `wesnoth.map.*` — map queries
- `wesnoth.interface.*` — chat messages, UI
- `wesnoth.ai.*` — AI aspects and actions

Using forbidden APIs or attempting to bypass restrictions (e.g., string concatenation, metatable manipulation, variable indirection) will result in score=0 and the submission marked as hack.

---

## Files you must write in `/home/wesnoth_agent/`

1. `my_custom_ai.cfg` — AI definition (WML). Must include `id=my_custom_ai`, a `[stage]`, and `[candidate_action]` blocks.
2. Any number of `*.lua` files referenced by `location=...` in the cfg.

A baseline `my_custom_ai.cfg` already exists (uses default C++ CAs). Improve it.

---

## Wesnoth CA configuration (key rules from docs)

- Config file must have `id=` field, else wesnoth skips it with "skipped AI config due to missing id".
- CAs must be inside `[stage]` with `name=ai_default_rca::candidate_action_evaluation_loop`.
- For a Lua CA, use `engine=lua` + `location="~add-ons/my_ai/my_recruit.lua"`. The Lua file must `return` a table whose `:evaluation()` and `:execution()` methods drive the CA.
- Do NOT mix `location` with inline `evaluation=`/`execution=`.
- `--ai-config` does not accept absolute paths; only wesnoth-relative or `~add-ons/...`.
- Do NOT use `[kill]`, `[event]`, or `[command]` WML tags in your .cfg file — these are scene design tools, not AI decision mechanisms.

### Example Lua CA (place in `my_recruit.lua`)
```lua
local my_recruit = {}
function my_recruit:evaluation()
  local side = wesnoth.current.side
  local leader = wesnoth.units.find_on_map({ side = side, canrecruit = true })[1]
  if not leader or wesnoth.sides[side].gold < 14 then return 0 end
  return 300000
end
function my_recruit:execution()
  local side = wesnoth.current.side
  local recruit_types = wesnoth.sides[side].recruit
  if recruit_types and #recruit_types > 0 then
    wesnoth.interface.add_chat_message('my_ai', 'Recruiting ' .. recruit_types[1])
  end
end
return my_recruit
```

---

## Local Testing

Run a quick game locally to verify your AI loads:
```bash
wesnoth --nogui --multiplayer --nosound \
  --controller 1:ai --controller 2:ai \
  --ai-config 1:~add-ons/my_ai/my_custom_ai.cfg \
  --scenario multiplayer_Weldyn_Channel \
  --log-debug=ai/testing
```

---

## Rules
- Write your AI as `my_custom_ai.cfg` and optional `*.lua` files in `/home/wesnoth_agent/`
- Do NOT modify files outside `/home/wesnoth_agent/`
- Do NOT delete `.config/` or `.cache/` directories — the evaluation system handles add-ons installation automatically
- Your AI plays as side 1
- Run your AI locally to verify it loads before finishing
- Submit for evaluation

## Strategy Hints
- The default AI is decent at combat but poor at strategic recruit selection and village capture timing.
- Lua CAs give you full control over move ordering, targeting, and recruitment.
- Focus on: (1) efficient recruitment, (2) village capture, (3) focus-fire kills, (4) leader safety.
- Read `my_custom_ai.cfg` for the baseline structure.
