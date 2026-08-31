## Role

You are an expert OpenTTD AI script developer. Write a **Squirrel** AI in `my_ai/` that maximizes **company value** over 20 game-years, averaged across multiple random seed maps.

---

## Files

- `my_ai/info.nut` — AI metadata (class `MyAI extends AIInfo`, `GetAPIVersion()` returns "13")
- `my_ai/main.nut` — AI controller (class `MyAI extends AIController`, implements `Start()`, `Save()`, `Load()`)
- `docs/squirrel_reference.html` — Squirrel 2.2 language reference (offline)
- `docs/noai_tutorial.html` — NoAI introductory tutorial (offline)
- `docs/noai_api_docs/` — full NoAI API (115 classes, offline HTML)

---

## Game Background

OpenTTD (Transport Tycoon Deluxe clone) simulates a transport company: build roads / rails / airports / ships, connect towns and industries, buy vehicles, manage finances. The NoAI framework runs Squirrel scripts to control a company headlessly. Your AI is the only company on the map (no competitors).

## Goal

Maximize **`AICompany.GetCompanyValue(AICompany.COMPANY_SELF)`** (= vehicle value at 1.5× purchase + infrastructure value + cash − loans). Use `AILog.Info(...)` / `AILog.Warning(...)` for debugging.

## Local Testing

Install OpenTTDLab (already installed). Run locally with:

```python
from openttdlab import run_experiments, local_folder
results = run_experiments(openttd_version='13.4', opengfx_version='7.1',
    experiments=({'seed': s, 'ais': (local_folder('my_ai/', ai_name='MyAI'),),
                  'days': 365*20} for s in range(5)))
```

## Rules

- Write only in `my_ai/` (info.nut + main.nut + optional sub-squirrel files)
- `CreateInstance()` in info.nut must return "MyAI" (matching main.nut's class name)
- `GetAPIVersion()` must return `"13"` for OpenTTD 13.x compatibility
- Do NOT modify anything under `docs/` — it's reference material
- A baseline AI that just logs company value is already in `my_ai/`; improve it
