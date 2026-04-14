# LOCKED STATE — SMARTCHART S1 STRATEGY (WEBSITE + API v1)

S1 Strategy (EMA 14–20 Retest System) is now fully integrated across backend and website.

## Backend
- Strategy file: playbook/w_strategy_s1.py
- Built from truth_engine output
- EMA 14–20 retest required
- MFI cross (fast > slow above 0 for long, below 0 for short)
- EMA distance filter (cc_ema_distance_calibration)
- AI RSI directional bias
- MACD alignment and reversal filter
- Exhaustion filter

## API
- /strategy/s1/latest → full debug output
- /website/s1/latest → clean website output

## Website
- Elementor HTML widget connected to API
- Displays:
  - direction
  - grade
  - trade_ready
  - price
  - reason
  - entry_type
  - timestamp
- Card is centered and styled
- Status updates dynamically from backend

## STATUS
S1 is now LOCKED as baseline (v1).
No structural changes unless explicitly unlocked.

## NEXT PHASE
Build S1.5 Strategy (EMA 100–200 Retest System)
# LOCKED STATE — SMARTCHART S1.5 BUILD (PLAYBOOK v1)

S1.5 Strategy (EMA 100–200 Retest System) has been fully designed and coded.

## Strategy Logic (S1.5 v1)

### LONG
- EMA 100–200 retest required
- MFI fast crosses above slow
- Cross must occur above zero line
- MACD must be bullish (above zero)
- AI must be bullish

### SHORT
- EMA 100–200 retest required
- MFI fast crosses below slow
- Cross must occur below zero line
- MACD must be bearish (below zero)
- AI must be bearish

## Architecture

- Strategy file created:
  playbook/w_strategy_s1_5.py

- Uses truth engine as input
- Clean separation from S1
- No MTF / momentum / volatility / forecaster yet (deferred to v2)

## State Model

- blocked → no setup
- watch → EMA zone touched
- building → partial alignment
- ready → full confirmation

## Output Fields

- direction
- state
- grade
- trade_ready
- reason
- entry_type = ema_100200_retest

## STATUS

S1.5 logic is now LOCKED (v1)
File created but NOT yet connected to API

## NEXT PHASE

Connect S1.5 to API:

- /strategy/s1_5/latest
- /strategy/s1_5/table
- /website/s1_5/latest

Then connect to Elementor UI