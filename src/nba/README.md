# NBA Betting Tools

## Quick Start

```bash
# Live scores
python src/nba/data.py scoreboard

# Backtest: halftime projection
cd src && python -m nba.backtest_total_score --improved --alpha 0.0

# Backtest: Q4 projection
cd src && python -m nba.backtest_total_score --q4 --alpha 0.0

# Compare all adjustments side-by-side
cd src && python -m nba.backtest_total_score --compare

# High-confidence P&L simulation
cd src && python -m nba.backtest_total_score --q4 --alpha 0.0 --high-conf
```

## Total Score Model

Projects game totals at halftime or Q4 start using per-team rolling quarter averages (last 20 games, home/away split) with pace dampening.

### Core Formula

```
expected_H2 = home_q3_avg + home_q4_avg + away_q3_avg + away_q4_avg
pace = actual_H1 / expected_H1
dampened_pace = 1.0 + alpha * (pace - 1.0)
projected_total = H1_score + expected_H2 * dampened_pace
```

### Baseline Performance (622 games, 2025-11-24 to 2026-03-06)

| Checkpoint | MAE | Bias | RMSE | 1-sigma Coverage |
|---|---|---|---|---|
| Halftime | 11.1 | +0.2 | 14.0 | 78.0% |
| Q4 Start | 7.7 | -0.1 | 10.0 | 81.7% |

Naive baseline (2x halftime) has MAE 14.8 — model improves by ~3.7 pts.

## Win Rate Targets

- Q4 +10 margin OVER: ~85% win rate
- Halftime +18 margin OVER: ~90% win rate

## Adjustment Experiments (2026-03-09)

Tested three additional signals on top of the baseline model:

### 1. Opponent Defensive Adjustment

Scale expected remaining scoring by opponent's defensive strength vs league average.

```
opp_def_q34 = opponent's avg pts allowed in Q3+Q4
multiplier = clamp(opp_def_q34 / league_avg_q34, 0.85, 1.15)
```

**Result: Hurts.** MAE +0.1, adds -0.6 bias (halftime). The offensive rolling averages already implicitly capture matchup difficulty — if a team consistently faces tough defenses, their scoring averages reflect it. The defensive adjustment double-counts.

### 2. Back-to-Back (B2B) Fatigue

Subtract penalty per remaining quarter when a team played the previous day.

334 b2b team-games found in dataset (~27% of sides).

**Result: Hurts.** MAE +0.2, adds -0.9 bias at halftime. The 1.5 pts/quarter penalty is too aggressive, and the signal may be too weak to detect at this sample size. NBA teams manage rest (load management, shorter rotations) enough that b2b fatigue doesn't reliably reduce total scoring.

### 3. Blowout (Large Margin) Adjustment

Reduce expected scoring when margin exceeds a threshold. Models garbage time — starters pulled, less competitive play.

```
reduction = max(0, margin - threshold) * rate * remaining_quarters
```



# Datasets

https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores?select=PlayerStatisticsUsage.csv