# Market Microstructure Feature Drift & Signal Research

## Overview
This project investigates whether high-frequency market microstructure features can produce stable predictive signals across market regimes.

## Motivation
Financial models often degrade when market conditions shift. This project studies feature drift, hidden liquidity behavior, order imbalance, and order lifespan metrics to evaluate signal robustness.

## What I Built
- Engineered L3-style market microstructure features including order imbalance, hidden trade rate, order lifespan, and liquidity behavior.
- Compared feature behavior across different time windows to identify distribution drift.
- Built calibration scripts to analyze feature stability and model degradation.
- Investigated forward-test degradation by identifying shifts in load-bearing predictive features.
- Developed drift-aware feature engineering approaches using rolling percentile ranks and regime-sensitive calibration.

## Key Techniques
Python, pandas, NumPy, statistical analysis, feature engineering, time-series analysis, market microstructure, drift analysis.

## Why It Matters
In financial ML, predictive performance often breaks because the market regime changes. This project focuses on diagnosing why a signal works in one window but weakens in another.