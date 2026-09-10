---
title: "训练不稳定"
tags: ["unstable", "optim"]
confidence: 0.6
---

## Advice
该配置不稳定（axis=optim.schedule.warmup_steps -> 200）。梯度 p99 过高或频繁 spike，考虑 warmup 与 clip。

## Trigger
- verdict = unstable, optim.schedule.warmup_steps = 200

## Evidence
- t011 (unstable): 16 loss spike(s) detected (>1.5x running best)

## Body
观察：

- 16 loss spike(s) detected (>1.5x running best)
