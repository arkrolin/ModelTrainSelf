---
title: "训练不稳定"
tags: ["unstable", "optim"]
confidence: 0.6
---

## Advice
该配置不稳定（axis=optim.lr -> 0.01）。梯度 p99 过高或频繁 spike，考虑 warmup 与 clip。

## Trigger
- verdict = unstable, optim.lr = 0.01

## Evidence
- t005 (unstable): 14 loss spike(s) detected (>1.5x running best)

## Body
观察：

- 14 loss spike(s) detected (>1.5x running best)
