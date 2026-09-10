---
title: "训练不稳定"
tags: ["unstable", "arch"]
confidence: 0.6
---

## Advice
该配置不稳定（axis=arch.init.residual_scale -> 0.5）。梯度 p99 过高或频繁 spike，考虑 warmup 与 clip。

## Trigger
- verdict = unstable, arch.init.residual_scale = 0.5

## Evidence
- t013 (unstable): 10 loss spike(s) detected (>1.5x running best)

## Body
观察：

- 10 loss spike(s) detected (>1.5x running best)
