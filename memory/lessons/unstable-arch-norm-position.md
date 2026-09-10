---
title: "训练不稳定"
tags: ["unstable", "arch"]
confidence: 0.6
---

## Advice
该配置不稳定（axis=arch.norm_position -> pre）。梯度 p99 过高或频繁 spike，考虑 warmup 与 clip。

## Trigger
- verdict = unstable, arch.norm_position = pre

## Evidence
- t012 (unstable): 3 loss spike(s) detected (>1.5x running best)

## Body
观察：

- 3 loss spike(s) detected (>1.5x running best)
