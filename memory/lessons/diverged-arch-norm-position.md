---
title: "训练发散与配置敏感"
tags: ["diverged", "arch"]
confidence: 0.6
---

## Advice
该配置导致发散（axis=arch.norm_position -> post）。降低 lr、提高 warmup、或改回 pre-norm。

## Trigger
- verdict = diverged, arch.norm_position = post

## Evidence
- t002 (diverged): loss became NaN/Inf — the update blew up, not just the metric; 13 loss spike(s) detected (>1.5x running best); final val accuracy=0.000 is far below the task ceiling — the model never learned (lr too small? capacity too low?)

## Body
观察：

- loss became NaN/Inf — the update blew up, not just the metric
- 13 loss spike(s) detected (>1.5x running best)
- final val accuracy=0.000 is far below the task ceiling — the model never learned (lr too small? capacity too low?)
