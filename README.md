# MolelTrainSelf（MTS）

> 让多个 Agent 基于共享看板，自主搜索深度模型训练的状态空间 —— 并把每一次实验的结论与内部诊断，沉淀成可追溯的知识图谱。

灵感来自 [Cairn](https://github.com/oritera/Cairn)
MolelTrainSelf 自主搜索深度模型训练的状态空间，自动探索"架构设计 × 超参数 × 优化策略"。

---

## 它到底在解决什么

调一次模型，人真正在做的事情是：

1. 跑一个 baseline，看曲线；
2. 曲线不对劲 → 去看梯度分布、参数分布、逐层激活；
3. 凭经验猜一个方向（"是不是 prenorm 的问题，换 postnorm 试试"）；
4. 改配置再跑一遍，对比；
5. 把这次的教训记在脑子里。

这个循环有三个硬伤：**思路会断**（没人记录为什么试了这个方向）、**诊断靠猜**（曲线相同，内部健康度可能完全不同）、**经验不沉淀**（换个人/换个模型重来一遍）。

MTS 的解法是：把这个循环变成一台**有分工、有调度、有记忆的多智能体流水线**，让 Agent 去走这个循环，人只负责定目标、看板、和随时插一句提示。

---

### 方式一：演示模式（推荐首次体验）

```bash
# 1. 进入项目目录
cd /root/work/nlp/xjzhao13/lijie_llama/MolelTrainSelf/ModelTrainSelf

# 2. 启动演示（自动创建示例项目）
./demo_with_provider.sh
```

演示模式会：

- 自动创建一个示例项目（字符级语言模型训练）
- 启动 Web 界面在 http://127.0.0.1:8765
- 预设 20 次实验预算

### 方式二：空白服务器（从零配置）

```bash
# 1. 进入项目目录
cd /root/work/nlp/xjzhao13/lijie_llama/MolelTrainSelf/ModelTrainSelf

# 2. 启动空白服务器
./start_server.sh
```

---

### 可能的风险

为了给 Claude 子进程随意探索模型架构的自由度，MTS 目前没有对训练任务做任何限制。请确保：

- 训练任务不会占满 GPU 内存（否则会导致整个服务器挂掉）
- 会默认使用 claude 的 --dangerously-skip-permissions 参数，如果使用root权限需要小心
