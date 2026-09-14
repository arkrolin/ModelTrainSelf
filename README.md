# MolelTrainSelf（MTS）

> 让多个 Agent 基于共享看板，自主搜索深度模型训练的状态空间 —— 并把每一次实验的结论与内部诊断，沉淀成可追溯的知识图谱。

灵感来自 [Cairn](https://github.com/oritera/Cairn)
MolelTrainSelf 自主搜索深度模型训练的状态空间，自动探索"架构设计 × 超参数 × 优化策略"。

---

## 快速开始

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
