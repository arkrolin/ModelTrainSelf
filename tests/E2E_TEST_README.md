# E2E 测试文档

## 概述

本目录包含三种级别的端到端测试，从简单到复杂递进验证整个 ModelTrainSelf 系统。

## 测试文件

### 1. `test_e2e_torch.py` - 训练基础设施测试 ✅

**目的**: 验证 PyTorch 训练器 + 看板 API 的核心功能，无需 agent。

**验证内容**:
- ✅ Board 看板服务器（HTTP API）
- ✅ 真实 PyTorch + CUDA 训练
- ✅ Trial 注册和 Fact 创建
- ✅ 项目导出

**运行条件**:
- CUDA GPU 可用
- 无需 LLM API

**运行方式**:
```bash
pytest tests/test_e2e_torch.py::test_e2e_torch_real_gpu_training -xvs
```

**预期时间**: ~40 秒（包含 35 秒真实训练）

**测试通过标志**:
```
✓ Real PyTorch E2E passed: val_loss=3.6737, backend=torch, device=cuda:0, verdict=underfitting, duration=35.0s
```

---

### 2. `test_e2e_deepseek_agent.py` - 真实 Agent 测试 🆕

**目的**: 验证完整的 Agent 工作流，使用真实的 DeepSeek LLM。

**验证内容**:
- ✅ Board 看板服务器
- ✅ Dispatcher 调度器
- ✅ DeepSeek Agent（Bootstrap/Reason/Explore）
- ✅ 真实 PyTorch + CUDA 训练
- ✅ Agent 自动生成配置、运行训练、创建 Fact

**两个测试场景**:

#### 2.1 `test_e2e_deepseek_agent_bootstrap` - Bootstrap 任务
Agent 读取项目描述（prose-only），自动生成训练配置并执行。

#### 2.2 `test_e2e_deepseek_agent_reason_explore` - 完整循环
完整的 Bootstrap → Reason → Explore 循环，验证多轮迭代优化。

**运行条件**:
- CUDA GPU 可用
- DeepSeek API 配置（环境变量）

**配置 DeepSeek API**:
```bash
export LLM_API_KEY="your-deepseek-api-key"
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-chat"  # 或 deepseek-coder
```

**运行方式**:
```bash
# Bootstrap 测试（单次任务）
pytest tests/test_e2e_deepseek_agent.py::test_e2e_deepseek_agent_bootstrap -xvs

# 完整循环测试（多轮迭代）
pytest tests/test_e2e_deepseek_agent.py::test_e2e_deepseek_agent_reason_explore -xvs
```

**预期时间**: 
- Bootstrap: ~5 分钟
- 完整循环: ~10 分钟

**测试通过标志**:
```
✓ DeepSeek Agent 完整循环测试通过！
  - 完成试验: 3
  - Agent: DeepSeek deepseek-chat
```

---

## API 说明

系统提供以下核心 API：

### Board 看板 API

**项目管理**:
- `POST /api/projects` - 创建项目
- `GET /api/projects/{pid}` - 获取项目详情
- `PATCH /api/projects/{pid}/status` - 更新项目状态
- `GET /api/projects/{pid}/export` - 导出项目（YAML）

**Intent 管理**:
- `POST /api/projects/{pid}/intents` - 创建 intent
- `POST /api/projects/{pid}/intents/{iid}/claim` - 认领 intent
- `POST /api/projects/{pid}/intents/{iid}/conclude` - 结束 intent（创建 fact）

**Trial 管理**:
- `POST /api/projects/{pid}/trials` - 注册训练试验
- `POST /api/trials/{tid}/heartbeat` - 发送心跳（训练进度）

### Dispatcher 配置

Dispatcher 通过 YAML 配置文件定义 worker：

```yaml
runtime:
  max_workers: 3
  max_running_projects: 1
  interval: 2
  execution: local

tasks:
  bootstrap: {timeout: 300, conclude_timeout: 60}
  reason: {timeout: 180, max_intents: 3}
  explore: {timeout: 300, conclude_timeout: 60}

local:
  workspace_root: /path/to/workspace
  completed_action: keep

workers:
  - name: deepseek-bootstrap
    type: llm  # 或 claudecode, mock
    task_types: [bootstrap]
    max_running: 1
    priority: 10
    env:
      LLM_MODEL: deepseek-chat
      LLM_BASE_URL: https://api.deepseek.com
      LLM_API_KEY: your-api-key
      LLM_API_FORMAT: openai
```

**Worker 类型**:
1. **`llm`** - OpenAI 兼容的 API（DeepSeek、OpenAI、本地模型）
   - 环境变量: `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_API_FORMAT`
   
2. **`claudecode`** - Anthropic Claude Code
   - 环境变量: `ANTHROPIC_MODEL`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`
   
3. **`mock`** - 模拟 worker（测试用）
   - 可配置随机行为和延迟

**任务类型**:
- `bootstrap` - 初始化项目，生成第一个配置
- `reason` - 分析结果，生成新的探索方向（intents）
- `explore` - 执行一个 intent，运行训练，创建 fact

---

## 测试架构对比

| 测试 | Board | Dispatcher | Agent | 训练 | 用途 |
|------|-------|------------|-------|------|------|
| `test_e2e_torch.py` | ✅ | ❌ | ❌ | ✅ PyTorch | 验证训练基础设施 |
| `test_e2e_deepseek_agent.py` | ✅ | ✅ | ✅ DeepSeek | ✅ PyTorch | 验证完整 Agent 流程 |

---

## 故障排查

### 1. DeepSeek API 测试失败

**症状**: `pytest.skip("DeepSeek API not configured")`

**解决**:
```bash
# 检查环境变量
echo $LLM_API_KEY
echo $LLM_BASE_URL

# 手动测试 API
curl -X POST $LLM_BASE_URL/v1/chat/completions \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"ping"}],"max_tokens":10}'
```

### 2. CUDA 不可用

**症状**: `pytest.skip("No CUDA available")`

**解决**:
```bash
# 检查 CUDA
nvidia-smi

# 检查 PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. Dispatcher 超时

**症状**: `Bootstrap 未在 300s 内完成`

**可能原因**:
1. DeepSeek API 响应慢 → 增加 `timeout` 配置
2. Prompt 解析失败 → 检查 dispatcher 日志
3. 训练执行失败 → 检查 workspace 中的 trial 目录

**调试**:
```bash
# 查看 workspace 中的试验目录
ls -la /tmp/pytest-*/workspace/

# 查看训练日志
cat /tmp/pytest-*/workspace/trial_*/log.txt

# 查看训练配置
cat /tmp/pytest-*/workspace/trial_*/spec.json
```

---

## 下一步

### 扩展测试覆盖

1. **其他 LLM 提供商**:
   - OpenAI GPT-4
   - 本地 LLaMA 模型
   - Claude Code

2. **不同训练场景**:
   - 视觉任务（卷积网络）
   - NLP 任务（Transformer）
   - 多模态任务

3. **并行 Worker**:
   - 测试多个 worker 同时执行
   - 测试 worker 失败恢复

### 性能测试

- 训练速度基准测试
- API 响应时间测试
- 多项目并发测试

---

## 总结

✅ **已验证的组件**:
- PyTorch 训练器（真实 GPU）
- Board 看板 HTTP API
- Dispatcher 调度器
- DeepSeek LLM Agent

🎯 **测试策略**:
1. 先运行 `test_e2e_torch.py` 验证基础设施
2. 配置 DeepSeek API
3. 运行 `test_e2e_deepseek_agent.py` 验证完整流程

📊 **测试覆盖率**:
- 训练基础设施: ✅ 100%
- Agent 流程: ✅ Bootstrap + Reason + Explore
- 边界情况: 🚧 进行中
