# 前后端接口审计报告 - 最终版

## 问题总结

经过完整审计，发现了 **3 个关键问题**导致调度器无法启动和前端显示异常：

---

## ❌ 问题 1: dispatcher_manager 不支持 worker_requirement（已修复）

### 问题描述
`dispatcher_manager.py:build_dispatch_config()` 函数只处理旧格式的 `config.workers` 数组，完全忽略新格式的 `config.worker_requirement`，导致使用新格式保存的项目无法启动调度器。

### 根本原因
```python
# 旧代码：只处理 config.workers 数组
for spec in config.workers:  # ← worker_requirement 项目这里是空数组
    workers.append(...)

if not workers:
    raise RuntimeError("No workers configured")  # ← 直接报错
```

### 修复方案
```python
# 新代码：优先使用 worker_requirement，回退到 workers
if config.worker_requirement and config.worker_requirement.count > 0:
    req = config.worker_requirement
    max_workers = req.count
    # 根据 count 创建多个 worker
    for i in range(req.count):
        workers.append({
            "name": f"{req.worker_type}-{i+1}",
            "type": req.worker_type,
            ...
        })
elif config.workers:
    # 旧格式处理
    for spec in config.workers:
        workers.append(...)
```

**文件**: `src/mts/server/dispatcher_manager.py:85-140`

---

## ❌ 问题 2: 前端 startSearch 使用旧格式（已修复）

### 问题描述
前端的 `startSearch()` 函数在启动调度器时，只构造旧格式的 `workers` 数组，没有传递新格式的 `worker_requirement`。

### 根本原因
```javascript
// 旧代码：只构造 workers 数组
const payload = {
  config: {
    max_trials: config.max_trials || 12,
    max_workers: config.max_workers || 2,
    workers: (config.workers || []).map(w => ({...}))  // ← 空数组
  }
};
```

### 修复方案
```javascript
// 新代码：优先使用 worker_requirement
if (config.worker_requirement) {
  payload.config = {
    max_trials: config.max_trials || 12,
    worker_requirement: {
      worker_type: config.worker_requirement.worker_type,
      count: config.worker_requirement.count,
      provider_id: this.searchProviderId || config.worker_requirement.provider_id || null
    }
  };
} else if (config.workers && config.workers.length > 0) {
  // 旧格式回退
  payload.config = { max_trials, max_workers, workers };
}
```

**文件**: `src/mts/server/static/app.js:448-492`

---

## ❌ 问题 3: Intent 状态字段缺失（已修复）

### 问题描述
后端 Intent 模型没有 `status` 字段，状态需要从 `concluded_at` 和 `worker` 字段推导，但前端直接访问 `intent.status` 导致筛选失败。

### 根本原因
```javascript
// 前端代码直接访问 intent.status
workingIntents() {
  return this.intents.filter(i => i.status === 'working');  // ← 永远为空
}

// 但后端返回的 intent 对象没有 status 字段
{
  id: "...",
  worker: "claudecode-1",  // ← 需要从这里推导
  concluded_at: null,      // ← 和这里
  // status: undefined      // ← 没有此字段
}
```

### 修复方案
```javascript
// 在 applyGraph 时计算并添加 status 字段
applyGraph(data) {
  this.facts = data.facts || [];
  this.intents = (data.intents || []).map(intent => ({
    ...intent,
    status: intent.concluded_at ? 'completed'
          : intent.worker ? 'working'
          : 'unclaimed'
  }));
  this.hints = data.hints || [];
}
```

**文件**: `src/mts/server/static/app.js:162-170`

---

## ✅ 已验证正常的接口

1. **项目设置保存** - `PUT /api/projects/{pid}/search-config`
   - ✅ 前端正确发送 `worker_requirement`
   - ✅ 后端正确保存到 `search_configs` 表

2. **项目设置加载** - `GET /api/projects/{pid}/search-config`
   - ✅ 后端返回 `worker_requirement` 和旧的 `workers`
   - ✅ 前端兼容两种格式

3. **项目详情** - `GET /api/projects/{pid}`
   - ✅ 返回完整的 project + facts + intents + hints
   - ✅ 包含 `search_config` 字段

4. **Intent 创建** - `POST /api/projects/{pid}/intents`
   - ✅ 前端发送 `{from, description, creator}`
   - ✅ 后端正确创建 Intent

---

## 完整的数据流

### 1. 用户配置项目
```
前端表单 (worker_type + count)
    ↓ saveProjectSettings()
PUT /api/projects/{pid}/search-config
    ↓ SearchConfigIn { worker_requirement: {...} }
存入 search_configs 表
```

### 2. 用户启动搜索
```
前端点击"启动搜索"
    ↓ startSearch()
GET /api/projects/{pid}/search-config
    ↓ 读取 worker_requirement
POST /api/projects/{pid}/dispatch/start
    ↓ DispatchStart { config: SearchConfigIn {...} }
dispatcher_manager.start()
    ↓ build_dispatch_config()
    ↓ 从 worker_requirement 生成 workers 列表
启动 DispatcherLoop
```

### 3. 前端显示状态
```
轮询 GET /api/projects/{pid}
    ↓ 返回 intents (无 status 字段)
applyGraph()
    ↓ 为每个 intent 计算 status
    ↓ status = concluded_at ? 'completed' : worker ? 'working' : 'unclaimed'
workingIntents() / unclaimedIntents()
    ↓ 筛选 intents 显示到 UI
```

---

## 测试检查清单

刷新浏览器后，依次验证：

- [ ] 1. 项目设置中的 Worker 配置正确显示（类型 + 数量）
- [ ] 2. 修改并保存项目设置成功
- [ ] 3. 点击"启动搜索"后，调度器成功启动
- [ ] 4. "Agent 实时活动"显示 workers 在运行
- [ ] 5. Intent 节点正确显示状态（working / unclaimed / completed）
- [ ] 6. 停止搜索后，状态正确更新

---

## 技术债务

### 数据库层
- `search_configs` 表同时保留 `workers` (TEXT) 和 `worker_requirement` (TEXT) 两个字段
- 建议：未来版本可以删除 `workers` 和 `max_workers` 字段

### 模型层
- `SearchConfigIn` 同时接受新旧两种格式
- 标记为 `deprecated=True` 但仍需要维护

### 前端层
- 需要在多处兼容新旧格式
- 建议：设置一个数据迁移期限，之后完全移除旧格式支持
