/* ModelTrainSelf — 自主训练搜索前端 (Alpine.js)
 *
 * 数据模型跟随后端的自主 Agent 架构：Agent 自己写 train.py，通过
 * POST /intents/{iid}/conclude 上报结果，落在 facts 表。所以主视图是
 * facts + intents 的搜索图谱，不是试验表（trials 表在这条链路里不写入）。
 * 指标键名由 Agent 决定，前端不做任何硬编码假设。
 */

function mtsApp() {
  return {
    // ===== 视图 =====
    view: 'list',                 // 'list' | 'detail'
    sideTab: 'detail',            // 'detail' | 'hints' | 'timeline' | 'log'

    // ===== 项目 =====
    projects: [],
    currentProject: null,

    // ===== 图谱：Agent 真实产出 =====
    facts: [],
    intents: [],
    hints: [],
    selectedNode: null,           // {type:'fact'|'intent', id}
    sortFactsByMetric: false,

    // ===== 搜索运行状态 =====
    searchStatus: null,
    searchRunning: false,
    dispatchLog: '',
    searchProviderId: '',         // '' = 宿主机环境变量
    agentActivity: [],            // agent 活动流
    activitySeq: 0,               // 已读到的最大 seq
    _activityInFlight: false,     // 防止「刷新」与 tick 并发拉回重复 seq

    // ===== 图表 =====
    progressChart: null,

    // ===== DAG 流程图（cytoscape）=====
    graphView: 'graph',           // 'graph' = DAG 流程图 | 'list' = 卡片列表
    layoutMode: 'dagre_tb',       // dagre_tb|dagre_lr|klay_tb|klay_lr|elk_tb|elk_lr
    cy: null,
    _resizeObserver: null,

    // ===== 模态框 =====
    showNewProject: false,
    showHintModal: false,
    showSettings: false,
    showProjectSettings: false,
    projectForm: null,            // 项目配置草稿，null = 未打开
    projectFormSaving: false,
    projectFormError: '',
    newProject: {
      title: '', origin: '', goal: '',
      goal_metric: 'val_loss', goal_direction: 'minimize', goal_target: null,
      budget_max_trials: 12, bootstrap_enabled: true,
      hints: [{ content: '' }]
    },
    newHint: { content: '', creator: 'human' },

    // ===== LLM Provider =====
    providers: [],
    providerForm: null,
    providerSaving: false,
    providerFormTest: null,
    providerFormTesting: false,
    providerTests: {},
    testingProviderId: null,
    providerTestError: '',
    providerPresets: [
      { label: 'OpenAI', kind: 'openai', base_url: 'https://api.openai.com', model: 'gpt-4o' },
      { label: 'DeepSeek', kind: 'openai', base_url: 'https://api.deepseek.com', model: 'deepseek-chat' },
      { label: '通义千问', kind: 'openai', base_url: 'https://dashscope.aliyuncs.com/compatible-mode', model: 'qwen-max' },
      { label: '智谱 GLM', kind: 'openai', base_url: 'https://open.bigmodel.cn/api/paas/v4', model: 'glm-4-plus' },
      { label: 'Moonshot', kind: 'openai', base_url: 'https://api.moonshot.cn', model: 'moonshot-v1-32k' },
      { label: 'OpenRouter', kind: 'openai', base_url: 'https://openrouter.ai/api', model: 'anthropic/claude-3.5-sonnet' },
      { label: '本地 vLLM', kind: 'openai', base_url: 'http://127.0.0.1:8000', model: '' },
      { label: '本地 Ollama', kind: 'openai', base_url: 'http://127.0.0.1:11434', model: 'qwen2.5:32b' },
      { label: 'Anthropic', kind: 'anthropic', base_url: 'https://api.anthropic.com', model: 'claude-opus-4-20250514' },
    ],

    // ===== 初始化 =====
    async init() {
      await Promise.all([this.loadProjects(), this.loadProviders()]);
      setInterval(() => this.tick(), 5000);
    },

    async tick() {
      if (this.view === 'list') {
        this.loadProjects();
      } else if (this.currentProject) {
        await this.loadGraph();
        await this.loadSearchStatus();
        // 活动流/日志都是后端的环形缓冲，调度停止后内容依然在。这里不能再加
        // searchRunning 条件：conclude 阶段的事件恰好落在 running 翻成 false
        // 前后，带上这个条件就会把最后那批（含结论 JSON）永久错过。
        if (this.sideTab === 'log') this.loadLogs();
        if (this.sideTab === 'activity') this.loadActivity();
      }
    },

    /** 切侧栏 tab：立刻拉一次，不然要等下一个 5s tick 才有内容。 */
    setSideTab(tab) {
      this.sideTab = tab;
      if (tab === 'activity') this.loadActivity();
      else if (tab === 'log') this.loadLogs();
    },

    // ===== API =====
    async api(url, options = {}) {
      const { silent = false, ...init } = options;
      try {
        const response = await fetch(url, {
          ...init,
          headers: { 'Content-Type': 'application/json', ...init.headers }
        });
        if (!response.ok) {
          let detail = response.statusText;
          try {
            const body = await response.json();
            if (body && body.detail) detail = body.detail;
          } catch (_) { /* 非 JSON 响应 */ }
          throw new Error(`HTTP ${response.status}: ${detail}`);
        }
        return await response.json();
      } catch (error) {
        console.error('API Error:', url, error);
        if (!silent) alert(`请求失败: ${error.message}`);
        throw error;
      }
    },

    // ===== 项目 =====
    async loadProjects() {
      try {
        this.projects = await this.api('/api/projects', { silent: true });
      } catch (_) { /* 轮询失败保留上次数据 */ }
    },

    async openProject(projectId) {
      try {
        const data = await this.api(`/api/projects/${projectId}`);
        this.currentProject = data.project || data;
        this.view = 'detail';
        this.selectedNode = null;
        this.sideTab = 'detail';
        // 活动流的 seq 是按项目计数的，换项目必须归零，否则新项目的低位 seq
        // 会被上一个项目的高位 activitySeq 全部过滤掉。
        this.agentActivity = [];
        this.activitySeq = 0;
        this.applyGraph(data);
        await this.loadSearchStatus();
      } catch (_) { /* alert 已提示 */ }
    },

    backToList() {
      this.view = 'list';
      this.currentProject = null;
      this.facts = [];
      this.intents = [];
      this.hints = [];
      this.selectedNode = null;
      this.dispatchLog = '';
      this.agentActivity = [];
      this.activitySeq = 0;
      this.disposeChart();
      this.disposeGraph();
      this.loadProjects();
    },

    async loadGraph() {
      if (!this.currentProject) return;
      const pid = this.currentProject.id;
      try {
        const data = await this.api(`/api/projects/${pid}`, { silent: true });
        // await 期间用户可能已经 backToList()（或换了项目）。此时容器已被 x-show
        // 隐藏，继续 applyGraph 会在 0×0 容器上建出一个坏的 cytoscape 实例，
        // 再次进入项目时 this.cy 非空 → 只走 updateGraph → 图永远不显示。
        if (this.view !== 'detail' || this.currentProject?.id !== pid) return;
        this.currentProject = data.project || this.currentProject;
        this.applyGraph(data);
      } catch (_) { /* 轮询失败保留上次数据 */ }
    },

    applyGraph(data) {
      this.facts = data.facts || [];
      // 后端已经在 services.get_intent 里算好 status 了（unclaimed / working /
      // concluded），直接用它。这里原本自己按 concluded_at 又算了一遍，同一个
      // 状态在后端叫 concluded、在前端叫 completed —— 现在时间线把 status 当标签
      // 显示，这种分歧会直接漏到界面上。只留后端那一套命名。
      // 兜底是为了老响应里没有 status 的情况，不是第二套真源。
      this.intents = (data.intents || []).map(intent => ({
        ...intent,
        status: intent.status
              || (intent.concluded_at ? 'concluded'
                : intent.worker ? 'working'
                : 'unclaimed')
      }));
      this.hints = data.hints || [];
      this.renderProgressChart();
      // cytoscape 需要容器已经上屏才能量测尺寸，等 Alpine 渲染完再建图
      this.$nextTick(() => {
        if (this.view !== 'detail') return;
        if (this.graphView !== 'graph') return;
        this.ensureGraph();
      });
    },

    /** 建图/更新图的唯一入口：顺带把 0 尺寸的坏实例回收重建。 */
    ensureGraph() {
      const container = document.getElementById('cy');
      if (!container) return;
      if (container.clientWidth === 0 || container.clientHeight === 0) return;
      // 实例是在隐藏容器上建起来的（宽高为 0）→ 画布量测已经错了，只能重建。
      if (this.cy && this.cy.width() === 0) this.disposeGraph();
      if (this.cy) {
        this.cy.resize();
        this.updateGraph();
      } else {
        this.initGraph();
      }
    },

    setGraphView(mode) {
      if (this.graphView === mode) return;
      this.graphView = mode;
      if (mode !== 'graph') return;
      this.$nextTick(() => {
        this.ensureGraph();
        if (this.cy) this.cy.fit(undefined, 50);
      });
    },

    async toggleProjectStatus(projectId, currentStatus) {
      const next = currentStatus === 'active' ? 'stopped' : 'active';
      try {
        await this.api(`/api/projects/${projectId}/status?status=${next}`, { method: 'PATCH' });
        await this.loadProjects();
        if (this.currentProject?.id === projectId) this.currentProject.status = next;
      } catch (_) { /* alert 已提示 */ }
    },

    async deleteProject(projectId, title) {
      if (!confirm(`确定删除项目 "${title}" 吗？此操作不可恢复。`)) return;
      try {
        await this.api(`/api/projects/${projectId}`, { method: 'DELETE' });
        if (this.currentProject?.id === projectId) this.backToList();
        await this.loadProjects();
      } catch (_) { /* alert 已提示 */ }
    },

    async createProject() {
      try {
        const hints = this.newProject.hints
          .map(h => (h.content || '').trim())
          .filter(Boolean);
        const target = parseFloat(this.newProject.goal_target);
        const payload = {
          title: this.newProject.title,
          origin: this.newProject.origin,
          goal: this.newProject.goal,
          goal_metric: (this.newProject.goal_metric || 'val_loss').trim(),
          goal_direction: this.newProject.goal_direction,
          goal_target: Number.isNaN(target) ? null : target,
          budget_max_trials: parseInt(this.newProject.budget_max_trials) || 12,
          bootstrap_enabled: !!this.newProject.bootstrap_enabled,
          hints: hints
        };
        const result = await this.api('/api/projects', {
          method: 'POST', body: JSON.stringify(payload)
        });
        this.showNewProject = false;
        this.resetNewProjectForm();
        await this.loadProjects();
        if (result.id) await this.openProject(result.id);
      } catch (_) { /* alert 已提示 */ }
    },

    resetNewProjectForm() {
      this.newProject = {
        title: '', origin: '', goal: '',
        goal_metric: 'val_loss', goal_direction: 'minimize', goal_target: null,
        budget_max_trials: 12, bootstrap_enabled: true,
        hints: [{ content: '' }]
      };
    },

    countProjectsByStatus(status) {
      return this.projects.filter(p => p.status === status).length;
    },

    budgetPercent(proj) {
      const budget = proj.budget_max_trials || 1;
      return Math.min(100, Math.round(((proj.trial_count || 0) / budget) * 100));
    },

    statusBadgeClass(status) {
      if (status === 'active') return 'bg-teal-50 text-teal-700';
      if (status === 'completed') return 'bg-indigo-50 text-indigo-700';
      return 'bg-slate-100 text-slate-500';
    },

    // ===== 图谱选择与排序 =====
    selectFact(factId) {
      this.selectedNode = { type: 'fact', id: factId };
      this.sideTab = 'detail';
      this.refreshGraphDecorations();
    },

    selectIntent(intentId) {
      this.selectedNode = { type: 'intent', id: intentId };
      this.sideTab = 'detail';
      this.refreshGraphDecorations();
    },

    get selectedFact() {
      if (this.selectedNode?.type !== 'fact') return null;
      return this.facts.find(f => f.id === this.selectedNode.id);
    },

    get selectedIntent() {
      if (this.selectedNode?.type !== 'intent') return null;
      return this.intents.find(i => i.id === this.selectedNode.id);
    },

    workingIntents() {
      return this.intents.filter(i => i.status === 'working');
    },

    unclaimedIntents() {
      return this.intents.filter(i => i.status === 'unclaimed');
    },

    sortedFacts() {
      const nonSeed = this.facts.filter(f => f.id !== 'origin' && f.id !== 'goal');
      if (!this.sortFactsByMetric || !this.currentProject?.goal_metric) {
        return nonSeed.sort((a, b) =>
          new Date(b.created_at || 0) - new Date(a.created_at || 0)
        );
      }
      const key = this.currentProject.goal_metric;
      const dir = this.currentProject.goal_direction === 'maximize' ? -1 : 1;
      return nonSeed.sort((a, b) => {
        const va = a.metrics?.[key];
        const vb = b.metrics?.[key];
        if (va == null && vb == null) return 0;
        if (va == null) return 1;
        if (vb == null) return -1;
        return dir * (va - vb);
      });
    },

    isBestFact(fact) {
      if (!this.currentProject?.goal_metric) return false;
      if (fact.id === 'origin' || fact.id === 'goal') return false;
      const key = this.currentProject.goal_metric;
      const val = fact.metrics?.[key];
      if (val == null) return false;
      const nonSeed = this.facts.filter(f =>
        f.id !== 'origin' && f.id !== 'goal' && f.metrics?.[key] != null
      );
      if (nonSeed.length === 0) return false;
      const dir = this.currentProject.goal_direction === 'maximize' ? -1 : 1;
      const sorted = [...nonSeed].sort((a, b) =>
        dir * (a.metrics[key] - b.metrics[key])
      );
      return sorted[0]?.id === fact.id;
    },

    // 调用点传的是 fact.id 字符串（也兼容传 fact 对象）。
    // 后端 fact 上没有 intent_id 字段，反查 intents 里 to 指向它的那条。
    producingIntent(factOrId) {
      const factId = typeof factOrId === 'string' ? factOrId : factOrId?.id;
      if (!factId) return null;
      return this.intents.find(i => i.to === factId) || null;
    },

    metricSeries() {
      if (!this.currentProject?.goal_metric) return [];
      const key = this.currentProject.goal_metric;
      const nonSeed = this.facts
        .filter(f => f.id !== 'origin' && f.id !== 'goal' && f.metrics?.[key] != null)
        .sort((a, b) => new Date(a.created_at || 0) - new Date(b.created_at || 0));
      return nonSeed.map((f, idx) => ({
        round: idx + 1,
        value: f.metrics[key],
        factId: f.id,
        timestamp: f.created_at
      }));
    },

    // 字段名必须和模板（index.html 的时间线 template）对齐：那边读的是
    // key / label / at / description。之前这里产出的是 created_at 和一个后端
    // 根本没有的 i.title，于是每一条时间线都是空 id 标签、空描述、时间显示
    // 「-」，:key 还全是 undefined。hints 同理：模板早就为它备好了 amber 配色
    // 分支，但这里从来没往 events 里放过 hint。
    timelineEvents() {
      const events = [];
      this.facts.forEach(f => {
        events.push({
          key: `fact_${f.id}`, type: 'fact', id: f.id, label: 'fact',
          description: f.description, at: f.created_at, status: null
        });
      });
      this.intents.forEach(i => {
        events.push({
          key: `intent_${i.id}`, type: 'intent', id: i.id,
          // intent 没有标题，描述就是它自己
          label: i.status || 'intent', description: i.description,
          at: i.created_at, status: i.status
        });
      });
      this.hints.forEach(h => {
        events.push({
          key: `hint_${h.id}`, type: 'hint', id: h.id, label: 'hint',
          description: h.content, at: h.created_at, status: null
        });
      });
      return events.sort((a, b) => new Date(b.at || 0) - new Date(a.at || 0));
    },

    selectTimelineEvent(evt) {
      if (evt.type === 'fact') this.selectFact(evt.id);
      else if (evt.type === 'intent') this.selectIntent(evt.id);
      // hint 没有图谱节点可选中，跳到「提示」页签比点了没反应好。
      else this.sideTab = 'hints';
    },

    // ===== 提示管理 =====
    openHintModal() {
      this.newHint = { content: '', creator: 'human' };
      this.showHintModal = true;
    },

    async createHint() {
      if (!this.currentProject || !this.newHint.content.trim()) return;
      try {
        await this.api(`/api/projects/${this.currentProject.id}/hints`, {
          method: 'POST',
          body: JSON.stringify({
            content: this.newHint.content.trim(),
            creator: this.newHint.creator
          })
        });
        this.showHintModal = false;
        await this.loadGraph();
      } catch (_) { /* alert 已提示 */ }
    },

    async deleteHint(hintId) {
      if (!confirm('确定删除此提示吗？')) return;
      try {
        await this.api(`/api/projects/${this.currentProject.id}/hints/${hintId}`, {
          method: 'DELETE'
        });
        await this.loadGraph();
      } catch (_) { /* alert 已提示 */ }
    },

    // ===== 从节点继续探索 =====
    async createIntentFromFact(factId) {
      if (!this.currentProject) return;
      const fact = this.facts.find(f => f.id === factId);
      if (!fact) return;

      const description = prompt(
        `从 ${factId} 继续探索\n\n请描述探索方向（例如：尝试更大的学习率、增加模型深度等）：`,
        `基于 ${factId} 的结果，尝试进一步优化`
      );

      if (!description || !description.trim()) return;

      try {
        await this.api(`/api/projects/${this.currentProject.id}/intents`, {
          method: 'POST',
          body: JSON.stringify({
            from: [factId],
            description: description.trim(),
            creator: 'human'
          })
        });
        await this.loadGraph();
        alert('✓ 已创建探索任务，调度器会自动分配 Agent 执行');
      } catch (err) {
        console.error('创建 intent 失败:', err);
      }
    },

    // ===== 搜索控制 =====
    async loadSearchStatus() {
      if (!this.currentProject) return;
      try {
        const data = await this.api(
          `/api/projects/${this.currentProject.id}/dispatch/status`,
          { silent: true }
        );
        this.searchStatus = data;
        this.searchRunning = data?.running === true;
      } catch (_) {
        this.searchStatus = null;
        this.searchRunning = false;
      }
    },

    async startSearch() {
      if (!this.currentProject) return;
      try {
        // search_config 不在项目详情响应里，必须单独拉，否则 workers 恒为空 → 后端 400
        const config = await this.api(
          `/api/projects/${this.currentProject.id}/search-config`,
          { silent: true }
        );

        // 从未保存过配置时，后端回的是 mock 默认值。直接拿它启动会跑出一批假
        // agent、假指标，还照样吃掉实验预算，所以这里必须先问一句。
        if (config.configured === false) {
          const ok = confirm(
            '这个项目还没有保存过搜索配置。\n\n' +
            '现在启动会使用 mock worker：不调用真实 LLM，产出的是假指标，' +
            '但仍然会写入知识图谱并消耗实验预算。\n\n' +
            '建议先打开「项目配置」选好 Worker 类型。仍要用 mock 启动吗？'
          );
          if (!ok) return;
        }

        // 构建 payload：优先使用 worker_requirement（新格式），回退到 workers（旧格式）
        const payload = { config: {} };

        if (config.worker_requirement) {
          // 新格式：使用 worker_requirement
          payload.config = {
            max_trials: config.max_trials || 12,
            worker_requirement: {
              worker_type: config.worker_requirement.worker_type,
              count: config.worker_requirement.count,
              provider_id: this.searchProviderId || config.worker_requirement.provider_id || null
            }
          };
        } else if (config.workers && config.workers.length > 0) {
          // 旧格式：使用 workers 数组
          payload.config = {
            max_trials: config.max_trials || 12,
            max_workers: config.max_workers || 2,
            workers: config.workers.map(w => ({
              name: w.name,
              driver: w.driver,
              provider_id: this.searchProviderId || w.provider_id || null
            }))
          };
        } else {
          throw new Error('调度器配置不完整，请先配置项目设置');
        }

        await this.api(`/api/projects/${this.currentProject.id}/dispatch/start`, {
          method: 'POST',
          body: JSON.stringify(payload)
        });
        // start 那边会清掉这个项目的活动流环形缓冲（seq 归零），这里必须同步把
        // 本地已读水位也归零：不归零的话新一轮的低位 seq 全部小于 activitySeq，
        // 会被增量拉取整段过滤掉，界面上就是活动流永远空着。
        this.agentActivity = [];
        this.activitySeq = 0;
        this.searchRunning = true;
        await this.loadSearchStatus();
      } catch (err) {
        alert('启动失败: ' + (err.message || '未知错误'));
      }
    },

    async stopSearch() {
      if (!this.currentProject) return;
      if (!confirm('确定停止搜索吗？')) return;
      try {
        await this.api(`/api/projects/${this.currentProject.id}/dispatch/stop`, {
          method: 'POST'
        });
        this.searchRunning = false;
        await this.loadSearchStatus();
      } catch (_) { /* alert 已提示 */ }
    },

    async loadLogs() {
      if (!this.currentProject) return;
      try {
        const data = await this.api(
          `/api/projects/${this.currentProject.id}/dispatch/logs?limit=200`,
          { silent: true }
        );
        this.dispatchLog = data?.output || '(无日志)';
      } catch (_) {
        this.dispatchLog = '(日志加载失败)';
      }
    },

    async loadActivity() {
      if (!this.currentProject) return;
      // 手动「刷新」和 5s tick 会并发，两个请求带同一个 after_seq 就会各拉回一份
      // 相同事件，push 两次后 x-for 的 :key="ev.seq" 出现重复 key，渲染直接乱掉。
      if (this._activityInFlight) return;
      this._activityInFlight = true;
      const pid = this.currentProject.id;
      try {
        const data = await this.api(
          `/api/projects/${pid}/activity?after_seq=${this.activitySeq}&limit=200`,
          { silent: true }
        );
        // await 期间用户可能已经退出/切换项目，这批事件就不属于当前视图了
        if (this.currentProject?.id !== pid) return;
        const events = data?.events || [];
        if (events.length === 0) return;
        const seen = new Set(this.agentActivity.map(e => e.seq));
        const fresh = events.filter(e => !seen.has(e.seq));
        if (fresh.length === 0) return;
        this.agentActivity.push(...fresh);
        // 保留最新 400 条，与后端 RING_SIZE 对齐
        if (this.agentActivity.length > 400) {
          this.agentActivity = this.agentActivity.slice(-400);
        }
        // 按实际收到的最大 seq 推进，而不是后端的全局 latest_seq：limit 截断时
        // 用 latest_seq 会把没拉到的那批事件永久跳过。
        this.activitySeq = Math.max(
          this.activitySeq,
          ...events.map(e => e.seq || 0)
        );
        // 积压超过一个 limit 时后端会截断，续拉到追平，不用干等下一个 tick。
        if ((data.latest_seq || 0) > this.activitySeq) {
          this._activityInFlight = false;
          await this.loadActivity();
        }
      } catch (_) {
        // 静默失败，不干扰主流程
      } finally {
        this._activityInFlight = false;
      }
    },

    // ===== 项目配置（进入项目后就地修改） =====
    async openProjectSettings() {
      if (!this.currentProject) return;
      const p = this.currentProject;
      this.projectFormError = '';
      // search_config 不在项目详情里，单独拉一份填表单
      let cfg = {};
      try {
        cfg = await this.api(
          `/api/projects/${p.id}/search-config`, { silent: true }
        );
      } catch (_) { /* 拉不到就用下面的默认值 */ }

      // 解析 worker_requirement，优先使用新格式
      let workerType = 'claudecode';
      let workerCount = 2;
      // provider_id 这个表单里没有对应控件（端点在顶栏按次选），但必须读出来带回去：
      // PUT 是整份覆盖，不回填就等于把用户存过的端点静默清空。
      let workerProviderId = null;
      if (cfg.worker_requirement) {
        workerType = cfg.worker_requirement.worker_type || 'claudecode';
        workerCount = cfg.worker_requirement.count || 2;
        workerProviderId = cfg.worker_requirement.provider_id || null;
      } else if (cfg.workers && cfg.workers.length > 0) {
        // 兼容旧格式：从 workers 数组推断
        workerType = cfg.workers[0].driver || 'mock';
        workerCount = cfg.max_workers || cfg.workers.length;
        workerProviderId = cfg.workers[0].provider_id || null;
      }

      this.projectForm = {
        title: p.title || '',
        origin: p.origin || '',
        goal: p.goal || '',
        goal_metric: p.goal_metric || 'val_loss',
        goal_direction: p.goal_direction || 'minimize',
        goal_target: p.goal_target,
        budget_max_trials: p.budget_max_trials || 12,
        bootstrap_enabled: !!p.bootstrap_enabled,
        // 同 provider_id：控件已撤（这个字段调度器根本不读，见 index.html 的注释），
        // 但 PUT 是整份覆盖，所以照样要读出来带回去，别把库里存着的值改掉。
        max_trials: cfg.max_trials || 12,
        worker_type: workerType,
        worker_count: workerCount,
        worker_provider_id: workerProviderId
      };
      this.showProjectSettings = true;
    },


    async saveProjectSettings() {
      if (!this.currentProject || !this.projectForm) return;
      const f = this.projectForm;

      // 验证必填字段
      if (!f.worker_type || !f.worker_count || f.worker_count < 1) {
        this.projectFormError = 'Worker 类型和数量必须填写';
        return;
      }

      const target = f.goal_target;
      this.projectFormSaving = true;
      this.projectFormError = '';
      try {
        // 项目本体字段
        await this.api(`/api/projects/${this.currentProject.id}`, {
          method: 'PATCH',
          body: JSON.stringify({
            title: f.title.trim(),
            origin: f.origin.trim(),
            goal: f.goal.trim(),
            goal_metric: f.goal_metric.trim(),
            goal_direction: f.goal_direction,
            goal_target:
              target === '' || target === null || target === undefined
                ? null : Number(target),
            budget_max_trials: Number(f.budget_max_trials),
            bootstrap_enabled: !!f.bootstrap_enabled
          })
        });
        // 搜索配置（独立表，独立端点）
        await this.api(
          `/api/projects/${this.currentProject.id}/search-config`, {
            method: 'PUT',
            body: JSON.stringify({
              max_trials: Number(f.max_trials),
              worker_requirement: {
                worker_type: f.worker_type,
                count: Number(f.worker_count),
                provider_id: f.worker_provider_id || null
              }
            })
          });
        this.showProjectSettings = false;
        this.projectForm = null;
        await this.loadGraph();
        await this.loadProjects();
      } catch (error) {
        this.projectFormError = error.message || '保存失败';
      } finally {
        this.projectFormSaving = false;
      }
    },

    // ===== Provider 管理 =====
    async loadProviders() {
      try {
        this.providers = await this.api('/api/providers', { silent: true });
      } catch (_) { /* 启动时可能还没有 */ }
    },

    applyProviderPreset(preset) {
      if (!this.providerForm) return;
      this.providerForm.kind = preset.kind;
      this.providerForm.base_url = preset.base_url;
      this.providerForm.model = preset.model;
    },

    openProviderForm(provider = null) {
      if (provider) {
        this.providerForm = {
          id: provider.id,
          name: provider.name,
          kind: provider.kind,
          base_url: provider.base_url,
          api_key: '',  // 不回显，用户需要重新输入或留空保持
          model: provider.model,
          is_default: provider.is_default,
          max_tokens: provider.max_tokens || 4096,
          temperature: provider.temperature != null ? provider.temperature : 0.7,
          headers: provider.headers || ''
        };
      } else {
        this.providerForm = {
          id: null, name: '', kind: 'openai',
          base_url: '', api_key: '', model: '',
          is_default: false, max_tokens: 4096, temperature: 0.7, headers: ''
        };
      }
      this.providerFormTest = null;
      this.providerFormTesting = false;
      this.showSettings = true;
    },

    async saveProvider() {
      if (!this.providerForm) return;
      this.providerSaving = true;
      try {
        const payload = {
          name: this.providerForm.name.trim(),
          kind: this.providerForm.kind,
          base_url: this.providerForm.base_url.trim(),
          api_key: this.providerForm.api_key.trim(),
          model: this.providerForm.model.trim(),
          is_default: !!this.providerForm.is_default,
          max_tokens: parseInt(this.providerForm.max_tokens) || 4096,
          temperature: parseFloat(this.providerForm.temperature),
          headers: this.providerForm.headers.trim()
        };
        if (this.providerForm.id) {
          await this.api(`/api/providers/${this.providerForm.id}`, {
            method: 'PATCH',
            body: JSON.stringify(payload)
          });
        } else {
          await this.api('/api/providers', {
            method: 'POST',
            body: JSON.stringify(payload)
          });
        }
        this.providerForm = null;
        await this.loadProviders();
      } catch (_) { /* alert 已提示 */ }
      this.providerSaving = false;
    },

    cancelProviderForm() {
      this.providerForm = null;
      this.providerFormTest = null;
    },

    async deleteProvider(id, name) {
      if (!confirm(`确定删除 Provider "${name}" 吗？`)) return;
      try {
        await this.api(`/api/providers/${id}`, { method: 'DELETE' });
        await this.loadProviders();
      } catch (_) { /* alert 已提示 */ }
    },

    async testProviderForm() {
      if (!this.providerForm) return;
      this.providerFormTesting = true;
      this.providerFormTest = null;
      try {
        const payload = {
          kind: this.providerForm.kind,
          base_url: this.providerForm.base_url.trim(),
          api_key: this.providerForm.api_key.trim(),
          model: this.providerForm.model.trim(),
          headers: this.providerForm.headers.trim()
        };
        this.providerFormTest = await this.api('/api/providers/test', {
          method: 'POST',
          body: JSON.stringify(payload),
          silent: true
        });
      } catch (err) {
        this.providerFormTest = { ok: false, error: err.message };
      }
      this.providerFormTesting = false;
    },

    async testProvider(id) {
      this.testingProviderId = id;
      this.providerTestError = '';
      try {
        const result = await this.api(`/api/providers/${id}/test`, {
          method: 'POST',
          silent: true
        });
        this.providerTests[id] = result;
      } catch (err) {
        this.providerTests[id] = { ok: false, error: err.message };
      }
      this.testingProviderId = null;
    },

    // ===== 工具方法 =====
    formatElapsed(seconds) {
      if (seconds == null) return '-';
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = Math.floor(seconds % 60);
      if (h > 0) return `${h}h${m}m`;
      if (m > 0) return `${m}m${s}s`;
      return `${s}s`;
    },

    formatDate(isoString) {
      if (!isoString) return '-';
      try {
        const d = new Date(isoString);
        const now = new Date();
        const diff = Math.floor((now - d) / 1000);
        if (diff < 60) return '刚刚';
        if (diff < 3600) return `${Math.floor(diff / 60)}分钟前`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}小时前`;
        return d.toLocaleString('zh-CN', {
          month: 'short', day: 'numeric',
          hour: '2-digit', minute: '2-digit'
        });
      } catch (_) {
        return isoString;
      }
    },

    renderProgressChart() {
      const series = this.metricSeries();
      if (series.length < 2) {
        this.disposeChart();
        return;
      }
      const container = document.getElementById('progressChart');
      if (!container) return;
      if (!this.progressChart) {
        this.progressChart = echarts.init(container);
      }
      const metric = this.currentProject?.goal_metric || 'metric';
      const dir = this.currentProject?.goal_direction || 'minimize';
      const bestValue = dir === 'maximize'
        ? Math.max(...series.map(s => s.value))
        : Math.min(...series.map(s => s.value));

      this.progressChart.setOption({
        title: { text: `${metric} 进展`, left: 'center', textStyle: { fontSize: 14 } },
        tooltip: {
          trigger: 'axis',
          formatter: (params) => {
            const p = params[0];
            const point = series[p.dataIndex];
            return `轮次 ${point.round}<br/>${metric}: ${point.value.toFixed(4)}<br/>${this.formatDate(point.timestamp)}`;
          }
        },
        grid: { left: 60, right: 20, top: 40, bottom: 40 },
        xAxis: {
          type: 'category',
          data: series.map(s => s.round),
          name: '实验轮次',
          nameLocation: 'middle',
          nameGap: 25
        },
        yAxis: {
          type: 'value',
          name: metric,
          nameLocation: 'middle',
          nameGap: 45
        },
        series: [{
          type: 'line',
          data: series.map(s => s.value),
          smooth: true,
          symbolSize: 6,
          itemStyle: {
            color: (params) => {
              return series[params.dataIndex].value === bestValue
                ? '#10b981' : '#3b82f6';
            }
          },
          lineStyle: { color: '#3b82f6' },
          markPoint: {
            data: [
              dir === 'maximize'
                ? { type: 'max', name: '最大值' }
                : { type: 'min', name: '最小值' }
            ],
            label: { formatter: '{c}' }
          }
        }]
      });
    },

    disposeChart() {
      if (this.progressChart) {
        this.progressChart.dispose();
        this.progressChart = null;
      }
    },

    // ══════════════ DAG 流程图 ══════════════
    // 拓扑与 Cairn 一致：fact 是节点，intent 是边。
    // 已结论的 intent → from 各 fact 指向它产出的 fact；
    // 未结论的 intent → 建一个占位节点，from 指向它，表示"正在探索的分支"。

    buildElements() {
      const nodes = [];
      const edges = [];
      const factIds = new Set(this.facts.map(f => f.id));

      for (const f of this.facts) {
        const nodeType = f.id === 'origin' ? 'origin' : f.id === 'goal' ? 'goal' : 'fact';
        const label = this.summarizeFactLabel(f);
        const size = this.factNodeSize(label, nodeType);
        nodes.push({ data: {
          id: f.id,
          label,
          description: f.description,
          nodeType,
          isBest: this.isBestFact(f) ? 1 : 0,
          width: size.width,
          height: size.height,
        }});
      }

      for (const intent of this.intents) {
        const label = this.edgeLabel(intent);
        const sources = (intent.from || []).filter(src => factIds.has(src));
        if (intent.to && factIds.has(intent.to)) {
          for (const src of sources) {
            edges.push({ data: {
              id: `${intent.id}_${src}`, source: src, target: intent.to,
              intentId: intent.id, label, status: 'concluded',
            }});
          }
          continue;
        }
        // 未结论：占位节点承载"进行中 / 待领取"状态
        const phId = `_ph_${intent.id}`;
        const nodeType = this.openIntentNodeType(intent);
        const size = this.openIntentNodeSize(intent);
        nodes.push({ data: {
          id: phId,
          label: this.openIntentNodeLabel(intent),
          description: intent.description,
          nodeType,
          intentId: intent.id,
          width: size.width,
          height: size.height,
        }});
        for (const src of sources) {
          edges.push({ data: {
            id: `${intent.id}_${src}`, source: src, target: phId,
            intentId: intent.id, label, status: nodeType,
          }});
        }
        // bootstrap 的语义是"为整个目标铺路"，用虚线连到 goal 表达作用域
        if (this.isBootstrapIntent(intent) && factIds.has('goal')) {
          edges.push({ data: {
            id: `${intent.id}_goal`, source: phId, target: 'goal',
            intentId: intent.id, label: '', status: nodeType, edgeType: 'bootstrap_scope',
          }});
        }
      }
      return { nodes, edges };
    },

    isBootstrapIntent(intent) {
      return Boolean(
        intent
        && intent.description === 'bootstrap'
        && (intent.creator || '').startsWith('dispatcher.bootstrap')
        && Array.isArray(intent.from)
        && intent.from.length === 1
        && intent.from[0] === 'origin'
        && !intent.to
      );
    },

    openIntentNodeType(intent) {
      if (this.isBootstrapIntent(intent)) return intent.worker ? 'bootstrap_running' : 'bootstrap_pending';
      return intent.worker ? 'in_progress' : 'unclaimed';
    },

    openIntentNodeLabel(intent) {
      if (this.isBootstrapIntent(intent)) return 'Bootstrap';
      // 显示 intent 的描述摘要
      const text = (intent.description || '').replace(/\s+/g, ' ').trim();
      const chars = Array.from(text);
      return chars.length <= 40 ? text : `${chars.slice(0, 40).join('')}…`;
    },

    openIntentNodeSize(intent) {
      if (this.isBootstrapIntent(intent)) return { width: 82, height: 30 };
      // intent 节点也使用动态尺寸，根据描述长度计算
      const label = this.openIntentNodeLabel(intent);
      const preset = { fontSize: 9, maxTextWidth: 140, minWidth: 80, minHeight: 32, padX: 10, padY: 8 };
      const m = this.measureWrappedText(label, preset.maxTextWidth, preset.fontSize);
      return {
        width: Math.max(preset.minWidth, Math.ceil(m.width + preset.padX * 2)),
        height: Math.max(preset.minHeight, Math.ceil(m.height + preset.padY * 2)),
      };
    },

    edgeLabel(intent) {
      const text = (intent.description || '').replace(/\s+/g, ' ').trim();
      const chars = Array.from(text);
      return chars.length <= 30 ? text : `${chars.slice(0, 30).join('')}…`;
    },

    summarizeFactLabel(fact) {
      if (fact.id === 'origin') return '起点';
      if (fact.id === 'goal') return '目标';
      // fact 节点优先显示目标指标，一眼看出这条分支跑出什么数
      const key = this.currentProject?.goal_metric;
      const val = key ? fact.metrics?.[key] : null;
      const text = (fact.description || '').replace(/\s+/g, ' ').trim();
      const chars = Array.from(text);
      const head = chars.length <= 22 ? (text || fact.id) : `${chars.slice(0, 22).join('')}…`;
      return val == null ? head : `${head}\n${key} ${this.formatMetric(val)}`;
    },

    factNodeSize(label, nodeType) {
      const preset = nodeType === 'fact'
        ? { fontSize: 10, maxTextWidth: 128, minWidth: 60, minHeight: 34, padX: 10, padY: 10 }
        : { fontSize: 11, maxTextWidth: 92, minWidth: 58, minHeight: 38, padX: 10, padY: 10 };
      const m = this.measureWrappedText(label, preset.maxTextWidth, preset.fontSize);
      return {
        width: Math.max(preset.minWidth, Math.ceil(m.width + preset.padX * 2)),
        height: Math.max(preset.minHeight, Math.ceil(m.height + preset.padY * 2)),
      };
    },

    // 中英混排的粗略量测：CJK 按 1 字宽，ASCII 按 0.58 字宽
    measureWrappedText(text, maxWidth, fontSize) {
      const content = (text || '').trim() || ' ';
      const lineHeight = Math.ceil(fontSize * 1.35);
      let maxLine = 0, lineW = 0, lines = 1;
      for (const ch of content) {
        if (ch === '\n') { maxLine = Math.max(maxLine, lineW); lineW = 0; lines++; continue; }
        const w = /[一-鿿　-〿＀-￯]/.test(ch) ? fontSize : fontSize * 0.58;
        if (lineW + w > maxWidth) { maxLine = Math.max(maxLine, lineW); lineW = w; lines++; }
        else lineW += w;
      }
      maxLine = Math.max(maxLine, lineW);
      return { width: Math.min(maxWidth, Math.ceil(maxLine)), height: lines * lineHeight };
    },

    initGraph() {
      const container = document.getElementById('cy');
      if (!container || typeof cytoscape === 'undefined') return;
      if (this.cy) return;
      // 容器被 x-show 隐藏时尺寸是 0×0，cytoscape 会建出一个量不到画布的实例，
      // 之后即使容器显示出来也只是一片空白。宁可不建，等真正上屏再建。
      if (container.clientWidth === 0 || container.clientHeight === 0) return;
      const { nodes, edges } = this.buildElements();
      const cy = cytoscape({
        container,
        elements: [...nodes, ...edges],
        style: this.graphStyles(),
        layout: this.layoutOpts(false),
        minZoom: 0.15,
        maxZoom: 3.5,
      });
      this.cy = cy;
      cy.on('tap', 'node', e => {
        const d = e.target.data();
        if (d.intentId) this.selectIntent(d.intentId);
        else this.selectFact(d.id);
        this.refreshGraphDecorations();
      });
      cy.on('tap', 'edge', e => {
        const iid = e.target.data('intentId');
        if (iid) this.selectIntent(iid);
        this.refreshGraphDecorations();
      });
      cy.on('tap', e => {
        if (e.target === cy) { this.selectedNode = null; this.refreshGraphDecorations(); }
      });
      this.refreshGraphDecorations();
      this.setupAutoFit();
    },

    graphStyles() {
      const common = {
        'text-valign': 'center', 'text-halign': 'center',
        'font-family': '-apple-system,BlinkMacSystemFont,PingFang SC,Microsoft YaHei,sans-serif',
      };
      const box = {
        ...common, shape: 'round-rectangle', label: 'data(label)', color: '#fff',
        'font-weight': 'bold', 'text-wrap': 'wrap', 'text-overflow-wrap': 'anywhere',
        width: 'data(width)', height: 'data(height)', 'border-width': 0,
      };
      const edgeBase = {
        'target-arrow-shape': 'triangle', 'curve-style': 'bezier', label: 'data(label)',
        'font-size': '7px', 'text-rotation': 'autorotate', 'text-margin-y': -9,
        'text-max-width': '90px', 'text-wrap': 'ellipsis', 'text-background-opacity': 0.85,
        'text-background-padding': '2px', 'text-events': 'yes',
      };
      return [
        { selector: 'node[nodeType="origin"]', style: { ...box, 'background-color': '#14b8a6', 'font-size': '11px', 'text-max-width': '92px' }},
        { selector: 'node[nodeType="goal"]', style: { ...box, 'background-color': '#f43f5e', 'font-size': '11px', 'text-max-width': '92px' }},
        { selector: 'node[nodeType="fact"]', style: { ...box, 'background-color': '#6366f1', 'font-size': '10px', 'text-max-width': '128px' }},
        { selector: 'node[nodeType="fact"][isBest=1]', style: { 'background-color': '#10b981', 'border-width': 2.5, 'border-color': '#047857' }},
        { selector: 'node[nodeType="in_progress"]', style: { ...box, 'background-color': '#f59e0b', color: '#fff', 'font-size': '9px', 'text-max-width': '140px', 'border-width': 2, 'border-color': '#d97706' }},
        { selector: 'node[nodeType="unclaimed"]', style: { ...box, 'background-color': '#e0e7ff', color: '#6366f1', 'font-size': '9px', 'text-max-width': '140px', 'border-width': 1.5, 'border-color': '#a5b4fc', 'border-style': 'dashed' }},
        { selector: 'node[nodeType="bootstrap_pending"]', style: { ...box, 'background-color': '#fff7ed', color: '#c2410c', 'font-size': '10px', 'border-width': 1.5, 'border-color': '#fdba74', 'border-style': 'dashed', 'text-max-width': '70px' }},
        { selector: 'node[nodeType="bootstrap_running"]', style: { ...box, 'background-color': '#fb923c', color: '#fff7ed', 'font-size': '10px', 'border-width': 2, 'border-color': '#ea580c', 'text-max-width': '70px' }},

        { selector: 'edge[status="concluded"]', style: { ...edgeBase, width: 2, 'line-color': '#6ee7b7', 'target-arrow-color': '#6ee7b7', color: '#94a3b8', 'text-background-color': '#f8fafc', 'arrow-scale': 0.9 }},
        { selector: 'edge[status="in_progress"]', style: { ...edgeBase, width: 2, 'line-color': '#fbbf24', 'line-style': 'dashed', 'line-dash-pattern': [8, 4], 'target-arrow-color': '#fbbf24', color: '#b45309', 'text-background-color': '#fffbeb', 'arrow-scale': 0.9 }},
        { selector: 'edge[status="unclaimed"]', style: { ...edgeBase, width: 1.5, 'line-color': '#cbd5e1', 'line-style': 'dashed', 'line-dash-pattern': [5, 5], 'target-arrow-color': '#cbd5e1', color: '#94a3b8', 'text-background-color': '#f8fafc', 'arrow-scale': 0.7 }},
        { selector: 'edge[status="bootstrap_pending"]', style: { ...edgeBase, width: 2, 'line-color': '#fdba74', 'line-style': 'dashed', 'line-dash-pattern': [8, 4], 'target-arrow-color': '#fdba74', color: '#c2410c', 'text-background-color': '#fff7ed', 'arrow-scale': 0.85 }},
        { selector: 'edge[status="bootstrap_running"]', style: { ...edgeBase, width: 2.5, 'line-color': '#fb923c', 'line-style': 'dashed', 'line-dash-pattern': [10, 4], 'target-arrow-color': '#fb923c', color: '#c2410c', 'text-background-color': '#fff7ed', 'arrow-scale': 0.95 }},
        { selector: 'edge[edgeType="bootstrap_scope"]', style: { label: '', width: 1.8, 'line-style': 'dotted', 'line-dash-pattern': [2, 5], 'arrow-scale': 0.75 }},

        { selector: 'node.focus', style: { 'border-width': 3, 'border-color': '#312e81', 'border-opacity': 0.95, 'z-index': 1000 }},
        { selector: 'edge.focus', style: { 'z-index': 1000, 'overlay-color': '#93c5fd', 'overlay-opacity': 0.22, 'overlay-padding': 5 }},
        { selector: '.faded', style: { opacity: 0.45 }},
      ];
    },

    layoutEngine(mode = this.layoutMode) {
      if (mode.startsWith('elk')) return 'elk';
      return mode.startsWith('klay') ? 'klay' : 'dagre';
    },

    layoutDirection(mode = this.layoutMode) {
      return mode.endsWith('_lr') ? 'LR' : 'TB';
    },

    layoutOpts(animate = true) {
      const direction = this.layoutDirection();
      const engine = this.layoutEngine();
      const base = { fit: true, padding: 50, animate, animationDuration: 350, animationEasing: 'ease-in-out-cubic' };

      if (engine === 'elk') {
        return { ...base, name: 'elk', elk: {
          algorithm: 'layered',
          'elk.direction': direction === 'TB' ? 'DOWN' : 'RIGHT',
          'elk.aspectRatio': '1.5',
          'elk.layered.nodePlacement.strategy': 'BRANDES_KOEPF',
          'elk.spacing.nodeNode': '50',
          'elk.layered.spacing.nodeNodeBetweenLayers': '80',
          'elk.spacing.edgeNode': '25',
          'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
          'elk.layered.nodePlacement.bk.fixedAlignment': 'BALANCED',
        }};
      }
      if (engine === 'klay') {
        return { ...base, name: 'klay', klay: {
          direction: direction === 'LR' ? 'RIGHT' : 'DOWN',
          spacing: 45,
          inLayerSpacingFactor: 1.1,
          nodePlacement: 'BRANDES_KOEPF',
          fixedAlignment: 'BALANCED',
          edgeRouting: 'ORTHOGONAL',
          thoroughness: 8,
        }};
      }
      return { ...base, name: 'dagre', rankDir: direction, nodeSep: 45, rankSep: 70, edgeSep: 12, ranker: 'network-simplex' };
    },

    applySelectedLayout() {
      if (!this.cy) return;
      this.cy.layout(this.layoutOpts()).run();
    },

    fitGraph() {
      if (this.cy) this.cy.fit(undefined, 50);
    },

    // 增量更新：只加/删/改变化的元素，布局有变化才重跑，避免每次轮询整图跳动
    updateGraph() {
      if (!this.cy) return;
      const { nodes, edges } = this.buildElements();
      const wantNodes = new Set(nodes.map(n => n.data.id));
      const wantEdges = new Set(edges.map(e => e.data.id));
      let changed = false;

      this.cy.nodes().forEach(n => { if (!wantNodes.has(n.id())) { n.remove(); changed = true; } });
      this.cy.edges().forEach(e => { if (!wantEdges.has(e.id())) { e.remove(); changed = true; } });

      for (const n of nodes) {
        const ex = this.cy.getElementById(n.data.id);
        if (ex.length === 0) {
          this.cy.add(n);
          changed = true;
        } else if (
          ex.data('nodeType') !== n.data.nodeType ||
          ex.data('label') !== n.data.label ||
          ex.data('isBest') !== n.data.isBest ||
          ex.data('width') !== n.data.width ||
          ex.data('height') !== n.data.height
        ) {
          ex.data(n.data);
          changed = true;
        }
      }
      for (const e of edges) {
        const ex = this.cy.getElementById(e.data.id);
        if (ex.length === 0) { this.cy.add(e); changed = true; }
        else if (ex.data('status') !== e.data.status || ex.data('label') !== e.data.label) ex.data(e.data);
      }
      if (changed) this.cy.layout(this.layoutOpts()).run();
      this.refreshGraphDecorations();
    },

    // 选中高亮：选中节点及其邻接边加 focus，其余淡出
    refreshGraphDecorations() {
      if (!this.cy) return;
      this.cy.elements().removeClass('focus faded');
      const sel = this.selectedNode;
      if (!sel) return;

      let focused = this.cy.collection();
      if (sel.type === 'fact') {
        focused = this.cy.getElementById(sel.id);
      } else {
        focused = this.cy.elements().filter(el => el.data('intentId') === sel.id);
      }
      if (focused.length === 0) return;
      const neighborhood = focused.union(focused.connectedEdges()).union(focused.connectedNodes());
      focused.addClass('focus');
      this.cy.elements().difference(neighborhood).addClass('faded');
    },

    setupAutoFit() {
      this.teardownAutoFit();
      const container = document.getElementById('cy');
      if (!container || !this.cy || typeof ResizeObserver === 'undefined') return;
      let timer = null;
      this._resizeObserver = new ResizeObserver(() => {
        clearTimeout(timer);
        timer = setTimeout(() => {
          if (!this.cy) return;
          this.cy.resize();
          this.cy.fit(undefined, 50);
        }, 200);
      });
      this._resizeObserver.observe(container);
    },

    teardownAutoFit() {
      if (this._resizeObserver) {
        this._resizeObserver.disconnect();
        this._resizeObserver = null;
      }
    },

    disposeGraph() {
      this.teardownAutoFit();
      if (this.cy) {
        this.cy.destroy();
        this.cy = null;
      }
    }
  };
}
