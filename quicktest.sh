#!/bin/bash
# 快速测试脚本 - 创建演示项目并启动 WebUI

source /root/anaconda3/bin/activate mts

echo "=== ModelTrainSelf 快速测试 ==="
echo ""
echo "功能："
echo "  - 创建演示项目"
echo "  - 启动 WebUI (http://127.0.0.1:8765)"
echo "  - 在 WebUI 中手动配置并启动搜索"
echo ""
echo "测试目标: 验证 Worker 配置 UI + Intent 可视化 + Activity 监控"
echo ""

# 检查是否有端口冲突
if lsof -Pi :8765 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  端口 8765 已被占用，请先关闭其他服务"
    exit 1
fi

echo "启动 WebUI..."
echo ""

# 只启动 WebUI，不启动调度器
mts demo --port 8765 --trials 5
