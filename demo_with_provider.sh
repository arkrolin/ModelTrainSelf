#!/bin/bash
# 启动带 Provider 配置的演示项目
source /root/anaconda3/bin/activate mts

echo "=== ModelTrainSelf 演示模式 ==="
echo "这将创建一个示例项目并启动 Web 界面"
echo ""
echo "访问地址: http://127.0.0.1:8765"
echo "按 Ctrl+C 停止服务器"
echo ""

mts demo --port 8765 --trials 20 --open-browser
