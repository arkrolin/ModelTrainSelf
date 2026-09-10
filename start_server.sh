#!/bin/bash
# 启动 ModelTrainSelf 服务器
source /root/anaconda3/bin/activate mts

echo "=== ModelTrainSelf 服务器启动 ==="
echo "访问地址: http://127.0.0.1:8765"
echo "按 Ctrl+C 停止服务器"
echo ""

mts serve --host 127.0.0.1 --port 8765
