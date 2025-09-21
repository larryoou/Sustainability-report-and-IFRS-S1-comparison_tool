#!/bin/bash

# IFRS S1 ESG分析器 - macOS雙擊啟動文件
# 作者: 游士弘
# 功能: 一鍵啟動所有服務並自動打開瀏覽器

echo "🚀 IFRS S1 ESG分析器啟動中..."
echo "📋 將啟動以下服務:"
echo "   • 後端加速服務 (8004)"
echo "   • 本地語義代理 (8000)" 
echo "   • 前端網頁服務 (9000-9005)"
echo "   • 自動打開瀏覽器"
echo ""

# 獲取腳本所在目錄
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# 檢查套件目錄是否存在
if [ ! -d "IFRS_S1_Package" ]; then
    echo "❌ 錯誤: 找不到 IFRS_S1_Package 目錄"
    echo "請確保此腳本位於正確的目錄中"
    echo ""
    echo "按任意鍵退出..."
    read -n 1
    exit 1
fi

# 執行啟動腳本
echo "🔄 正在啟動服務..."

# 啟動服務（背景執行）
bash "IFRS_S1_Package/scripts/start.sh" &
START_PID=$!

# 等待服務啟動
echo "⏳ 等待服務啟動..."
sleep 8

# 檢查服務狀態並強制打開瀏覽器
FRONTEND_URL=""
for port in 9000 9001 9002 9003 9004 9005; do
    if lsof -i:$port >/dev/null 2>&1; then
        FRONTEND_URL="http://localhost:$port/ifrs_s1_auto_keywords_tool.html"
        echo "✅ 找到前端服務在端口 $port"
        break
    fi
done

if [ -n "$FRONTEND_URL" ]; then
    echo "🌐 正在打開瀏覽器: $FRONTEND_URL"
    
    # 強制打開瀏覽器（多種方法確保成功）
    if command -v open >/dev/null 2>&1; then
        # macOS - 使用 open 命令
        open "$FRONTEND_URL"
        sleep 1
        # 備用方法：使用 Safari
        osascript -e "tell application \"Safari\" to open location \"$FRONTEND_URL\"" 2>/dev/null || true
        # 備用方法：使用 Chrome
        osascript -e "tell application \"Google Chrome\" to open location \"$FRONTEND_URL\"" 2>/dev/null || true
    fi
    
    echo "✅ 瀏覽器應該已經打開"
    echo "📱 如果沒有自動打開，請手動訪問: $FRONTEND_URL"
else
    echo "❌ 前端服務未啟動，請檢查日誌"
fi

# 等待背景啟動腳本完成
wait $START_PID

echo ""
echo "🛑 按任意鍵停止所有服務並關閉..."
read -n 1

# 停止服務
echo ""
echo "🔄 正在停止服務..."
if [ -f "IFRS_S1_Package/scripts/stop.sh" ]; then
    bash "IFRS_S1_Package/scripts/stop.sh"
else
    echo "⚠️  找不到停止腳本，請手動關閉服務"
fi

echo "✅ 已退出"
