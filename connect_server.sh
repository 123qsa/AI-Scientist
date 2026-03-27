#!/bin/bash
# AI-Scientist 云端服务器连接脚本

SERVER_IP="166.111.86.21"
SERVER_USER="hanjiajun"
SERVER_PORT="22"
SSH_KEY="~/Desktop/服务器公私钥/id_ed25526574_qq_com"

echo "Connecting to AI-Scientist server..."
ssh -i ${SSH_KEY} -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_IP}
