# 🚀 部署指南

> 本指南包含本地部署和服务器部署两部分

## 📋 目录
1. [本地部署](#本地部署)
2. [服务器部署](#服务器部署)
3. [故障排查](#故障排查)

---

## 🏠 本地部署

### 快速开始（3步完成）

**1. 安装依赖**
```bash
./scripts/setup/install_streamlit.sh
```

**2. 配置API密钥**
```bash
cp .env.example .env
nano .env  # 填入你的 API keys
```

**3. 启动应用**
```bash
./scripts/start_frontend.sh
```

访问 http://localhost:8501

**完成！** 🎉

### 详细说明

查看主 [README.md](../../README.md) 的"快速开始"部分了解详细步骤。

---

## 🌐 服务器部署

### 📋 目录
1. [服务器要求](#服务器要求)
2. [SSH 连接设置](#ssh-连接设置)
3. [环境配置](#环境配置)
4. [项目部署](#项目部署)
5. [公网访问配置](#公网访问配置)
6. [后台运行与监控](#后台运行与监控)

---

## 🖥️ 服务器要求

### 最低配置
- **操作系统**: Debian 10+ / Ubuntu 20.04+
- **CPU**: 2 核
- **内存**: 4GB RAM
- **存储**: 20GB 可用空间
- **Python**: 3.10+（推荐 3.12）

### 网络要求
- **端口**: 8501 (Streamlit 默认)
- **带宽**: 建议 10Mbps+
- **公网 IP**: 需要配置端口映射

---

## 🔑 SSH 连接设置

### 1. 生成 SSH 密钥（本地操作）

如果您还没有 SSH 密钥：

```bash
# 生成 ED25519 密钥（推荐）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 或生成 RSA 密钥（兼容性更好）
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# 查看公钥
cat ~/.ssh/id_ed25519.pub
# 或
cat ~/.ssh/id_rsa.pub
```

### 2. 提供公钥给服务器管理员

**您的当前公钥：**
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPU8zkWYdMcU1QeciKm81IdET2G0IC77G5/vyXImIwyA 13732580643@163.com
```

将这整行发送给管理员，他们会添加到服务器的 `~/.ssh/authorized_keys`。

### 3. 测试连接

```bash
# 管理员提供服务器地址后
ssh username@server_ip

# 例如：
ssh ocean@192.168.1.100
```

---

## 🐧 Debian 环境配置

### 1. 连接到服务器后，更新系统

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. 安装 Python 3.12

```bash
# 添加 deadsnakes PPA（如果是 Ubuntu）
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# 安装 Python 3.12
sudo apt install python3.12 python3.12-venv python3.12-dev -y

# 验证安装
python3.12 --version
```

**如果是 Debian 12+**：
```bash
# Debian 12 自带 Python 3.11，也可以使用
sudo apt install python3 python3-venv python3-pip -y
```

### 3. 安装系统依赖

```bash
# 安装必要的系统包
sudo apt install -y \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev
```

---

## 📦 项目部署

### 1. 上传项目到服务器

**方法 A: 使用 Git（推荐）**
```bash
# 在服务器上
cd ~
git clone https://github.com/IlyaRice/RAG-Challenge-2.git
cd RAG-Challenge-2
```

**方法 B: 使用 SCP 上传**
```bash
# 在本地电脑上
scp -r /path/to/RAG-Challenge-2 username@server_ip:~/
```

**方法 C: 使用 rsync（推荐，支持增量）**
```bash
# 在本地电脑上
rsync -avz --progress \
    --exclude 'venv*' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    /Users/oceanchen/Library/CloudStorage/GoogleDrive-*/RAG-Challenge-2/ \
    username@server_ip:~/RAG-Challenge-2/
```

### 2. 创建虚拟环境

```bash
cd ~/RAG-Challenge-2

# 使用 Python 3.12
python3.12 -m venv venv_streamlit

# 激活虚拟环境
source venv_streamlit/bin/activate

# 升级 pip
pip install --upgrade pip
```

### 3. 安装依赖

```bash
# 安装 Streamlit 和依赖
pip install -r config/requirements-frontend.txt

# 或完整安装
pip install -r requirements.txt
```

### 4. 配置 API 密钥

```bash
# 编辑配置文件
nano config/api_config.json

# 或设置环境变量
export OPENAI_API_KEY="your-api-key-here"
export DASHSCOPE_API_KEY="your-qwen-key-here"
```

### 5. 测试运行

```bash
# 测试启动
streamlit run app_jinpan_qa.py --server.port 8501
```

在浏览器访问：`http://server_ip:8501`

---

## 🌐 公网访问配置

### 方案 1: 使用 Nginx 反向代理（推荐）

#### 1. 安装 Nginx

```bash
sudo apt install nginx -y
```

#### 2. 配置 Nginx

```bash
sudo nano /etc/nginx/sites-available/streamlit
```

添加以下配置：

```nginx
server {
    listen 80;
    server_name your-domain.com;  # 或使用服务器 IP

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

#### 3. 启用配置

```bash
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 4. 配置防火墙

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 方案 2: 直接端口映射

如果服务器在内网，需要在路由器配置端口转发：

```
外网端口 80   → 服务器 IP:8501
外网端口 443  → 服务器 IP:8501
```

### 方案 3: 使用 frp 内网穿透

如果没有公网 IP：

```bash
# 在服务器上安装 frp 客户端
wget https://github.com/fatedier/frp/releases/download/v0.51.3/frp_0.51.3_linux_amd64.tar.gz
tar -xzf frp_0.51.3_linux_amd64.tar.gz
cd frp_0.51.3_linux_amd64

# 配置 frpc.ini
nano frpc.ini
```

```ini
[common]
server_addr = your_frp_server_ip
server_port = 7000
token = your_token

[streamlit]
type = tcp
local_ip = 127.0.0.1
local_port = 8501
remote_port = 6000
```

---

## 🔄 后台运行与监控

### 方法 1: 使用 systemd（推荐）

#### 1. 创建 systemd 服务

```bash
sudo nano /etc/systemd/system/streamlit.service
```

```ini
[Unit]
Description=Streamlit RAG Challenge Frontend
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/RAG-Challenge-2
Environment="PATH=/home/your_username/RAG-Challenge-2/venv_streamlit/bin"
ExecStart=/home/your_username/RAG-Challenge-2/venv_streamlit/bin/streamlit run app_jinpan_qa.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 2. 启动服务

```bash
# 重载 systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start streamlit

# 设置开机自启
sudo systemctl enable streamlit

# 查看状态
sudo systemctl status streamlit

# 查看日志
sudo journalctl -u streamlit -f
```

### 方法 2: 使用 screen

```bash
# 安装 screen
sudo apt install screen -y

# 创建新会话
screen -S streamlit

# 启动应用
cd ~/RAG-Challenge-2
source venv_streamlit/bin/activate
streamlit run app_jinpan_qa.py --server.port 8501

# 按 Ctrl+A 然后 D 退出会话（保持运行）

# 重新连接
screen -r streamlit

# 查看所有会话
screen -ls
```

### 方法 3: 使用 tmux

```bash
# 安装 tmux
sudo apt install tmux -y

# 创建会话
tmux new -s streamlit

# 启动应用
cd ~/RAG-Challenge-2
source venv_streamlit/bin/activate
streamlit run app_jinpan_qa.py --server.port 8501

# 按 Ctrl+B 然后 D 退出会话

# 重新连接
tmux attach -t streamlit
```

---

## 🔧 故障排查

### 常见问题与解决方案

#### 问题 1: ModuleNotFoundError: No module named 'rank_bm25'

**错误信息**:
```python
ModuleNotFoundError: No module named 'rank_bm25'
  File "src/retrieval.py", line 4, in <module>
    from rank_bm25 import BM25Okapi
```

**原因**: `rank_bm25` 依赖未安装或 requirements.txt 中缺失

**解决方案**:
```bash
# 激活虚拟环境
source venv_streamlit/bin/activate

# 安装缺失的包
pip install rank-bm25

# 或使用清华镜像加速
pip install rank-bm25 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

#### 问题 2: OSError: [Errno 28] No space left on device

**错误信息**:
```
OSError: [Errno 28] No space left on device
```

**原因**: 磁盘空间不足，通常是因为：
- 依赖包过大（如 torch, docling）
- 临时文件占用空间
- 备份文件累积

**解决方案**:
```bash
# 1. 检查磁盘使用情况
df -h
du -sh /root/* | sort -h

# 2. 清理不必要的备份
rm -rf /root/*_backup_*

# 3. 清理 pip 缓存
pip cache purge

# 4. 清理临时文件
rm -rf /tmp/*

# 5. 只安装必需依赖（跳过 docling 等大型包）
pip install aiohttp tiktoken python-dotenv pydantic openai \
    requests tqdm rank-bm25 tabulate pyprojroot PyPDF2 \
    faiss-cpu langchain json_repair click httpx PyMuPDF \
    -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

**预防措施**:
- 确保至少有 20GB 可用空间
- 定期清理日志和缓存
- 避免安装非必需的大型依赖

#### 问题 3: bash: rsync: command not found

**错误信息**:
```bash
bash: line 1: rsync: command not found
```

**原因**: Debian 服务器默认未安装 rsync

**解决方案 A**: 安装 rsync
```bash
sudo apt update
sudo apt install rsync -y
```

**解决方案 B**: 使用 tar 压缩传输（推荐）
```bash
# 本地压缩
cd data
tar czf val_set.tar.gz val_set/

# 上传
scp val_set.tar.gz root@server:/path/to/data/

# 服务器解压
tar xzf val_set.tar.gz
rm val_set.tar.gz
```

**性能对比**:
- rsync: 适合增量同步，但需要双方都安装
- tar + scp: 适合首次全量传输，压缩比 3.5:1

#### 问题 4: 依赖安装速度慢

**症状**: pip install 速度很慢（<100KB/s）

**原因**: 使用默认的 PyPI 源

**解决方案**:
```bash
# 使用清华镜像源（推荐）
pip install <package> -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 或永久配置
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 阿里云镜像（备选）
pip install <package> -i https://mirrors.aliyun.com/pypi/simple/
```

**速度提升**: 从 1-5MB/s 提升到 30-50MB/s

#### 问题 5: 端口被占用

```bash
# 查看端口占用
sudo lsof -i :8501

# 或使用 ss（推荐）
ss -tlnp | grep 8501

# 杀死进程
pkill -f 'streamlit run app_jinpan_qa.py'

# 或使用 PID
sudo kill -9 <PID>
```

#### 问题 6: 防火墙问题

```bash
# 检查防火墙状态
sudo ufw status

# 允许端口
sudo ufw allow 8501/tcp

# 检查 iptables
sudo iptables -L -n | grep 8501
```

#### 问题 7: 权限问题

```bash
# 确保有执行权限
chmod +x scripts/start_frontend.sh

# 确保数据目录可写
chmod -R 755 data/

# 检查虚拟环境权限
ls -la venv_streamlit/bin/python
```

### 查看日志

```bash
# Streamlit 应用日志
tail -f streamlit.log

# 最近 100 行
tail -100 streamlit.log

# 搜索错误
grep -i error streamlit.log

# systemd 日志（如果使用 systemd）
sudo journalctl -u streamlit -f --since "1 hour ago"

# Nginx 日志（如果使用 Nginx）
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 调试技巧

```bash
# 1. 检查 Python 环境
which python
python --version

# 2. 检查依赖安装
pip list | grep streamlit
pip list | grep rank-bm25

# 3. 测试 API 连接
python -c "import openai; print('OpenAI installed')"

# 4. 验证端口监听
curl -I http://localhost:8501

# 5. 查看进程资源占用
ps aux | grep streamlit
top -p $(pgrep -f streamlit)
```

---

## 📊 性能优化

### 1. 配置 Streamlit

创建 `~/.streamlit/config.toml`：

```toml
[server]
port = 8501
address = "0.0.0.0"
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[runner]
magicEnabled = true
fastReruns = true
```

### 2. 使用 Gunicorn（可选）

```bash
pip install gunicorn

# 启动
gunicorn -w 4 -b 0.0.0.0:8501 your_app:app
```

### 3. 配置 Nginx 缓存

在 Nginx 配置中添加：

```nginx
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=streamlit_cache:10m max_size=1g;
proxy_cache streamlit_cache;
proxy_cache_valid 200 1h;
```

---

## 🔒 安全建议

1. **使用 HTTPS**：配置 Let's Encrypt SSL 证书
   ```bash
   sudo apt install certbot python3-certbot-nginx -y
   sudo certbot --nginx -d your-domain.com
   ```

2. **限制访问**：配置 Nginx 认证
   ```bash
   sudo apt install apache2-utils -y
   sudo htpasswd -c /etc/nginx/.htpasswd admin
   ```

3. **防火墙**：只开放必要端口
   ```bash
   sudo ufw default deny incoming
   sudo ufw default allow outgoing
   sudo ufw allow ssh
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   ```

4. **定期更新**：
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

---

## 📱 访问地址

部署完成后，可以通过以下方式访问：

- **本地**: `http://localhost:8501`
- **局域网**: `http://服务器IP:8501`
- **公网（Nginx）**: `http://your-domain.com`
- **公网（端口映射）**: `http://公网IP:映射端口`

---

## 🆘 获取帮助

如果遇到问题：
1. 查看 [故障排查](#故障排查) 部分
2. 检查日志文件（`streamlit.log`）
3. 查看 [实战部署案例](#实战部署案例)
4. 在 GitHub 提交 Issue

---

## 📖 实战部署案例

### 10.222.4.30 服务器部署实录

查看完整的生产服务器部署过程：  
👉 **[DEPLOYMENT_10.222.4.30.md](../DEPLOYMENT_10.222.4.30.md)**

包含内容：
- ✅ 完整的138分钟部署时间线
- 🐛 7个实际遇到的问题和解决方案
- 📊 磁盘空间、数据传输等性能数据
- 🔧 维护命令和调试技巧
- 💡 关键经验总结

**推荐阅读**: 在部署前先阅读实战案例，可以避免大部分常见问题。

---

**部署日期**: 2025-11-06  
**最后更新**: 2025-11-10  
**维护者**: Ocean Chen
