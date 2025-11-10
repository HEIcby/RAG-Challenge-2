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

### 端口被占用

```bash
# 查看端口占用
sudo lsof -i :8501

# 或使用 netstat
sudo netstat -tulpn | grep 8501

# 杀死进程
sudo kill -9 <PID>
```

### 防火墙问题

```bash
# 检查防火墙状态
sudo ufw status

# 允许端口
sudo ufw allow 8501/tcp
```

### 权限问题

```bash
# 确保有执行权限
chmod +x scripts/start_frontend.sh

# 确保数据目录可写
chmod -R 755 data/
```

### 查看日志

```bash
# Streamlit 日志
tail -f ~/.streamlit/logs/*.log

# systemd 日志
sudo journalctl -u streamlit -f --since "1 hour ago"

# Nginx 日志
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
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
1. 查看日志文件
2. 检查防火墙和端口
3. 确认所有依赖已安装
4. 验证 API 密钥配置

---

**部署日期**: 2025-11-06  
**维护者**: Ocean Chen
