# Crab 云端部署指南

## 前置条件

| 项目 | 要求 |
|------|------|
| 操作系统 | Ubuntu 20.04+ (x86_64) |
| 权限 | root |
| 内存 | ≥ 4 GB |
| 磁盘 | ≥ 30 GB 可用空间 |
| 网络 | 公网出口（需访问 Docker Hub 拉取镜像） |
| 内核 | 支持 `CONFIG_CHECKPOINT_RESTORE=y`、cgroup v2、ZFS 模块 |

> 大多数主流云厂商（阿里云、AWS、GCP、Azure）的 Ubuntu 镜像默认满足上述条件。

---

## 一键部署

在目标机器上以 root 执行：

```bash
curl -sL https://raw.githubusercontent.com/open-agent-infra/crab/experimental/scripts/deploy-cloud.sh | bash
```

或克隆仓库后执行：

```bash
git clone --branch experimental https://github.com/open-agent-infra/crab /root/crab
bash /root/crab/scripts/deploy-cloud.sh --tenant myproject --max-sandboxes 20 --max-memory 8G
```

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo` | `https://github.com/open-agent-infra/crab` | 源码仓库 URL |
| `--branch` | `experimental` | 分支名 |
| `--tenant` | `default` | 创建的租户名 |
| `--max-sandboxes` | `20` | 租户最大沙箱数 |
| `--max-memory` | `8G` | 租户最大内存配额 |
| `--gateway-port` | `8900` | Gateway 监听端口 |

---

## 部署完成后

脚本执行成功后会打印连接信息：

```
Endpoint:   http://<公网IP>:8900
API Key:    crab_sk_xxxxxxxxxxxx
Tenant:     default (tn_xxxx)
```

### SDK 安装与连接

在你的本地开发机上：

```bash
pip install crab
```

```python
from crab import Sandbox

sbx = Sandbox(
    base_url="http://<公网IP>:8900",
    api_key="crab_sk_xxxxxxxxxxxx"
)

# 执行命令
result = sbx.exec("echo hello")
print(result.stdout)  # hello

# 完成后销毁
sbx.kill()
```

更多用法参见 [SDK 文档](sdk.md) 和 [examples/sdk/04_remote_service_tutorial.py](../examples/sdk/04_remote_service_tutorial.py)。

---

## 安全组 / 防火墙

部署脚本不修改主机防火墙规则。你需要在云控制台手动放行 Gateway 端口：

| 云厂商 | 操作路径 |
|--------|----------|
| 阿里云 | ECS → 安全组 → 入方向 → 添加规则 → 端口 8900/tcp |
| AWS | EC2 → Security Groups → Inbound Rules → Custom TCP 8900 |
| GCP | VPC → Firewall → Create rule → tcp:8900 |

建议限制来源 IP 为你的客户端地址段，避免公网暴露。

---

## 管理命令

所有管理命令在 crab-server 上以 root 执行。

### 服务状态

```bash
systemctl status crabd crab-gateway
curl -s localhost:8900/healthz
```

### 查看日志

```bash
tail -f /var/lib/crab/logs/daemon.log
tail -f /var/lib/crab/logs/gateway.log
```

### 创建新租户

```bash
crab-gateway tenants create <name> --max-sandboxes 10 --max-memory 4G
```

### 生成 API Key

```bash
# 先获取 tenant id
crab-gateway tenants list
# 然后生成 key
crab-gateway keys create --tenant <tenant_id>
```

### 重启服务

```bash
systemctl restart crabd crab-gateway
```

### 完全重置

```bash
systemctl stop crab-gateway crabd
# 清理所有沙箱状态（不可逆）
rm -rf /var/lib/crab/runtime /var/lib/crab/checkpoints /var/lib/crab/work
zfs destroy -r crab/sandboxes 2>/dev/null; zfs create crab/sandboxes
systemctl start crabd crab-gateway
```

---

## 已知限制

1. **无 TLS/HTTPS**：Gateway 监听明文 HTTP。生产环境建议在前面挂 nginx/caddy 做 TLS 终结，或使用云负载均衡器的 HTTPS 入口。
2. **端口暴露依赖安全组**：脚本不自动操作云防火墙，需手动放行。
3. **单机部署**：当前版本不支持多节点集群，一台机器运行一个 daemon。
4. **无自动升级**：更新代码后需手动 `git pull` + `pip install` + `systemctl restart`。
5. **sandbox 网络默认隔离**：新建 runc sandbox 默认使用独立 netns；只有显式传入 `network=false` 才共享 host 网络。`ports.expose` 不支持 host 网络模式。
