# Crab 多租户管理指南

Crab Gateway 是部署在 daemon 前面的多租户服务入口，负责租户、API Key、
配额和沙箱归属管理。本文介绍如何查看和管理租户。

这里的租户边界属于控制面：Gateway 会隔离资源可见性和操作权限。默认的
独立 network namespace 会隔离各 sandbox 的网络栈和端口空间，但所有
isolated sandbox 目前仍连接同一个 Crab bridge，尚没有按 tenant 阻断东西向
流量，因此不能把它当作不可信租户之间的网络分段。

- 所有租户状态保存在 Gateway 的 SQLite 数据库中（见[数据存储](#数据存储)），
  没有独立的配置文件。
- 所有管理命令只走本机 Unix socket（管理面永不暴露在 TCP 端口上），
  需要在 crab-server 上以运行 Gateway 的同一用户（通常是 root）执行。

---

## 快速上手

```bash
# 1. 创建租户（所有配额参数可选，不设 = 不限）
crab-gateway tenants create myteam --max-sandboxes 20 --max-memory 8G

# 2. 查看租户，拿到 tenant id
crab-gateway tenants list

# 3. 签发 API Key（明文只显示这一次，务必保存）
crab-gateway keys create --tenant tn_xxxxxxxxxxxx
```

客户端随后用 `Bearer crab_sk_...` 请求 Gateway 的 `/v1/*` 接口，
例如通过 [SDK](sdk.md) 连接。

---

## 租户管理

### 创建租户

```bash
crab-gateway tenants create <name> \
  [--max-sandboxes N] \
  [--max-memory 8G] \
  [--max-cpu N] \
  [--admin-socket <path>]
```

| 参数 | 说明 |
|------|------|
| `<name>` | 租户名，全局唯一 |
| `--max-sandboxes N` | 最大存活沙箱数（pending + active），不设 = 不限 |
| `--max-memory 8G` | 所有存活沙箱内存总和上限，支持字节数或 K/M/G/T 后缀，存储为 `max_memory_bytes` |
| `--max-cpu N` | 所有存活沙箱 CPU 总和上限（整数） |
| `--admin-socket` | 管理 socket 路径，一般不用改（见[数据存储](#数据存储)） |

### 查看租户

```bash
crab-gateway tenants list
```

输出示例：

```json
{
  "ok": true,
  "tenants": [
    {
      "id": "tn_01c3ef7f1e68",
      "name": "default",
      "quotas": {
        "max_memory_bytes": 6442450944,
        "max_sandboxes": 20
      }
    }
  ]
}
```

### 修改已有租户的配额

```bash
crab-gateway quotas set --tenant <tenant_id> \
  --max-sandboxes 30 --max-memory 16G --max-cpu 8
```

> **注意：`quotas set` 是整体替换，不是增量更新。** 命令会用你传入的配额
> 完整覆盖该租户原有的 `quota_json`——省略某一项等于**清除**该项上限。
> 想只调大沙箱数、保留内存上限，必须把内存上限也一起写上。

配额变更立即生效，作用于之后的创建/ fork 请求；已存在的沙箱不受影响。

---

## API Key 管理

### 签发

```bash
crab-gateway keys create --tenant <tenant_id>
```

返回示例：

```json
{
  "api_key": "crab_sk_...",
  "key_sha256": "...",
  "tenant_id": "tn_01c3ef7f1e68"
}
```

- 一个租户可以签发多个 Key。
- **明文（`api_key`）只在此处显示这一次。** 数据库里只存 SHA-256 摘要，
  之后无法找回。丢失了就只能新签一个。建议同时记下 `key_sha256`，
  吊销时可以用它定位。

### 吊销

```bash
# 明文或 sha256 摘要均可
crab-gateway keys revoke crab_sk_xxxxxxxx
crab-gateway keys revoke <key_sha256>
```

吊销立即生效，该 Key 之后的请求返回 401。

### 查看现有 Key

CLI 没有 `keys list` 子命令，需要直接查库（只有摘要，没有明文）：

```bash
python3 -c "
import sqlite3
con = sqlite3.connect('/var/lib/crab/gateway/gateway.sqlite3')
for r in con.execute(
    'SELECT key_sha256, tenant_id, created_at, revoked FROM api_keys'):
    print(r)"
```

（service VM 等非 systemd 部署先确认实际的数据库路径，见[数据存储](#数据存储)）

### 认证行为

- 公开接口要求请求头 `Authorization: Bearer crab_sk_...`。
- 未知或已吊销的 Key 返回 **401**。
- 访问其他租户的资源返回 **404**（而非 403），避免泄露资源是否存在。

---

## 配额语义

配额在**沙箱创建 / fork 的那一刻**做预检，按创建时**声明的资源量**
（`resources` claim）记账，**不按运行时的实际用量**：

- `max_sandboxes`：该租户存活（pending + active）沙箱总数不能超过上限。
- `max_memory_bytes`：所有存活沙箱声明的内存之和不能超过上限。
- `max_cpu`：所有存活沙箱声明的 CPU 之和不能超过上限。

由此产生两条规则：

1. **声明制**：租户一旦设置了 `max_memory_bytes` 或 `max_cpu`，其沙箱
   创建时**必须声明**对应的资源量，否则直接拒绝（防止无界沙箱逃过计量）。
   未设置对应上限的租户不受此限制。
2. **声明即占用**：沙箱声明了 2G 内存，即使实际只用了 200M，配额也按 2G
   记账；沙箱被销毁（`killed`）后其声明量立即释放。

资源声明在创建沙箱时通过 `resources` 字段传入，用户侧写法：

```python
from crab import Sandbox

sbx = Sandbox(image="...", resources={"cpus": 2, "memory": "2G"}, engine=engine)
```

`memory` 支持字节数或 K/M/G/T 后缀，内部归一化为 `memory_bytes`
（见 `crab/resources.py`）。fork 子沙箱继承父沙箱的资源声明，同样计入配额。

### 查看配额使用情况

目前只能通过 `tenants list` 看上限，再用数据库统计当前占用：

```bash
python3 -c "
import sqlite3, json
con = sqlite3.connect('/var/lib/crab/gateway/gateway.sqlite3')
for tid, res in con.execute(
    \"SELECT tenant_id, resources_json FROM sandboxes\"
    \" WHERE status IN ('pending', 'active')\"):
    print(tid, json.loads(res))"
```

---

## 沙箱归属

### 收养已有沙箱

将 daemon 侧已存在（例如绕过 Gateway 直接创建的）沙箱挂到某个租户名下：

```bash
crab-gateway sandboxes adopt --tenant <tenant_name_or_id> SANDBOX_ID [SANDBOX_ID ...] \
  [--resources '{"memory":"512M"}']
```

### 查看租户的沙箱

数据库 `sandboxes` 表记录了每个沙箱的归属、状态和资源声明：

```bash
python3 -c "
import sqlite3
con = sqlite3.connect('/var/lib/crab/gateway/gateway.sqlite3')
for r in con.execute(
    'SELECT sandbox_id, tenant_id, status, resources_json FROM sandboxes'):
    print(r)"
```

`status` 取值：`pending` / `active` / `killed` / `lost`，
只有 `pending` 和 `active` 计入配额。

---

## 数据存储

| 项目 | 默认路径 | 覆盖方式 |
|------|----------|----------|
| 注册库（SQLite） | root: `/var/lib/crab/gateway/gateway.sqlite3`；非 root: `~/.local/share/crab/gateway/gateway.sqlite3` | 环境变量 `CRAB_GATEWAY_DATA_DIR` 或 `crab-gateway serve --data-dir` |
| 管理 socket | root: `/run/crab/crab-gateway.sock`；非 root: `$XDG_RUNTIME_DIR/crab/crab-gateway.sock`（无 XDG 时 `~/.cache/crab/...`） | 环境变量 `CRAB_GATEWAY_SOCKET` 或各命令的 `--admin-socket` |

数据库包含三张表：`tenants`（id / name / quota_json）、
`api_keys`（key_sha256 / tenant_id / created_at / revoked）、
`sandboxes`（sandbox_id / tenant_id / status / resources_json 等）。

管理命令都要求 Gateway 进程正在运行（命令通过 socket 与运行中的
Gateway 通信）；如果报 "not reachable"，先确认 Gateway 是否在跑。

---

## 各部署形态下的操作入口

### systemd 云部署（`scripts/deploy-cloud.sh`）

直接在服务器上以 root 执行 `crab-gateway ...` 即可。服务管理：

```bash
systemctl status crabd crab-gateway
```

### Service VM（`tools/vm/provision-service-vm.sh`）

Gateway 跑在 QEMU 虚拟机内，代码在 VM 的 `/root/crab`，通过 PYTHONPATH
加载。SSH 进去后用 `python3 -m crab.gateway` 调用：

```bash
# 从宿主机
ssh -i ~/crab-vm/work/id_ed25519 -p 2223 root@127.0.0.1

# VM 内（cd / 避免 namespace 包问题）
cd / && python3 -m crab.gateway tenants list
cd / && python3 -m crab.gateway keys create --tenant <tenant_id>
```

注意：`--reset` 会删除整个注册库（所有租户和 Key 一并清空）并重建
默认租户，操作前务必确认。

---

## 管理 HTTP API（本机 Unix socket）

CLI 只是下面这组接口的封装，需要脚本化时可直接调用（仅限管理
socket，不监听 TCP）：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/admin/tenants` | 列出租户 |
| `POST` | `/admin/tenants` | 创建租户，body: `{"name": "...", "quotas": {...}}` |
| `POST` | `/admin/keys` | 签发 Key，body: `{"tenant_id": "..."}` |
| `POST` | `/admin/keys/revoke` | 吊销 Key，body: `{"key": "..."}` |
| `POST` | `/admin/quotas` | 替换配额，body: `{"tenant_id": "...", "quotas": {...}}` |
| `POST` | `/admin/sandboxes/adopt` | 收养沙箱，body: `{"tenant": "...", "sandbox_ids": [...]}` |

---

## 相关文档

- [云端部署指南](deploy-cloud.md)：一键部署脚本及部署后运维。
- [SDK 文档](sdk.md)：客户端如何用 API Key 连接 Gateway。
- [Daemon 与 CLI](daemon.md)：daemon 层的行为与限制。
