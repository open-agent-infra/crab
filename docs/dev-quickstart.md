# Crab 开发环境部署 + 试用指南

> 写给坐在开发 VM 前面的同事看的。全程命令行，5 分钟跑通。

---

## 1. 前置条件 Checklist

跑之前先确认这几项（VM 里都应该已经到位）：

```bash
id -u                  # 必须是 0（root）
zpool list             # 应该看到 crab pool
which runc criu        # 都有输出
python3 --version      # 3.11+
pip show crab          # 有输出，说明已装
```

如果 `pip show crab` 没东西，在项目根目录装一下：

```bash
cd /root/crab  # 或你的项目路径
pip install -e .
```

## 2. 启动 Daemon

```bash
crab daemon start --foreground --config config/crab.yaml
```

- `--foreground`：前台跑，日志直接打终端，方便调试。去掉就后台跑。
- `--config`：指向配置文件。不传则用内置默认值。
- 默认 socket 在 `~/.cache/crab/crab/crab.sock`（或 `$CRAB_DAEMON_SOCKET`）。

确认活着：

```bash
crab daemon status   # 显示 "daemon is reachable"
crab info            # 打印版本/PID/运行时等信息
```

**常见问题：socket 已存在**

报 "Address already in use" → 上次没干净退出。手动删：

```bash
rm -f ~/.cache/crab/crab/crab.sock
```

## 3. 创建沙箱

### CLI 方式

```bash
# 创建并进入（交互式，像 docker run -it）
crab sandbox run ubuntu:22.04 -- bash

# 创建后台沙箱（不执行命令，返回 sandbox_id）
crab sandbox run ubuntu:22.04 --detach
```

### SDK 方式

```python
from crab import Engine, Sandbox

engine = Engine.connect()                       # 连本地 daemon
sbx = Sandbox(image="ubuntu:22.04", engine=engine)
print(sbx.sandbox_id)                           # 拿到 id

result = sbx.commands.run("echo hello world")
print(result.stdout)                            # "hello world"
```

## 4. 操作沙箱

假设你的 sandbox_id 是 `sbx-abc123`：

```bash
# 执行命令
crab sandbox exec sbx-abc123 -- ls /

# 查看所有沙箱
crab sandbox ls

# 创建 checkpoint（进程 + 文件系统快照）
crab checkpoint create sbx-abc123

# 恢复到某个 checkpoint
crab restore sbx-abc123 ckpt-xxxx

# fork（克隆出 2 个独立运行的副本）
crab sandbox fork sbx-abc123 -n 2

# 销毁
crab sandbox rm sbx-abc123
```

SDK 对应写法：

```python
result = sbx.commands.run("cat /etc/os-release")
ckpt_id = sbx.checkpoint()                      # 快照
sbx.restore(ckpt_id)                            # 恢复
forks = sbx.fork(2)                             # 克隆
sbx.kill()                                      # 销毁
```

## 5. 启动 Gateway

Gateway 是给 daemon 加的一层 HTTP 皮——认证、租户隔离、配额。需要 daemon 先跑着。

```bash
crab-gateway serve \
  --bind 0.0.0.0:8080 \
  --daemon-socket ~/.cache/crab/crab/crab.sock \
  --log-level INFO
```

参数说明：
- `--bind`：对外监听地址。默认只 loopback（127.0.0.1），线上请用反代加 TLS。
- `--daemon-socket`：连后端 daemon 的 socket 路径。
- `--data-dir`：gateway 自己的 SQLite 数据库放哪，默认 `~/.local/share/crab/gateway`。

确认启动：

```bash
curl http://127.0.0.1:8080/healthz
# {"ok": true, "started": true}
```

## 6. 创建租户和 API Key

Gateway 管理操作走本地 admin socket（不需要认证）：

```bash
# 创建租户
crab-gateway tenants create acme --max-sandboxes 10

# 记住输出里的 tenant id（tn_xxx）

# 给租户发 key（明文只展示这一次！）
crab-gateway keys create --tenant tn_xxx

# 输出类似：
# api_key: crab_key_AbCdEf...（抄下来）
```

可选：设置资源配额：

```bash
crab-gateway tenants create metered --max-sandboxes 5 --max-memory 4G --max-cpu 8
```

查看租户列表：

```bash
crab-gateway tenants list
```

## 7. 远程模式使用（SDK 通过 Gateway）

拿到 gateway 地址 + API key 后，SDK 走 HTTP：

```python
from crab import Engine, Sandbox
from crab.errors import SandboxExecTimeout

engine = Engine.connect(
    url="http://192.168.1.100:8080",   # gateway 地址
    api_key="crab_key_AbCdEf...",      # 上一步拿到的 key
)

# 创建沙箱。公共 Docker Hub image 缺失时会按 daemon policy 自动拉取；
# network=None/省略使用 daemon 默认（标准配置为独立 netns）；
# False 显式使用 host 网络，True 要求独立 netns。
sbx = Sandbox(image="python:3.12-slim", network=None, engine=engine)
info = sbx.describe()
print(sbx.sandbox_id, info.metadata["image_digest"], info.metadata["network_mode"])

# 执行命令
result = sbx.commands.run("whoami")
print(result.stdout)

# timeout 由 daemon 强制执行，返回前已经回收命令及其 descendants
try:
    sbx.commands.run("sleep 30 & wait", timeout=1)
except SandboxExecTimeout:
    print("timed out and reaped")

# fork
forks = sbx.fork(2)

# 销毁
sbx.kill()
```

同一台机器测也行——把 url 改成 `http://127.0.0.1:8080` 就好。

环境变量方式（免硬编码）：

```bash
export CRAB_API_KEY="crab_key_AbCdEf..."
```

```python
engine = Engine.connect(url="http://127.0.0.1:8080")  # 自动读 $CRAB_API_KEY
```

重连已有沙箱：

```python
sbx = Sandbox.connect("sbx-abc123", engine=engine)
result = sbx.commands.run("echo reconnected")
```

## 8. 资源限制演示（S3）

创建带 cgroup 限制的沙箱：

```python
from crab import Engine, Sandbox

engine = Engine.connect()
sbx = Sandbox(
    image="ubuntu:22.04",
    engine=engine,
    resources={"cpus": 2, "memory": "512M", "pids": 128},
)
```

字段说明：
- `cpus`：CPU 核数（映射为 cgroup cpu.quota/period）
- `memory`：内存上限，支持后缀 K/M/G/T（1024 进位），如 `"512M"` = 512MiB
- `pids`：最大进程数

验证限制生效：

```bash
# 在沙箱里看 cgroup 限制
crab sandbox exec sbx-xxx -- cat /sys/fs/cgroup/memory.max
# 应该输出 536870912（512*1024*1024）
```

fork 会继承父沙箱的资源限制——子沙箱和父沙箱拿同样的 cgroup 配置。

通过 gateway 使用时，如果租户配了 `max_memory_bytes`/`max_cpu` 配额，
所有 create/fork 必须声明对应 resources，否则会收到 409 QuotaExceeded。

## 9. 停止 / 清理

```bash
# 杀掉所有沙箱
crab sandbox ls --json | python3 -c "
import json, sys
for sb in json.loads(sys.stdin.read())['sandboxes']:
    print(sb['sandbox_id'])
" | xargs -I{} crab sandbox rm {}

# 停 gateway（Ctrl+C 或关终端）

# 停 daemon
crab daemon stop

# 清理 ZFS 数据集（谨慎！会删所有沙箱数据）
zfs destroy -r crab/sandboxes
zfs create crab/sandboxes
```

## 10. 已知限制与注意事项

| 项目 | 状态 |
|------|------|
| 远程创建沙箱 | ✅ 正常工作（gateway 转发 create 到 daemon） |
| exec 流式输出 | ✅ `commands.stream()` 通过 Gateway 流式转发 stdout/stderr |
| 端口暴露 | ✅ isolated netns 下支持；service VM/NAT 仍需转发分配出的 host port |
| `commands.run(timeout=...)` | ✅ daemon 强制硬超时并回收完整 payload tree |
| `Sandbox(timeout=...)` | ✅ Gateway idle-reclaim 窗口；配合 `idle_action` 使用 |
| `labels` 参数 | ⚠️ 仅为 advisory 元数据，不影响运行时行为 |
| daemon 重启 | ⚠️ 会丢失所有运行中的沙箱，gateway 对它们返回 410 Gone |
| PTY / 交互终端 | ❌ 不支持，远程只有 exec 一次性调用 |

## 11. 故障排查

| 症状 | 一句话解法 |
|------|-----------|
| `crab daemon status` 说连不上 | daemon 没起或 socket 路径不对。确认进程在跑 + socket 文件存在 |
| "Address already in use" | `rm -f ~/.cache/crab/crab/crab.sock` 然后重新启动 |
| gateway 报 "daemon unreachable" | `--daemon-socket` 路径写错了，或者 daemon 没在跑 |
| 认证失败 401 | API key 拼错或已被 revoke。`crab-gateway keys create --tenant tn_xxx` 重新发一个 |
| 409 QuotaExceeded | 配额满了。`crab sandbox rm` 杀掉旧沙箱释放配额，或 `crab-gateway quotas set` 调高 |
| "zpool not found" / ZFS 报错 | `zpool list` 确认 pool 存在。不存在就重建：`zpool create crab /path/to/file.img` |
| checkpoint 失败 | 常见原因：沙箱里有 TCP 连接但配置没开 `tcp_established: true`。默认配置已开，自定义配置注意 |
| fork 后子沙箱起不来 | CRIU restore 失败。看 daemon 日志（`/var/lib/crab/logs/engine.log`）里的具体错误 |
| pip install 报 PEP 668 | 加 `--break-system-packages`，或在 venv 里装 |

---

*文档对应代码版本：main @ afc911b（S1–S3 shipped）*
