"""Crab SDK 远程接口全功能教程

本脚本演示如何从一台 *远程* 客户端机器连接 Crab Gateway（或 daemon），
并覆盖 SDK 暴露的所有远程操作，包括公共镜像按需拉取、运行时 baseline、
daemon 强制执行超时、网络 tri-state、富返回值 `ActionResult`、异步
checkpoint / changeset、inspector 只读 peek，以及 Sandbox 级
`auto_checkpoint` 模式。教程只清理本次运行创建的 sandbox，不会删除同一
tenant 中已有的 workload。

使用前准备:
  1. 确保目标机器已部署 crab-gateway（或 daemon）并在监听 HTTP 端口。
  2. 安装 SDK: pip install crab  （或在项目根目录 pip install -e .）
  3. 通过环境变量提供 Gateway 地址和 API key（不要把凭证写死进源码）：
         export CRAB_GATEWAY_URL=http://your-gateway-host:8900
         export CRAB_API_KEY=crab_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

运行:
    export CRAB_GATEWAY_URL=http://host:8900 CRAB_API_KEY=crab_sk_xxx
    python examples/sdk/04_remote_service_tutorial.py

可选配置:
    export CRAB_TUTORIAL_IMAGE=python:3.12-slim  # 未缓存时自动 pull
    export CRAB_TUTORIAL_NETWORK=auto            # auto | host | isolated
    export CRAB_TUTORIAL_RUN_APT=1               # 验证普通 apt 工作流
"""
from __future__ import annotations

import os
import sys
import time
import traceback
import uuid
from urllib.parse import urlparse

from crab import Engine, Sandbox
from crab.errors import SandboxExecTimeout
from crab.models import ExecEvent, ExecDone

FAILURES: list[str] = []

# ============================================================
# ★ 用户配置区 ★  —— 通过环境变量提供凭证（勿硬编码真实 key/URL）
# ============================================================
# export CRAB_GATEWAY_URL=http://your-gateway-host:8900
# export CRAB_API_KEY=crab_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GATEWAY_URL = os.environ.get("CRAB_GATEWAY_URL", "http://YOUR_GATEWAY_HOST:8900")
API_KEY = os.environ.get("CRAB_API_KEY", "YOUR_API_KEY")


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (1/0, true/false, yes/no, on/off)")


def network_setting() -> bool | None:
    raw = os.environ.get("CRAB_TUTORIAL_NETWORK", "auto").strip().lower()
    if raw in {"auto", "default", "null", "none"}:
        return None
    if raw in {"host", "false", "off", "0"}:
        return False
    if raw in {"isolated", "true", "on", "1"}:
        return True
    raise ValueError("CRAB_TUTORIAL_NETWORK must be auto, host, or isolated")


# 非敏感配置。默认资源保持较小，避免教程挤占同 tenant 的真实 workload。
SANDBOX_IMAGE = os.environ.get("CRAB_TUTORIAL_IMAGE", "ubuntu:22.04")
SANDBOX_NETWORK = network_setting()
SANDBOX_RESOURCES = {
    "memory": os.environ.get("CRAB_TUTORIAL_MEMORY", "512M"),
    "cpus": int(os.environ.get("CRAB_TUTORIAL_CPUS", "1")),
}
AUX_SANDBOX_MEMORY = os.environ.get("CRAB_TUTORIAL_AUX_MEMORY", "256M")
RUN_APT_CHECK = env_flag("CRAB_TUTORIAL_RUN_APT", True)
TUTORIAL_RUN_ID = os.environ.get("CRAB_TUTORIAL_RUN_ID", uuid.uuid4().hex[:8])


def tutorial_name(role: str) -> str:
    return f"crab-tutorial-{TUTORIAL_RUN_ID}-{role}"


# ============================================================
# 工具函数
# ============================================================

def banner(title: str) -> None:
    """打印分隔线和步骤标题"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def step_ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def step_fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  ✗ {msg}")


def step_warn(msg: str) -> None:
    print(f"  ! {msg}")


def safe_run(description: str, func, *args, **kwargs):
    """安全地执行一步操作。如果失败则打印错误并继续。"""
    try:
        result = func(*args, **kwargs)
        return result
    except Exception as exc:
        step_fail(f"{description} 失败: {exc}")
        traceback.print_exc()
        return None


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    # 凭证检查：未设置环境变量时友好提示并退出
    if API_KEY == "YOUR_API_KEY" or GATEWAY_URL == "http://YOUR_GATEWAY_HOST:8900":
        print("请先设置环境变量后再运行:")
        print("  export CRAB_GATEWAY_URL=http://your-gateway-host:8900")
        print("  export CRAB_API_KEY=crab_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        sys.exit(1)

    # 用于最终统一清理的 sandbox 列表。只向这里加入本次运行创建的对象。
    sandboxes_to_kill: list[Sandbox] = []
    created_ids: set[str] = set()

    # ----------------------------------------------------------
    # 1. 连接 Gateway
    # ----------------------------------------------------------
    banner("1. 连接 Gateway")
    print(f"  URL: {GATEWAY_URL}")

    t0 = time.time()
    engine = Engine.connect(url=GATEWAY_URL, api_key=API_KEY)
    step_ok(f"Engine 连接成功 ⏱ {time.time()-t0:.3f}s")
    print(f"  Engine 类型: {type(engine).__name__}")

    # ----------------------------------------------------------
    # 1.5 记录已有沙箱（绝不清理其他 workload）
    # ----------------------------------------------------------
    # 旧版教程会 kill 当前 API key 可见的全部 sandbox，这会误删同 tenant
    # 的真实任务。现在只记录基线；finally 只回收 created_ids 中的对象。
    banner("1.5 记录已有沙箱（保留现有 workload）")
    t0 = time.time()
    existing = safe_run("列出已有沙箱", engine.list_sandboxes) or []
    preexisting_ids = {
        str(row["sandbox_id"])
        for row in existing
        if row.get("sandbox_id")
    }
    step_ok(
        f"发现并保留 {len(preexisting_ids)} 个已有沙箱 "
        f"⏱ {time.time()-t0:.3f}s"
    )

    try:
        # ----------------------------------------------------------
        # 2. 远程创建沙箱
        # ----------------------------------------------------------
        banner("2. 远程创建沙箱")
        print(f"  镜像: {SANDBOX_IMAGE}")
        print(f"  资源: {SANDBOX_RESOURCES}")
        print(f"  网络请求: {SANDBOX_NETWORK!r} (None 表示 daemon 默认)")

        t0 = time.time()
        sandbox = Sandbox(
            image=SANDBOX_IMAGE,
            resources=SANDBOX_RESOURCES,
            network=SANDBOX_NETWORK,
            name=tutorial_name("main"),
            engine=engine,
        )
        sandboxes_to_kill.append(sandbox)
        created_ids.add(str(sandbox.sandbox_id))
        step_ok(f"沙箱创建成功, id = {sandbox.sandbox_id} ⏱ {time.time()-t0:.3f}s")

        info = sandbox.describe()
        metadata = info.metadata
        print(f"  image_reference: {metadata.get('image_reference')}")
        print(f"  image_digest: {metadata.get('image_digest')}")
        print(f"  rootfs_preparation_schema: {metadata.get('rootfs_preparation_schema')}")
        print(f"  effective network_mode: {metadata.get('network_mode')}")
        required_metadata = (
            "image_reference",
            "image_id",
            "image_digest",
            "rootfs_preparation_schema",
            "network_mode",
        )
        missing = [key for key in required_metadata if not metadata.get(key)]
        if missing:
            step_fail(f"创建结果缺少 baseline metadata: {', '.join(missing)}")
            raise RuntimeError(
                "Gateway/daemon 尚未运行支持 runtime baseline 的版本；"
                "请先完成服务升级并重启后再运行本教程"
            )
        else:
            step_ok("创建结果包含不可变镜像身份、baseline schema 和有效网络模式")

        # ----------------------------------------------------------
        # 2a. 新 sandbox runtime baseline
        # ----------------------------------------------------------
        banner("2a. Runtime baseline — DNS + 非特权 capability")
        baseline = safe_run(
            "验证 runtime baseline",
            sandbox.commands.run,
            """set -eu
test -s /etc/resolv.conf
dns_answer=$(getent hosts archive.ubuntu.com | head -n 1)
cap_hex=$(awk '/^CapEff:/{print $2}' /proc/self/status)
cap_dec=$(printf '%d' "0x${cap_hex}")
test $((cap_dec & (1 << 6))) -ne 0
test $((cap_dec & (1 << 7))) -ne 0
test $((cap_dec & (1 << 21))) -eq 0
printf 'dns=%s\nCapEff=%s\n' "$dns_answer" "$cap_hex"
""",
            timeout=60.0,
            check=True,
        )
        if baseline is not None:
            step_ok("DNS 可用；CAP_SETGID/CAP_SETUID 存在；CAP_SYS_ADMIN 不存在")
            print(f"  {baseline.stdout.strip()}")

        if RUN_APT_CHECK:
            apt = safe_run(
                "验证普通 APT 权限切换",
                sandbox.commands.run,
                "DEBIAN_FRONTEND=noninteractive apt-get update -qq && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl && "
                "curl --version | head -n 1",
                timeout=240.0,
                check=True,
            )
            if apt is not None:
                step_ok("apt-get 无需 APT::Sandbox::User=root 即可安装 curl")
                print(f"  {apt.stdout.strip()}")
        else:
            step_warn("已通过 CRAB_TUTORIAL_RUN_APT=0 跳过 APT 验证")

        # ----------------------------------------------------------
        # 2b. daemon 强制硬超时
        # ----------------------------------------------------------
        banner("2b. Exec hard timeout — 完整回收 payload")
        timed_out_cleanly = False
        try:
            sandbox.commands.run(
                "echo $$ > /tmp/crab-tutorial-timeout-shell.pid; "
                "sleep 30 & echo $! > /tmp/crab-tutorial-timeout-child.pid; wait",
                timeout=1.0,
            )
            step_fail("超时命令意外正常返回")
        except SandboxExecTimeout as exc:
            timed_out_cleanly = True
            step_ok(
                "收到稳定的 SandboxExecTimeout；daemon 已完成 payload 清理后才返回 "
                f"(timeout={exc.timeout}s)"
            )
        except Exception as exc:
            step_fail(f"超时返回了非预期错误类型 {type(exc).__name__}: {exc}")

        if timed_out_cleanly:
            reaped = safe_run(
                "确认超时 shell 与子进程均已消失",
                sandbox.commands.run,
                """set -eu
for pid_file in /tmp/crab-tutorial-timeout-shell.pid /tmp/crab-tutorial-timeout-child.pid; do
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        echo "payload still alive: pid=$pid" >&2
        exit 1
    fi
done
echo reaped
""",
                timeout=10.0,
                check=True,
            )
            if reaped is not None:
                step_ok("超时 payload 及其 descendants 均已回收")

        # ----------------------------------------------------------
        # 3. 列出当前 tenant 的沙箱 (engine.list_sandboxes)
        # ----------------------------------------------------------
        # 演示新增的 list_sandboxes()：走 GET /sandboxes 路由，返回
        # 当前 API key 可见的所有沙箱条目（gateway 会按 tenant 过滤）。
        banner("3. 列出沙箱 (engine.list_sandboxes)")
        t0 = time.time()
        sandboxes = safe_run("列出沙箱", engine.list_sandboxes) or []
        step_ok(f"当前沙箱列表: {len(sandboxes)} 个 ⏱ {time.time()-t0:.3f}s")
        for s in sandboxes:
            print(f"    {s.get('sandbox_id')} - {s.get('status', '?')}")

        # ----------------------------------------------------------
        # 4. 执行命令（一次性）
        # ----------------------------------------------------------
        banner("4. 执行命令 (commands.run)")
        t0 = time.time()
        result = sandbox.commands.run("echo hello && uname -a && whoami")
        step_ok(f"返回码: {result.returncode} ⏱ {time.time()-t0:.3f}s")
        print(f"  stdout:\n{result.stdout.rstrip()}")

        # ----------------------------------------------------------
        # 5. 富返回值 — checkpoint + observe
        # ----------------------------------------------------------
        # 演示 commands.run(checkpoint=True, observe=True) 的富返回值：
        #   * checkpoint_id 在客户端侧预分配（ckpt-<uuid>），daemon 的
        #     /action 端点做完 exec + observe 后 *立即返回*，checkpoint
        #     在 daemon 后台线程异步执行。
        #   * 这个 id 是 logical checkpoint id，wait 前后不会改变。daemon
        #     根据 inspector 选择复用旧物理状态、只做 filesystem checkpoint，
        #     或做 process + filesystem checkpoint。
        #   * observe=True 会调用 daemon 的只读 inspector peek 路由，
        #     不重置任何游标。
        #
        # 异步行为（重要）：run() 的返回延迟 *不包含* checkpoint 开销。
        # 返回时 result.checkpoint.done == False（后台正在做）；调用
        # result.checkpoint.wait() 会轮询 daemon 的 jobs 端点直到完成。
        #
        # 语义（重要）：observe 返回的 filesystem_changed / process_changed
        # 表示 “本次 action 是否改变了状态” —— daemon 在 *启动后台
        # checkpoint 之前* 就完成 peek，因此读到的是 checkpoint 重置游标
        # 之前的干净状态。所以下面 mutating 的命令应该稳定看到
        # filesystem_changed=True。
        banner("5. 富返回值 — checkpoint + observe")
        t0 = time.time()
        result = sandbox.commands.run(
            "echo 'hello' > /tmp/observed.txt && mkdir -p /opt/new",
            checkpoint=True,
            observe=True,
        )
        exec_latency = time.time() - t0
        step_ok(f"返回码: {result.returncode} (exec+observe 返回) ⏱ {exec_latency:.3f}s")
        print(f"  checkpoint_id (预分配): {result.checkpoint.checkpoint_id}")
        print(f"  checkpoint 完成?: {result.checkpoint.done}")  # 期望 False（后台异步）
        print(f"  filesystem_changed: {result.filesystem_changed}")
        print(f"  process_changed: {result.process_changed}")
        # 阻塞等待后台 checkpoint 完成（轮询 daemon jobs 端点）
        try:
            t1 = time.time()
            ckpt_id = result.checkpoint.wait(timeout=60.0)
            wait_latency = time.time() - t1
            step_ok(f"checkpoint 已完成: {ckpt_id} (wait 轮询) ⏱ {wait_latency:.3f}s")
            print(f"  checkpoint 完成?: {result.checkpoint.done}")  # 期望 True
            print(f"  → exec 返回 {exec_latency:.3f}s + checkpoint 等待 {wait_latency:.3f}s")
            checkpoint_info = next(
                (
                    item
                    for item in sandbox.checkpoints.list()
                    if item.get("checkpoint_id") == ckpt_id
                ),
                None,
            )
            if checkpoint_info is not None:
                print(
                    "  materialization: "
                    f"{checkpoint_info.get('materialization')} "
                    f"(process={checkpoint_info.get('process_checkpoint_id')}, "
                    f"filesystem={checkpoint_info.get('filesystem_checkpoint_id')})"
                )
        except Exception as exc:
            step_fail(f"等待 checkpoint 失败: {exc}")

        # ----------------------------------------------------------
        # 5b. 背压 — 上一个后台 checkpoint 阻塞下一个 run
        # ----------------------------------------------------------
        # daemon 侧 per-sandbox 背压：新的 /action 到来时，如果该 sandbox
        # 还有后台 checkpoint 在跑，daemon 会先等它完成再开始 exec。用户
        # 感知为 "上一个 checkpoint 慢了，这次请求就稍等一等"。
        #
        # 如何 *隔离* 出背压带来的额外延迟（否则会被 exec 本身的耗时淹没）：
        #   run#A: 触发一个后台 checkpoint（不等待）——制造一个 pending ckpt
        #   run#B: 紧跟 run#A 发起 -> 其 exec 返回时间 *包含* 等待 run#A
        #          checkpoint 的背压时间
        #   run#C: 等到完全空闲后再发起相同的命令 -> 无背压，作为基线
        #   背压额外延迟 ≈ run#B 延迟 - run#C 延迟
        #
        # 重要事实：logical checkpoint 会由 inspector 自适应决定物理工作；
        # 本段 run#A 写了文件，所以至少会生成 filesystem checkpoint。ZFS
        # 快照是 O(1) 写时复制，与文件数量基本无关。无变化轮次则可完全
        # 复用之前的物理状态。所以真实环境下背压额外延迟通常较小；逻辑由
        # 单元测试 tests/test_daemon_action_backpressure.py 注入 1s 延迟
        # 做了严格验证。
        banner("5b. 背压 — 上一个后台 checkpoint 阻塞下一个 run")
        # run#A：制造一个 pending 后台 checkpoint（创建一批文件后 checkpoint）
        sandbox.commands.run(
            "mkdir -p /opt/many && cd /opt/many && "
            "for i in $(seq 1 10240); do echo x > f$i; done",
            checkpoint=True,
        )
        step_ok("run#A 已触发后台 checkpoint（不等待，制造 pending 状态）")
        # run#B：紧跟发起 -> 被 run#A 的后台 checkpoint 背压阻塞
        t1 = time.time()
        result_blocked = sandbox.commands.run("echo blocked", checkpoint=True)
        blocked_latency = time.time() - t1
        step_ok(f"run#B (紧跟 run#A) exec 返回 ⏱ {blocked_latency:.3f}s (含背压等待)")
        # 等到完全空闲
        try:
            result_blocked.checkpoint.wait(timeout=120.0)
        except Exception as exc:
            step_fail(f"等待 run#B checkpoint 失败: {exc}")
        # run#C：空闲基线 -> 无 pending checkpoint，无背压
        t2 = time.time()
        result_idle = sandbox.commands.run("echo idle", checkpoint=True)
        idle_latency = time.time() - t2
        step_ok(f"run#C (空闲基线) exec 返回 ⏱ {idle_latency:.3f}s (无背压)")
        try:
            result_idle.checkpoint.wait(timeout=120.0)
        except Exception as exc:
            step_fail(f"等待 run#C checkpoint 失败: {exc}")
        delta_ms = (blocked_latency - idle_latency) * 1000.0
        print(
            f"  → 背压额外延迟 ≈ {delta_ms:.0f}ms "
            f"(run#B {blocked_latency:.3f}s - run#C {idle_latency:.3f}s)"
        )
        if blocked_latency > idle_latency:
            step_ok("背压生效：紧跟的 run#B 比空闲的 run#C 慢（等待了上一个 checkpoint）")
        else:
            print("  (注: ZFS 快照 O(1)，checkpoint ~0.1s，背压额外延迟很小/接近噪声)")

        # ----------------------------------------------------------
        # 6. 富返回值 — changeset (异步 + 同步)
        # ----------------------------------------------------------
        # 演示 changeset 的两种模式：
        #   * changeset=True     -> AsyncChangeset，后台线程计算，.wait() 阻塞取
        #   * changeset_sync=True -> 立即返回 list，同步计算
        # 两种模式都隐式使用 force=True 绕过 inspector gate。SDK 会
        # 自动用 *上一个* checkpoint（此处是 step 5 的 checkpoint）作
        # 为 since，只 diff 本次 run 引入的变更；这也避免了对普通（非
        # fork）沙箱走 fork_changeset 路径而 400。
        banner("6. 富返回值 — changeset (异步 + 同步)")

        # 6a. 异步 changeset
        t0 = time.time()
        result = sandbox.commands.run("touch /tmp/async_test", changeset=True)
        try:
            entries = result.changeset.wait(timeout=60.0)
            step_ok(f"异步 changeset: {len(entries)} 条变更 ⏱ {time.time()-t0:.3f}s")
        except Exception as exc:
            step_fail(f"等待异步 changeset 失败: {exc}")

        # 6b. 同步 changeset
        t0 = time.time()
        result = sandbox.commands.run(
            "rm /tmp/async_test", changeset=True, changeset_sync=True
        )
        if isinstance(result.changeset, list):
            step_ok(f"同步 changeset: {len(result.changeset)} 条变更 ⏱ {time.time()-t0:.3f}s")
        else:
            step_fail(f"同步 changeset 返回类型异常: {type(result.changeset).__name__}")

        # ----------------------------------------------------------
        # 7. auto_checkpoint 模式
        # ----------------------------------------------------------
        # Sandbox 级 auto_checkpoint=True：每次 commands.run 后自动
        # 触发后台 logical checkpoint，无需每次都传 checkpoint=True。
        # 预分配的 logical id 可以通过 sandbox.last_checkpoint_id 直接读出；
        # 是否产生新物理 checkpoint、是 partial 还是 full 由 inspector 决定。
        banner("7. auto_checkpoint 模式")
        auto_sb = safe_run(
            "创建 auto_checkpoint 沙箱",
            Sandbox,
            image=SANDBOX_IMAGE,
            resources={"memory": AUX_SANDBOX_MEMORY},
            network=SANDBOX_NETWORK,
            name=tutorial_name("auto"),
            engine=engine,
            auto_checkpoint=True,
        )
        if auto_sb is not None:
            sandboxes_to_kill.append(auto_sb)
            created_ids.add(str(auto_sb.sandbox_id))
            step_ok(f"auto_checkpoint 沙箱创建成功: {auto_sb.sandbox_id}")
            t0 = time.time()
            auto_sb.commands.run("echo step1 > /tmp/s1")
            # 后台进程演示：detach=True 让 SDK 把命令 stdio 重定向到
            # /dev/null（等价于前置 `exec 1>/dev/null 2>&1;`），从而解除
            # `runc exec` 的管道继承阻塞——否则光加 `&` 会一直阻塞到进程退出。
            # detach 只解除“管道继承阻塞”，不改变进程生命周期：命令自身
            # 仍须用 `&` 后台化，run 才会立即返回；detach 模式不返回输出。
            # 这个 sleep 30 后台进程会被本次 auto_checkpoint 捕获。
            # observe=True 在 exec 之后、checkpoint 之前做一次只读 inspector peek。
            # process_changed 由 host-inspector 对比“当前 cgroup 活进程集”与“上次
            # checkpoint 基线”得出（实时读 cgroup PID，不是靠消费 eBPF 事件），所以
            # 只要后台进程还活着 process_changed 就稳定为 True；本命令没写磁盘，
            # 因此 filesystem_changed=False（VM 实测：proc=True / fs=False）。
            tmp1 = auto_sb.commands.run(
                "sleep 30 & sleep 2", detach=True, observe=True
            )
            step_ok(f"自动 checkpoint: {auto_sb.last_checkpoint_id} ⏱ {time.time()-t0:.3f}s")
            print(f"  filesystem_changed: {tmp1.filesystem_changed}")
            print(f"  process_changed: {tmp1.process_changed}")
            t0 = time.time()
            tmp2 = auto_sb.commands.run("echo step2 > /tmp/s2", observe=True)
            step_ok(f"第二次自动 checkpoint（含背压）: {auto_sb.last_checkpoint_id} ⏱ {time.time()-t0:.3f}s")
            print(f"  filesystem_changed: {tmp2.filesystem_changed}")
            print(f"  process_changed: {tmp2.process_changed}")
            try:
                tmp2.checkpoint.wait(timeout=120.0)
            except Exception as exc:
                step_fail(f"等待 step 2 checkpoint 失败: {exc}")
            time.sleep(5)
            t0 = time.time()
            tmp3 = auto_sb.commands.run("echo step3 > /tmp/s3", observe=True)
            step_ok(f"第三次自动 checkpoint（无背压）: {auto_sb.last_checkpoint_id} ⏱ {time.time()-t0:.3f}s")
            print(f"  filesystem_changed: {tmp3.filesystem_changed}")
            print(f"  process_changed: {tmp3.process_changed}")
        else:
            step_fail("auto_checkpoint 沙箱不可用，跳过后续演示")

        # ----------------------------------------------------------
        # 8. 流式执行
        # ----------------------------------------------------------
        banner("8. 流式执行 (commands.stream)")
        print("  命令: for i in 1 2 3; do echo step-$i; sleep 0.5; done")
        print("  输出:")

        t0 = time.time()
        for event in sandbox.commands.stream(
            "for i in 1 2 3; do echo step-$i; sleep 0.5; done"
        ):
            if isinstance(event, ExecEvent):
                print(f"    [{event.channel}] {event.text}", end="")
            elif isinstance(event, ExecDone):
                print(f"    [exit] returncode={event.returncode}")

        step_ok(f"流式执行完成 ⏱ {time.time()-t0:.3f}s")

        # ----------------------------------------------------------
        # 9. 查看文件变更 (changeset with force)
        # ----------------------------------------------------------
        # 使用 sandbox.changeset(force=True) 跳过 daemon 的 inspector gate
        # 优化。不启用 eBPF host inspector 时，gate 会保守地判定"无变更"
        # 而返回空列表；force=True 强制走 diff 逻辑，拿到真实的变更集。
        banner("9. 文件变更 (changeset, force=True)")
        print("  先写一个测试文件…")

        # 先做一次 checkpoint 作为 changeset 的基线
        t0 = time.time()
        base_ckpt = safe_run(
            "创建基线 checkpoint",
            sandbox.checkpoint,
            "changeset-base",
        )
        if base_ckpt:
            step_ok(f"基线 checkpoint: {base_ckpt} ⏱ {time.time()-t0:.3f}s")
            # 写入测试文件
            sandbox.commands.run("echo 'hello changeset' > /tmp/changeset_test.txt")
            sandbox.commands.run("mkdir -p /opt/demo && echo 123 > /opt/demo/data.txt")
            step_ok("写入了 /tmp/changeset_test.txt 和 /opt/demo/data.txt")

            # force=True: 无需等 inspector 观察到变更即可拿到结果
            t0 = time.time()
            changes = safe_run(
                "查询 changeset(force=True)",
                sandbox.changeset,
                base_ckpt,
                force=True,
            )
            if changes is not None:
                step_ok(f"changeset 返回 {len(changes)} 条变更 (force 跳过 inspector gate) ⏱ {time.time()-t0:.3f}s")
                for entry in changes[:10]:  # 最多显示 10 条
                    print(f"    {entry}")
        else:
            step_fail("跳过 changeset 测试 (checkpoint 不可用)")

        # ----------------------------------------------------------
        # 10. Checkpoint
        # ----------------------------------------------------------
        banner("10. Checkpoint")
        t0 = time.time()
        ckpt_id = safe_run("创建 checkpoint", sandbox.checkpoint, "tutorial-ckpt")
        if ckpt_id:
            step_ok(f"Checkpoint 创建成功: {ckpt_id} ⏱ {time.time()-t0:.3f}s")
        else:
            step_fail("Checkpoint 不可用 (可能缺少 CRIU 或权限不足)")

        # ----------------------------------------------------------
        # 11. Fork
        # ----------------------------------------------------------
        banner("11. Fork")
        t0 = time.time()
        forks = safe_run("Fork 沙箱", sandbox.fork, 1)
        if forks:
            fork_sbx = forks[0]
            sandboxes_to_kill.append(fork_sbx)
            created_ids.add(str(fork_sbx.sandbox_id))
            step_ok(f"Fork 成功, fork id = {fork_sbx.sandbox_id} ⏱ {time.time()-t0:.3f}s")

            fork_metadata = fork_sbx.describe().metadata
            source_mode = sandbox.describe().metadata.get("network_mode")
            fork_mode = fork_metadata.get("network_mode")
            if fork_mode == source_mode:
                step_ok(f"Fork 继承有效网络模式: {fork_mode}")
            else:
                step_fail(f"Fork 网络模式不一致: source={source_mode}, fork={fork_mode}")
            if source_mode == "isolated":
                source_netns = sandbox.describe().metadata.get("network_namespace_path")
                fork_netns = fork_metadata.get("network_namespace_path")
                if source_netns and fork_netns and source_netns != fork_netns:
                    step_ok("隔离网络 fork 获得独立 network namespace")
                else:
                    step_fail("隔离网络 fork 未报告独立 network namespace")

            # 在 fork 里做一些修改
            fork_result = fork_sbx.commands.run(
                "echo 'I am a fork' > /tmp/fork_marker.txt && cat /tmp/fork_marker.txt"
            )
            step_ok(f"Fork 内执行结果: {fork_result.stdout.rstrip()}")

            # 验证原沙箱没有这个文件
            orig_check = sandbox.commands.run(
                "cat /tmp/fork_marker.txt 2>&1 || true"
            )
            step_ok(f"原沙箱检查 (应找不到文件): {orig_check.stdout.rstrip()}")
        else:
            step_fail("Fork 不可用 (可能缺少 CRIU 或 ZFS)")

        # ----------------------------------------------------------
        # 11b. 从历史 checkpoint fork
        # ----------------------------------------------------------
        # 默认的 fork（上面 step 11）从沙箱的 *当前活状态* 分叉。传入
        # checkpoint_id 则从一个 *已存在的 checkpoint* 分叉：不再新打
        # checkpoint，父沙箱也不会被 restore —— 父沙箱继续在原处运行，
        # 只有 fork 回到那个历史点。用 step 10 的 checkpoint 作时间锚点：
        #   1) 先在父沙箱写一个 "checkpoint 之后才存在" 的标记文件
        #   2) 从 step 10 的 checkpoint fork —— 该文件不应出现在 fork 里
        #   3) 父沙箱仍然持有该文件（证明父沙箱没有被回滚）
        banner("11b. 从历史 checkpoint fork")
        if ckpt_id:
            sandbox.commands.run("echo 'after ckpt' > /tmp/after_ckpt.txt")
            t0 = time.time()
            past_forks = safe_run(
                "从历史 checkpoint fork", sandbox.fork, 1, checkpoint_id=ckpt_id
            )
            if past_forks:
                past_fork = past_forks[0]
                created_ids.add(str(past_fork.sandbox_id))
                step_ok(
                    f"Fork 成功, fork id = {past_fork.sandbox_id} "
                    f"(fork 点 = {ckpt_id}) ⏱ {time.time()-t0:.3f}s"
                )
                in_fork = past_fork.commands.run("cat /tmp/after_ckpt.txt 2>&1 || true")
                step_ok(f"fork 内检查 (应找不到该文件): {in_fork.stdout.rstrip()}")
                in_parent = sandbox.commands.run("cat /tmp/after_ckpt.txt 2>&1 || true")
                step_ok(f"父沙箱检查 (文件应仍在): {in_parent.stdout.rstrip()}")
                # 立刻回收：每个沙箱都占 tenant 配额，后面的步骤还要用。
                try:
                    past_fork.kill()
                    step_ok(f"已回收历史 fork: {past_fork.sandbox_id}")
                except Exception as exc:
                    step_fail(f"回收历史 fork 失败: {exc}")
                    sandboxes_to_kill.append(past_fork)
            else:
                step_fail("从历史 checkpoint fork 不可用 (可能缺少 CRIU 或 ZFS)")
        else:
            step_fail("跳过：step 10 未能创建 checkpoint，没有可用的 fork 点")

        # ----------------------------------------------------------
        # 12. Transaction
        # ----------------------------------------------------------
        banner("12. Transaction (begin / exec / commit|abort)")
        t0 = time.time()
        txn = safe_run("开启事务", sandbox.begin, "tutorial-txn")
        if txn:
            step_ok(f"事务已开启: {txn} ⏱ {time.time()-t0:.3f}s")
            # 在事务内执行操作
            try:
                t0 = time.time()
                txn_result = txn.exec("echo 'inside txn' > /tmp/txn_file.txt")
                step_ok(f"事务内 exec 返回码: {txn_result.returncode} ⏱ {time.time()-t0:.3f}s")

                # 提交事务
                t0 = time.time()
                commit_result = txn.commit()
                step_ok(f"事务提交成功: {commit_result} ⏱ {time.time()-t0:.3f}s")
            except Exception as exc:
                step_fail(f"事务操作失败, 尝试 abort: {exc}")
                try:
                    txn.abort()
                    step_ok("事务已回滚")
                except Exception:
                    pass

            # 验证事务结果
            verify = sandbox.commands.run("cat /tmp/txn_file.txt 2>&1 || true")
            step_ok(f"事务提交后验证: {verify.stdout.rstrip()}")
        else:
            step_fail("Transaction 不可用 (daemon 可能不支持)")

        # ----------------------------------------------------------
        # 13. 连接已有沙箱 (Sandbox.connect)
        # ----------------------------------------------------------
        banner("13. 连接已有沙箱 (Sandbox.connect)")
        existing_id = sandbox.sandbox_id
        print(f"  重新连接到: {existing_id}")

        t0 = time.time()
        reconnected = Sandbox.connect(existing_id, engine=engine)
        step_ok(f"重新连接成功, id = {reconnected.sandbox_id} ⏱ {time.time()-t0:.3f}s")

        # 验证可以执行命令
        re_result = reconnected.commands.run("echo 'reconnected!'")
        step_ok(f"通过重连沙箱执行命令: {re_result.stdout.rstrip()}")

        # ----------------------------------------------------------
        # 14. 端口暴露 (ports.expose)
        # ----------------------------------------------------------
        banner("14. 端口暴露 (ports.expose)")
        print("  在沙箱内启动一个简单 HTTP server，然后暴露端口…")

        if sandbox.describe().metadata.get("network_mode") != "isolated":
            step_warn("当前为 host 网络模式；端口暴露仅适用于 isolated netns，跳过")
        else:
            try:
                # 完整脱离 stdio；否则 runc exec 会一直等待继承了管道的
                # 后台进程，并在 timeout 后按契约回收它。
                sandbox.commands.run(
                    "nohup python3 -m http.server 8080 --directory /tmp "
                    ">/tmp/crab-tutorial-http.log 2>&1 </dev/null &",
                    timeout=3.0,
                    check=True,
                )
                # 等待 server 启动
                time.sleep(1)

                # 暴露端口
                t0 = time.time()
                allocation = sandbox.ports.expose(8080)
                step_ok(f"端口暴露成功 ⏱ {time.time()-t0:.3f}s:")
                print(f"    guest_port: {allocation.guest_port}")
                print(f"    host_port:  {allocation.host_port}")
                print(f"    url:        {allocation.url}")

                # 尝试从外部验证（使用 urllib，不依赖 requests）
                # ports.expose 返回的 url 通常是 tcp:// 格式（L4 转发），
                # 需要用 http:// 协议访问沙箱内的 HTTP server
                print("  尝试从外部访问暴露的端口…")
                try:
                    import urllib.request

                    # 从 GATEWAY_URL 提取主机名，拼接 host_port 构造 HTTP URL
                    gw_parsed = urlparse(GATEWAY_URL)
                    gw_host = gw_parsed.hostname or "127.0.0.1"
                    http_url = f"http://{gw_host}:{allocation.host_port}/"
                    print(f"    构造 HTTP URL: {http_url}")
                    with urllib.request.urlopen(http_url, timeout=5) as resp:
                        body = resp.read(200).decode("utf-8", errors="replace")
                        step_ok(
                            f"外部访问成功 (HTTP {resp.status}), 响应前 200 字节:"
                        )
                        print(f"    {body[:200]}")
                except Exception as fetch_exc:
                    step_warn(f"外部访问失败 (可能网络不通): {fetch_exc}")
                    print(
                        "    提示: 端口已暴露，但从当前网络可能无法直接访问 "
                        "gateway 主机。"
                    )
            except Exception as port_exc:
                step_fail(f"端口暴露不可用: {port_exc}")
                print("    提示: 端口暴露需要沙箱具有网络命名空间 (netns) 支持。")

    finally:
        # ----------------------------------------------------------
        # 15. Kill — 清理所有创建的沙箱
        # ----------------------------------------------------------
        banner("15. 清理 (kill)")
        for sbx in sandboxes_to_kill:
            try:
                sid = sbx.sandbox_id
                t0 = time.time()
                sbx.kill()
                step_ok(f"已清理沙箱: {sid} ⏱ {time.time()-t0:.3f}s")
            except Exception as kill_exc:
                step_fail(f"清理沙箱失败: {kill_exc}")

        remaining = safe_run("清理后列出沙箱", engine.list_sandboxes)
        if remaining is not None:
            remaining_ids = {
                str(row["sandbox_id"])
                for row in remaining
                if row.get("sandbox_id")
            }
            missing_preexisting = sorted(preexisting_ids - remaining_ids)
            leaked_tutorial = sorted(created_ids & remaining_ids)
            if missing_preexisting:
                step_fail(
                    "教程运行期间已有 sandbox 消失: "
                    + ", ".join(missing_preexisting)
                )
            else:
                step_ok(f"运行前的 {len(preexisting_ids)} 个 sandbox 全部保留")
            if leaked_tutorial:
                step_fail("以下 tutorial sandbox 未清理: " + ", ".join(leaked_tutorial))
            else:
                step_ok("本次 tutorial 创建的 sandbox 已全部清理")

    # ----------------------------------------------------------
    # 完成
    # ----------------------------------------------------------
    banner("全部测试完成")
    if FAILURES:
        print(f"  完成，但有 {len(FAILURES)} 个失败项:")
        for failure in FAILURES:
            print(f"    - {failure}")
        print()
        raise SystemExit(1)
    print("  所有远程 SDK 接口已依次测试并通过。\n")


if __name__ == "__main__":
    main()
