"""Crab SDK 远程接口全功能教程

本脚本演示如何从一台 *远程* 客户端机器连接 Crab Gateway（或 daemon），
并覆盖 SDK 暴露的所有远程操作，包括最新引入的富返回值 `ActionResult`、
异步 checkpoint / changeset、inspector 只读 peek、以及 Sandbox 级
`auto_checkpoint` 模式。

使用前准备:
  1. 确保目标机器已部署 crab-gateway（或 daemon）并在监听 HTTP 端口。
  2. 安装 SDK: pip install crab  （或在项目根目录 pip install -e .）
  3. 修改下方 GATEWAY_URL 和 API_KEY 为你的实际值。

运行:
    python examples/sdk/04_remote_service_tutorial.py
"""
from __future__ import annotations

import time
import traceback
from urllib.parse import urlparse

from crab import Engine, Sandbox
from crab.models import ExecEvent, ExecDone

# ============================================================
# ★ 用户配置区 ★  —— 修改下面两行即可
# ============================================================
GATEWAY_URL = "http://YOUR_GATEWAY_HOST:8900"  # Gateway 地址（含端口）
API_KEY = "YOUR_API_KEY"                       # 在 Gateway 上创建的 tenant key

# 创建沙箱时使用的镜像和资源配置
SANDBOX_IMAGE = "ubuntu:22.04"
SANDBOX_RESOURCES = {"memory": "2G", "cpus": 2}


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
    print(f"  ✗ {msg}")


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
    # 用于最终统一清理的沙箱列表
    sandboxes_to_kill: list[Sandbox] = []

    # ----------------------------------------------------------
    # 1. 连接 Gateway
    # ----------------------------------------------------------
    banner("1. 连接 Gateway")
    print(f"  URL: {GATEWAY_URL}")

    engine = Engine.connect(url=GATEWAY_URL, api_key=API_KEY)
    step_ok("Engine 连接成功")
    print(f"  Engine 类型: {type(engine).__name__}")

    try:
        # ----------------------------------------------------------
        # 2. 远程创建沙箱
        # ----------------------------------------------------------
        banner("2. 远程创建沙箱")
        print(f"  镜像: {SANDBOX_IMAGE}")
        print(f"  资源: {SANDBOX_RESOURCES}")

        sandbox = Sandbox(
            image=SANDBOX_IMAGE,
            resources=SANDBOX_RESOURCES,
            engine=engine,
        )
        sandboxes_to_kill.append(sandbox)
        step_ok(f"沙箱创建成功, id = {sandbox.sandbox_id}")

        # ----------------------------------------------------------
        # 3. 列出当前 tenant 的沙箱 (engine.list_sandboxes)
        # ----------------------------------------------------------
        # 演示新增的 list_sandboxes()：走 GET /sandboxes 路由，返回
        # 当前 API key 可见的所有沙箱条目（gateway 会按 tenant 过滤）。
        banner("3. 列出沙箱 (engine.list_sandboxes)")
        sandboxes = safe_run("列出沙箱", engine.list_sandboxes) or []
        step_ok(f"当前沙箱列表: {len(sandboxes)} 个")
        for s in sandboxes:
            print(f"    {s.get('sandbox_id')} - {s.get('status', '?')}")

        # ----------------------------------------------------------
        # 4. 执行命令（一次性）
        # ----------------------------------------------------------
        banner("4. 执行命令 (commands.run)")
        result = sandbox.commands.run("echo hello && uname -a && whoami")
        step_ok(f"返回码: {result.returncode}")
        print(f"  stdout:\n{result.stdout.rstrip()}")

        # ----------------------------------------------------------
        # 5. 富返回值 — checkpoint + observe
        # ----------------------------------------------------------
        # 演示 commands.run(checkpoint=True, observe=True) 的富返回值：
        #   * checkpoint_id 在客户端侧预分配（ckpt-<uuid>），run() 立刻
        #     返回；后台线程执行真正的 checkpoint。
        #   * observe=True 会调用 daemon 的只读 inspector peek 路由，
        #     不重置任何游标。
        #
        # 语义（重要）：observe 返回的 filesystem_changed / process_changed
        # 表示 “本次 action 是否改变了状态” —— SDK 在 *启动后台 checkpoint
        # 之前* 就完成 peek，因此读到的是 checkpoint 重置游标之前的干净
        # 状态。checkpoint 完成后会调 host-inspector.reset() 把两个游标
        # 清零；若 peek 排在 checkpoint 之后，就会与该 reset 竞争而报 False
        # （这是一个历史 bug，已修复；参见 test_peek_runs_before_async_
        # checkpoint_reset）。所以下面 mutating 的命令应该稳定看到
        # filesystem_changed=True。
        banner("5. 富返回值 — checkpoint + observe")
        result = sandbox.commands.run(
            "echo 'hello' > /tmp/observed.txt && mkdir -p /opt/new",
            checkpoint=True,
            observe=True,
        )
        step_ok(f"返回码: {result.returncode}")
        print(f"  checkpoint_id (预分配): {result.checkpoint.checkpoint_id}")
        print(f"  checkpoint 完成?: {result.checkpoint.done}")
        print(f"  filesystem_changed: {result.filesystem_changed}")
        print(f"  process_changed: {result.process_changed}")
        # 阻塞等待后台 checkpoint 完成
        try:
            ckpt_id = result.checkpoint.wait(timeout=60.0)
            step_ok(f"checkpoint 已完成: {ckpt_id}")
        except Exception as exc:
            step_fail(f"等待 checkpoint 失败: {exc}")

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
        result = sandbox.commands.run("touch /tmp/async_test", changeset=True)
        try:
            entries = result.changeset.wait(timeout=60.0)
            step_ok(f"异步 changeset: {len(entries)} 条变更")
        except Exception as exc:
            step_fail(f"等待异步 changeset 失败: {exc}")

        # 6b. 同步 changeset
        result = sandbox.commands.run(
            "rm /tmp/async_test", changeset=True, changeset_sync=True
        )
        if isinstance(result.changeset, list):
            step_ok(f"同步 changeset: {len(result.changeset)} 条变更")
        else:
            step_fail(f"同步 changeset 返回类型异常: {type(result.changeset).__name__}")

        # ----------------------------------------------------------
        # 7. auto_checkpoint 模式
        # ----------------------------------------------------------
        # Sandbox 级 auto_checkpoint=True：每次 commands.run 后自动
        # 触发后台 checkpoint，无需每次都传 checkpoint=True。预分配的
        # checkpoint_id 可以通过 sandbox.last_checkpoint_id 直接读出。
        banner("7. auto_checkpoint 模式")
        auto_sb = safe_run(
            "创建 auto_checkpoint 沙箱",
            Sandbox,
            image=SANDBOX_IMAGE,
            resources={"memory": "1G"},
            engine=engine,
            auto_checkpoint=True,
        )
        if auto_sb is not None:
            sandboxes_to_kill.append(auto_sb)
            step_ok(f"auto_checkpoint 沙箱创建成功: {auto_sb.sandbox_id}")
            auto_sb.commands.run("echo step1 > /tmp/s1")
            step_ok(f"自动 checkpoint: {auto_sb.last_checkpoint_id}")
            auto_sb.commands.run("echo step2 > /tmp/s2")
            step_ok(f"第二次自动 checkpoint: {auto_sb.last_checkpoint_id}")
        else:
            step_fail("auto_checkpoint 沙箱不可用，跳过后续演示")

        # ----------------------------------------------------------
        # 8. 流式执行
        # ----------------------------------------------------------
        banner("8. 流式执行 (commands.stream)")
        print("  命令: for i in 1 2 3; do echo step-$i; sleep 0.5; done")
        print("  输出:")

        for event in sandbox.commands.stream(
            "for i in 1 2 3; do echo step-$i; sleep 0.5; done"
        ):
            if isinstance(event, ExecEvent):
                print(f"    [{event.channel}] {event.text}", end="")
            elif isinstance(event, ExecDone):
                print(f"    [exit] returncode={event.returncode}")

        step_ok("流式执行完成")

        # ----------------------------------------------------------
        # 9. 查看文件变更 (changeset with force)
        # ----------------------------------------------------------
        # 使用 sandbox.changeset(force=True) 跳过 daemon 的 inspector gate
        # 优化。不启用 eBPF host inspector 时，gate 会保守地判定"无变更"
        # 而返回空列表；force=True 强制走 diff 逻辑，拿到真实的变更集。
        banner("9. 文件变更 (changeset, force=True)")
        print("  先写一个测试文件…")

        # 先做一次 checkpoint 作为 changeset 的基线
        base_ckpt = safe_run(
            "创建基线 checkpoint",
            sandbox.checkpoint,
            "changeset-base",
        )
        if base_ckpt:
            step_ok(f"基线 checkpoint: {base_ckpt}")
            # 写入测试文件
            sandbox.commands.run("echo 'hello changeset' > /tmp/changeset_test.txt")
            sandbox.commands.run("mkdir -p /opt/demo && echo 123 > /opt/demo/data.txt")
            step_ok("写入了 /tmp/changeset_test.txt 和 /opt/demo/data.txt")

            # force=True: 无需等 inspector 观察到变更即可拿到结果
            changes = safe_run(
                "查询 changeset(force=True)",
                sandbox.changeset,
                base_ckpt,
                force=True,
            )
            if changes is not None:
                step_ok(f"changeset 返回 {len(changes)} 条变更 (force 跳过 inspector gate)")
                for entry in changes[:10]:  # 最多显示 10 条
                    print(f"    {entry}")
        else:
            step_fail("跳过 changeset 测试 (checkpoint 不可用)")

        # ----------------------------------------------------------
        # 10. Checkpoint
        # ----------------------------------------------------------
        banner("10. Checkpoint")
        ckpt_id = safe_run("创建 checkpoint", sandbox.checkpoint, "tutorial-ckpt")
        if ckpt_id:
            step_ok(f"Checkpoint 创建成功: {ckpt_id}")
        else:
            step_fail("Checkpoint 不可用 (可能缺少 CRIU 或权限不足)")

        # ----------------------------------------------------------
        # 11. Fork
        # ----------------------------------------------------------
        banner("11. Fork")
        forks = safe_run("Fork 沙箱", sandbox.fork, 1)
        if forks:
            fork_sbx = forks[0]
            sandboxes_to_kill.append(fork_sbx)
            step_ok(f"Fork 成功, fork id = {fork_sbx.sandbox_id}")

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
        # 12. Transaction
        # ----------------------------------------------------------
        banner("12. Transaction (begin / exec / commit|abort)")
        txn = safe_run("开启事务", sandbox.begin, "tutorial-txn")
        if txn:
            step_ok(f"事务已开启: {txn}")
            # 在事务内执行操作
            try:
                txn_result = txn.exec("echo 'inside txn' > /tmp/txn_file.txt")
                step_ok(f"事务内 exec 返回码: {txn_result.returncode}")

                # 提交事务
                commit_result = txn.commit()
                step_ok(f"事务提交成功: {commit_result}")
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

        reconnected = Sandbox.connect(existing_id, engine=engine)
        step_ok(f"重新连接成功, id = {reconnected.sandbox_id}")

        # 验证可以执行命令
        re_result = reconnected.commands.run("echo 'reconnected!'")
        step_ok(f"通过重连沙箱执行命令: {re_result.stdout.rstrip()}")

        # ----------------------------------------------------------
        # 14. 端口暴露 (ports.expose)
        # ----------------------------------------------------------
        banner("14. 端口暴露 (ports.expose)")
        print("  在沙箱内启动一个简单 HTTP server，然后暴露端口…")

        try:
            # 在沙箱内后台起一个 python HTTP server
            sandbox.commands.run(
                "python3 -m http.server 8080 --directory /tmp &",
                timeout=3.0,
            )
            # 等待 server 启动
            time.sleep(1)

            # 暴露端口
            allocation = sandbox.ports.expose(8080)
            step_ok("端口暴露成功:")
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
                    step_ok(f"外部访问成功 (HTTP {resp.status}), 响应前 200 字节:")
                    print(f"    {body[:200]}")
            except Exception as fetch_exc:
                step_fail(f"外部访问失败 (可能网络不通): {fetch_exc}")
                print("    提示: 端口已暴露，但从当前网络可能无法直接访问 gateway 主机。")

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
                sbx.kill()
                step_ok(f"已清理沙箱: {sid}")
            except Exception as kill_exc:
                step_fail(f"清理沙箱失败: {kill_exc}")

    # ----------------------------------------------------------
    # 完成
    # ----------------------------------------------------------
    banner("全部测试完成")
    print("  所有远程 SDK 接口已依次测试。")
    print("  如有失败项，请检查 Gateway/Daemon 配置和网络连通性。\n")


if __name__ == "__main__":
    main()
