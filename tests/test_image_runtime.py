from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from crab.errors import (
    ImageAuthenticationError,
    ImageNotFoundError,
    ImagePlatformError,
    ImagePullError,
    ImageRateLimitError,
    ImageReferenceError,
)
from crab.engine import EngineConfig
from integrations.sandboxes.runtime.image import (
    _classify_pull_error,
    container_rootfs_tar_filter,
    export_image_rootfs,
    normalize_public_image_reference,
    resolve_image,
)


class _FakeDockerClient:
    def __init__(self, *, reference: str = "docker.io/library/python:3.12-slim") -> None:
        self.reference = reference
        self.image_id = "a" * 64
        self.digest = "sha256:" + "b" * 64
        self.present = False
        self.pull_count = 0
        self.create_count = 0
        self.export_count = 0
        self.removed_containers: list[str] = []
        self._lock = threading.Lock()

    def run(
        self,
        args,
        *,
        check=False,
        capture_output=False,
        text=False,
        stdout=None,
        stderr=None,
        timeout=None,
    ):
        _ = (check, capture_output, text, stderr, timeout)
        with self._lock:
            if args[:2] == ["image", "inspect"]:
                if not self.present:
                    return subprocess.CompletedProcess(args, 1, "", "not found")
                payload = [
                    {
                        "Id": f"sha256:{self.image_id}",
                        "RepoDigests": [f"docker.io/library/python@{self.digest}"],
                        "Os": "linux",
                        "Architecture": "amd64",
                        "Size": 1024,
                    }
                ]
                return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")
            if args and args[0] == "pull":
                self.pull_count += 1
                self.present = True
                return subprocess.CompletedProcess(args, 0, "pulled", "")
            if args and args[0] == "create":
                self.create_count += 1
                return subprocess.CompletedProcess(args, 0, "container-1\n", "")
            if args and args[0] == "export":
                self.export_count += 1
                assert stdout is not None
                with tarfile.open(fileobj=stdout, mode="w") as archive:
                    for name, payload in (
                        ("bin/sh", b"#!/bin/sh\n"),
                        ("bin/sleep", b"sleep\n"),
                    ):
                        member = tarfile.TarInfo(name)
                        member.mode = 0o755
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
                return subprocess.CompletedProcess(args, 0, None, b"")
            if args[:2] == ["rm", "-f"]:
                self.removed_containers.append(str(args[2]))
                return subprocess.CompletedProcess(args, 0, "", "")
            if args[:2] == ["image", "rm"]:
                return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(f"unexpected docker command: {args!r}")


class ImageResolutionTests(unittest.TestCase):
    def test_normalizes_tags_and_digest_pins(self) -> None:
        self.assertEqual(
            normalize_public_image_reference("python:3.12-slim"),
            ("docker.io/library/python:3.12-slim", "docker.io"),
        )
        digest = "sha256:" + "1" * 64
        self.assertEqual(
            normalize_public_image_reference(f"docker.io/library/python@{digest}"),
            (f"docker.io/library/python@{digest}", "docker.io"),
        )

    def test_rejects_malformed_or_uppercase_repository(self) -> None:
        for reference in ("", "https://docker.io/python", "Library/Python:3.12"):
            with self.subTest(reference=reference):
                with self.assertRaises(ImageReferenceError):
                    normalize_public_image_reference(reference)

    def test_concurrent_cold_resolve_pulls_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            docker = _FakeDockerClient()
            results = []
            failures = []

            def worker() -> None:
                try:
                    results.append(
                        resolve_image(
                            reference="python:3.12-slim",
                            cache_root=Path(temp_dir),
                            min_free_bytes=1,
                            docker=docker,
                        )
                    )
                except Exception as exc:  # pragma: no cover - asserted below
                    failures.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5.0)

            self.assertEqual(failures, [])
            self.assertEqual(len(results), 2)
            self.assertEqual(docker.pull_count, 1)
            self.assertEqual({result.image_id for result in results}, {"a" * 64})
            self.assertEqual({result.digest for result in results}, {"sha256:" + "b" * 64})
            self.assertEqual(sum(result.cache_hit for result in results), 1)

    def test_pull_errors_have_stable_categories(self) -> None:
        cases = (
            ("manifest unknown", ImageNotFoundError),
            (
                "failed to resolve reference docker.io/library/missing:latest: "
                "unexpected status from HEAD request to "
                "https://mirror.example/v2/library/missing/manifests/latest: "
                "403 Forbidden",
                ImageNotFoundError,
            ),
            ("unauthorized: authentication required", ImageAuthenticationError),
            ("toomanyrequests: rate limit exceeded", ImageRateLimitError),
            ("no matching manifest for linux/amd64", ImagePlatformError),
        )
        for output, expected in cases:
            with self.subTest(output=output):
                self.assertIsInstance(_classify_pull_error("example", output), expected)

    def test_inspect_daemon_failure_is_not_misreported_as_cache_miss(self) -> None:
        class BrokenInspectDocker(_FakeDockerClient):
            def run(self, args, **kwargs):
                if args[:2] == ["image", "inspect"]:
                    return subprocess.CompletedProcess(
                        args,
                        1,
                        "",
                        "Cannot connect to the Docker daemon",
                    )
                return super().run(args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            docker = BrokenInspectDocker()
            with self.assertRaisesRegex(ImagePullError, "Cannot connect"):
                resolve_image(
                    reference="python:3.12-slim",
                    cache_root=Path(temp_dir),
                    min_free_bytes=1,
                    docker=docker,
                )
            self.assertEqual(docker.pull_count, 0)

    def test_concurrent_export_publishes_one_compatible_rootfs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docker = _FakeDockerClient()
            docker.present = True
            results = []

            def worker() -> None:
                results.append(
                    export_image_rootfs(
                        tag=docker.reference,
                        output_dir=root / docker.image_id,
                        cache_root=root,
                        image_id=docker.image_id,
                        image_size_bytes=1024,
                        max_image_bytes=1024 * 1024,
                        cache_max_bytes=1024 * 1024,
                        min_free_bytes=1,
                        docker=docker,
                    )
                )

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5.0)

            self.assertEqual(len(results), 2)
            self.assertEqual(docker.create_count, 1)
            self.assertEqual(docker.export_count, 1)
            self.assertTrue((results[0] / "bin" / "sh").exists())
            self.assertEqual(results[0], results[1])

    def test_corrupt_published_rootfs_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docker = _FakeDockerClient()
            docker.present = True
            corrupt = root / docker.image_id / "rootfs"
            corrupt.mkdir(parents=True)
            (corrupt / "partial").write_text("bad", encoding="utf-8")

            result = export_image_rootfs(
                tag=docker.reference,
                output_dir=root / docker.image_id,
                cache_root=root,
                image_id=docker.image_id,
                image_size_bytes=1024,
                max_image_bytes=1024 * 1024,
                cache_max_bytes=1024 * 1024,
                min_free_bytes=1,
                docker=docker,
            )

            self.assertEqual(docker.export_count, 1)
            self.assertFalse((result / "partial").exists())
            self.assertTrue((result / "bin" / "sleep").exists())

    def test_empty_rootfs_and_stale_export_are_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docker = _FakeDockerClient()
            docker.present = True
            image_cache = root / docker.image_id
            (image_cache / "rootfs").mkdir(parents=True)
            stale = image_cache / "rootfs-export-stale"
            stale.mkdir()
            (stale / "partial").write_text("bad", encoding="utf-8")
            (image_cache / ".export-container").write_text(
                "stale\n", encoding="utf-8"
            )

            result = export_image_rootfs(
                tag=docker.reference,
                output_dir=image_cache,
                cache_root=root,
                image_id=docker.image_id,
                image_size_bytes=1024,
                max_image_bytes=1024 * 1024,
                cache_max_bytes=1024 * 1024,
                min_free_bytes=1,
                docker=docker,
            )

            self.assertEqual(docker.export_count, 1)
            self.assertFalse(stale.exists())
            self.assertFalse((image_cache / ".export-container").exists())
            self.assertEqual(len(docker.removed_containers), 2)
            self.assertEqual(
                docker.removed_containers[0], docker.removed_containers[1]
            )
            self.assertTrue((result / "bin" / "sh").exists())

    def test_interrupted_publish_recovers_valid_backup_without_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docker = _FakeDockerClient()
            docker.present = True
            image_cache = root / docker.image_id
            broken_rootfs = image_cache / "rootfs"
            broken_rootfs.mkdir(parents=True)
            (broken_rootfs / "partial").write_text("bad", encoding="utf-8")
            backup = image_cache / "rootfs.previous"
            (backup / "bin").mkdir(parents=True)
            (backup / "bin" / "sh").write_text("shell", encoding="utf-8")
            (backup / "bin" / "sleep").write_text("sleep", encoding="utf-8")

            result = export_image_rootfs(
                tag=docker.reference,
                output_dir=image_cache,
                cache_root=root,
                image_id=docker.image_id,
                image_size_bytes=1024,
                max_image_bytes=1024 * 1024,
                cache_max_bytes=1024 * 1024,
                min_free_bytes=1,
                docker=docker,
            )

            self.assertEqual(docker.export_count, 0)
            self.assertFalse(backup.exists())
            self.assertEqual((result / "bin" / "sh").read_text(), "shell")

    def test_export_setup_failure_removes_created_container(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docker = _FakeDockerClient()
            docker.present = True

            with mock.patch(
                "integrations.sandboxes.runtime.image.tempfile.mkdtemp",
                side_effect=OSError("cannot create staging directory"),
            ):
                with self.assertRaisesRegex(OSError, "cannot create"):
                    export_image_rootfs(
                        tag=docker.reference,
                        output_dir=root / docker.image_id,
                        cache_root=root,
                        image_id=docker.image_id,
                        image_size_bytes=1024,
                        max_image_bytes=1024 * 1024,
                        cache_max_bytes=1024 * 1024,
                        min_free_bytes=1,
                        docker=docker,
                    )

            self.assertEqual(len(docker.removed_containers), 1)
            self.assertTrue(
                docker.removed_containers[0].startswith("crab-export-")
            )


class ImageConfigurationTests(unittest.TestCase):
    def test_image_controls_preserve_explicit_empty_allowlist_and_zero(self) -> None:
        config = EngineConfig.from_mapping(
            {
                "images": {
                    "allowed_registries": [],
                    "max_image_bytes": 0,
                    "cache_max_bytes": 0,
                    "min_free_bytes": 0,
                }
            }
        )

        self.assertEqual(config.image_allowed_registries, ())
        self.assertEqual(config.image_max_bytes, 0)
        self.assertEqual(config.image_cache_max_bytes, 0)
        self.assertEqual(config.image_min_free_bytes, 0)

    def test_image_list_controls_reject_scalar_strings(self) -> None:
        for key in (
            "allowed_registries",
            "allowed_references",
            "prewarm",
            "required_prewarm",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "list of strings"):
                    EngineConfig.from_mapping({"images": {key: "python:3.12-slim"}})


class ContainerRootfsTarFilterTests(unittest.TestCase):
    def test_absolute_container_symlink_is_rewritten_inside_rootfs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "rootfs.tar"
            extract_path = root / "extract"
            extract_path.mkdir()

            payload = b"hello\n"
            target = tarfile.TarInfo("usr/bin/tool")
            target.size = len(payload)
            link = tarfile.TarInfo("bin/tool")
            link.type = tarfile.SYMTYPE
            link.linkname = "/usr/bin/tool"

            with tarfile.open(archive_path, "w") as archive:
                archive.addfile(target, io.BytesIO(payload))
                archive.addfile(link)
            with tarfile.open(archive_path) as archive:
                archive.extractall(extract_path, filter=container_rootfs_tar_filter)

            self.assertEqual(os.readlink(extract_path / "bin" / "tool"), "../usr/bin/tool")
            self.assertEqual((extract_path / "bin" / "tool").read_bytes(), payload)

    def test_container_directory_mode_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "rootfs.tar"
            extract_path = root / "extract"
            extract_path.mkdir()

            tmp = tarfile.TarInfo("tmp")
            tmp.type = tarfile.DIRTYPE
            tmp.mode = 0o1777
            with tarfile.open(archive_path, "w") as archive:
                archive.addfile(tmp)
            with tarfile.open(archive_path) as archive:
                archive.extractall(extract_path, filter=container_rootfs_tar_filter)

            self.assertEqual((extract_path / "tmp").stat().st_mode & 0o7777, 0o1777)

    def test_parent_traversal_is_still_rejected(self) -> None:
        member = tarfile.TarInfo("../../outside")
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(tarfile.OutsideDestinationError):
                container_rootfs_tar_filter(member, temp_dir)

    def test_special_files_from_public_images_are_rejected(self) -> None:
        member = tarfile.TarInfo("dev/untrusted")
        member.type = tarfile.CHRTYPE
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(tarfile.SpecialFileError):
                container_rootfs_tar_filter(member, temp_dir)


if __name__ == "__main__":
    unittest.main()
