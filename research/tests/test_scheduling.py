from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import lidar_model_selection.scheduling as scheduling
from lidar_model_selection.scheduling import (
    GpuSlot,
    RunOutcome,
    build_train_command,
    discover_gpu_slots,
    schedule_runs,
)


def _run_id(index: int) -> str:
    return f"20260818T1200{index:02d}Z-run-{index:024x}"


class _FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        names: tuple[str, ...] = ("GPU zero",),
    ) -> None:
        self._available = available
        self._names = names

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return len(self._names)

    def get_device_name(self, index: int) -> str:
        return self._names[index]


def _fake_torch(cuda: _FakeCuda) -> SimpleNamespace:
    return SimpleNamespace(cuda=cuda)


def test_scheduling_module_does_not_eagerly_import_torch() -> None:
    source = Path(scheduling.__file__).read_text(encoding="utf-8")

    assert "import torch" not in source
    assert "from torch" not in source


def test_discover_gpu_slots_is_lazy_and_uses_unmasked_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = []

    def import_module(name: str) -> object:
        imported.append(name)
        return _fake_torch(_FakeCuda(names=("A", "B")))

    monkeypatch.setattr(scheduling.importlib, "import_module", import_module)

    slots = discover_gpu_slots(environment={})

    assert imported == ["torch"]
    assert slots == (
        GpuSlot(0, "0", "A"),
        GpuSlot(1, "1", "B"),
    )


def test_discover_gpu_slots_preserves_cuda_visibility_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scheduling.importlib,
        "import_module",
        lambda name: _fake_torch(_FakeCuda(names=("first", "second"))),
    )

    slots = discover_gpu_slots(
        environment={
            "CUDA_VISIBLE_DEVICES": " GPU-f00 , MIG-deadbeef , ignored "
        }
    )

    assert [(slot.logical_index, slot.visibility_token) for slot in slots] == [
        (0, "GPU-f00"),
        (1, "MIG-deadbeef"),
    ]


@pytest.mark.parametrize(
    "cuda",
    [
        _FakeCuda(available=False, names=("GPU",)),
        _FakeCuda(available=True, names=()),
    ],
)
def test_discover_gpu_slots_requires_cuda(
    monkeypatch: pytest.MonkeyPatch,
    cuda: _FakeCuda,
) -> None:
    monkeypatch.setattr(
        scheduling.importlib,
        "import_module",
        lambda name: _fake_torch(cuda),
    )

    with pytest.raises(RuntimeError, match="at least one CUDA"):
        discover_gpu_slots(environment={})


def test_discover_gpu_slots_reports_missing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_torch(name: str) -> object:
        error = ModuleNotFoundError("No module named 'torch'")
        error.name = "torch"
        raise error

    monkeypatch.setattr(scheduling.importlib, "import_module", missing_torch)

    with pytest.raises(RuntimeError, match="Torch is required"):
        discover_gpu_slots(environment={})


@pytest.mark.parametrize(
    "mask",
    ["GPU-only", "GPU-same,GPU-same"],
)
def test_discover_gpu_slots_rejects_inconsistent_visibility_masks(
    monkeypatch: pytest.MonkeyPatch,
    mask: str,
) -> None:
    monkeypatch.setattr(
        scheduling.importlib,
        "import_module",
        lambda name: _fake_torch(_FakeCuda(names=("A", "B"))),
    )

    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        discover_gpu_slots(environment={"CUDA_VISIBLE_DEVICES": mask})


def test_build_train_command_uses_explicit_tool_and_run_id(tmp_path: Path) -> None:
    command = build_train_command(
        Path("research/tools/train.py"),
        _run_id(1),
        python_executable=tmp_path / "python",
    )

    assert command == (
        str(tmp_path / "python"),
        str(Path("research/tools/train.py").absolute()),
        "--run",
        _run_id(1),
    )
    with pytest.raises(ValueError, match="run_id"):
        build_train_command(Path("train.py"), "not-a-run-id")


def test_run_outcome_is_terminal_and_immutable() -> None:
    gpu = GpuSlot(0, "GPU-a", "Test GPU")

    assert RunOutcome(_run_id(1), gpu, 0, None).successful is True
    assert RunOutcome(_run_id(2), gpu, 7, "failed").successful is False
    assert RunOutcome(_run_id(3), gpu, None, "could not start").successful is False
    with pytest.raises(ValueError, match="must contain an error"):
        RunOutcome(_run_id(4), gpu, 1, None)


class _CompletingProcess:
    def __init__(
        self,
        *,
        token: str,
        return_code: int,
        polls_before_completion: int,
        active_tokens: set[str],
    ) -> None:
        assert token not in active_tokens
        active_tokens.add(token)
        self.token = token
        self.return_code = return_code
        self.polls_left = polls_before_completion
        self.active_tokens = active_tokens

    def poll(self) -> int | None:
        if self.polls_left:
            self.polls_left -= 1
            return None
        self.active_tokens.discard(self.token)
        return self.return_code


def test_schedule_runs_uses_one_child_per_gpu_and_returns_input_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_ids = tuple(_run_id(index) for index in range(4))
    slots = (GpuSlot(0, "GPU-a", "A"), GpuSlot(1, "GPU-b", "B"))
    active_tokens: set[str] = set()
    launches = []

    def popen(command: tuple[str, ...], **kwargs: object) -> _CompletingProcess:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        token = environment["CUDA_VISIBLE_DEVICES"]
        launches.append((command[-1], token, kwargs["cwd"], environment.copy()))
        return _CompletingProcess(
            token=token,
            return_code=0,
            polls_before_completion=1 if command[-1] == run_ids[0] else 0,
            active_tokens=active_tokens,
        )

    monkeypatch.setattr(scheduling.subprocess, "Popen", popen)
    monkeypatch.setattr(scheduling.time, "sleep", lambda seconds: None)

    outcomes = schedule_runs(
        run_ids,
        slots,
        lambda run_id: ("train", "--run", run_id),
        cwd=tmp_path,
        environment={"BASE": "preserved"},
        poll_interval=0,
    )

    assert [outcome.run_id for outcome in outcomes] == list(run_ids)
    assert all(outcome.successful for outcome in outcomes)
    assert len(launches) == 4
    assert all(cwd == str(tmp_path) for _, _, cwd, _ in launches)
    assert all(env["BASE"] == "preserved" for _, _, _, env in launches)
    assert all(env["PYTHONUNBUFFERED"] == "1" for _, _, _, env in launches)
    assert not active_tokens


def test_schedule_runs_isolates_child_and_start_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_ids = tuple(_run_id(index) for index in range(3))
    slot = GpuSlot(0, "0", "GPU")
    launched = []

    class ImmediateProcess:
        def __init__(self, return_code: int) -> None:
            self.return_code = return_code

        def poll(self) -> int:
            return self.return_code

    def command(run_id: str) -> tuple[str, ...]:
        if run_id == run_ids[0]:
            raise ValueError("bad command")
        return ("train", run_id)

    def popen(arguments: tuple[str, ...], **kwargs: object) -> ImmediateProcess:
        launched.append(arguments[-1])
        return ImmediateProcess(9 if arguments[-1] == run_ids[1] else 0)

    monkeypatch.setattr(scheduling.subprocess, "Popen", popen)

    outcomes = schedule_runs(run_ids, (slot,), command)

    assert launched == [run_ids[1], run_ids[2]]
    assert outcomes[0].return_code is None
    assert "could not start: ValueError: bad command" == outcomes[0].error
    assert outcomes[1].return_code == 9
    assert outcomes[1].error == "child exited with code 9"
    assert outcomes[2].successful


def test_schedule_runs_rejects_missing_slots_and_duplicate_runs() -> None:
    with pytest.raises(RuntimeError, match="without a CUDA GPU"):
        schedule_runs((_run_id(1),), (), lambda run_id: ("train", run_id))
    with pytest.raises(ValueError, match="duplicates"):
        schedule_runs(
            (_run_id(1), _run_id(1)),
            (GpuSlot(0, "0", "GPU"),),
            lambda run_id: ("train", run_id),
        )
    assert schedule_runs((), (), lambda run_id: ("train", run_id)) == ()


class _InterruptProcess:
    def __init__(self, *, stubborn: bool) -> None:
        self.stubborn = stubborn
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float | None] = []

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminate_calls += 1

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.stubborn and self.kill_calls == 0:
            raise subprocess.TimeoutExpired("train", timeout)
        return -9 if self.kill_calls else -15

    def kill(self) -> None:
        self.kill_calls += 1


def test_keyboard_interrupt_terminates_waits_kills_and_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_InterruptProcess] = []
    original_processes = [
        _InterruptProcess(stubborn=False),
        _InterruptProcess(stubborn=True),
    ]

    def tracked_popen(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> _InterruptProcess:
        process = original_processes[len(created)]
        created.append(process)
        return process

    monkeypatch.setattr(scheduling.subprocess, "Popen", tracked_popen)

    def interrupt(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(scheduling.time, "sleep", interrupt)

    with pytest.raises(KeyboardInterrupt):
        schedule_runs(
            (_run_id(1), _run_id(2)),
            (GpuSlot(0, "0", "A"), GpuSlot(1, "1", "B")),
            lambda run_id: ("train", run_id),
            poll_interval=0.1,
            termination_timeout=2.5,
        )

    assert len(created) == 2
    assert [process.terminate_calls for process in created] == [1, 1]
    assert created[0].wait_calls == [2.5]
    assert created[0].kill_calls == 0
    assert created[1].wait_calls == [2.5, None]
    assert created[1].kill_calls == 1


def test_unexpected_poll_failure_also_cleans_up_active_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenProcess(_InterruptProcess):
        def __init__(self) -> None:
            super().__init__(stubborn=False)
            self.poll_calls = 0

        def poll(self) -> None:
            self.poll_calls += 1
            if self.poll_calls == 1:
                raise OSError("poll failed")
            return None

    process = BrokenProcess()
    monkeypatch.setattr(
        scheduling.subprocess,
        "Popen",
        lambda command, **kwargs: process,
    )

    with pytest.raises(OSError, match="poll failed"):
        schedule_runs(
            (_run_id(1),),
            (GpuSlot(0, "0", "GPU"),),
            lambda run_id: ("train", run_id),
        )

    assert process.terminate_calls == 1
    assert process.wait_calls == [10.0]


def test_cleanup_failures_do_not_mask_the_scheduler_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OriginalFailure(RuntimeError):
        pass

    class BrokenCleanupProcess:
        def __init__(self) -> None:
            self.poll_calls = 0

        def poll(self) -> None:
            self.poll_calls += 1
            if self.poll_calls == 1:
                raise OriginalFailure("original scheduler failure")
            raise RuntimeError("cleanup poll failure")

        def terminate(self) -> None:
            raise RuntimeError("cleanup terminate failure")

        def wait(self, timeout: float | None = None) -> int:
            raise RuntimeError("cleanup wait failure")

        def kill(self) -> None:
            raise RuntimeError("cleanup kill failure")

    process = BrokenCleanupProcess()
    monkeypatch.setattr(
        scheduling.subprocess,
        "Popen",
        lambda command, **kwargs: process,
    )

    with pytest.raises(OriginalFailure, match="original scheduler failure"):
        schedule_runs(
            (_run_id(1),),
            (GpuSlot(0, "0", "GPU"),),
            lambda run_id: ("train", run_id),
        )
