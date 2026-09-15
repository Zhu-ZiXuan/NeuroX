"""Schedule independent model checks on free GPUs, retrying OOM after one hour."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

from .evaluate import DATA
from .factory import PRESETS, ROOT

ARCHITECTURES = (
    "lenet",
    "mlp",
    "rnn",
    "convrnn",
    "spikingvgg9",
    "spikingvgg16",
    "msresnet34",
    "msresnet50",
    "sewresnet34",
    "sewresnet50",
    "spikformer256",
    "qkformer256",
    "qkformer384",
    "spikingresformer256",
    "spikingresformer384",
    "metaspikeformer256",
)


def gpu_status() -> list[tuple[int, ...]]:
    output = subprocess.check_output(
        ["/usr/bin/nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [tuple(map(int, row.split(","))) for row in output.strip().splitlines()]


def _worker(arguments: list[str], gpu: int, log: str) -> None:
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", TORCHINDUCTOR_COMPILE_THREADS="2")
    os.chdir(ROOT)
    sys.argv = ["energy", *arguments]
    with Path(log).open("w") as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        from .evaluate import main as evaluate

        evaluate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--datasets", nargs="+", default=list(DATA))
    parser.add_argument("--models", nargs="+", default=list(ARCHITECTURES))
    parser.add_argument(
        "--families",
        nargs="+",
        default=["example", "hardware_comparable", "lenet_sparse_adc", "soul_fullgrid_sparse_adc"],
    )
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=list(PRESETS))
    parser.add_argument("--min-free-mib", type=int, default=12000)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for family in args.families:
        cases = [
            (model, dataset)
            for model in (args.models if family == "soul_fullgrid_sparse_adc" else ["lenet"])
            for dataset in args.datasets
        ]
        if family == "example":
            cases = [(model, "ucihar") for model in ("lenet", "bert", "sorbet")]
        elif family == "hardware_comparable":
            cases = [("bert", "ucihar")]
        for model, dataset in cases:
            for preset in args.presets:
                key = f"{family}_{dataset}_{model}_{preset}"
                if not (output_dir / f"{key}.json").exists():
                    jobs.append(
                        {
                            "family": family,
                            "model": model,
                            "dataset": dataset,
                            "preset": preset,
                            "key": key,
                            "ready": 0.0,
                        }
                    )
    running = {}
    failures = []
    while jobs or running:
        for gpu, (process, job) in list(running.items()):
            status = process.exitcode
            if status is None:
                continue
            process.join()
            del running[gpu]
            if status:
                log = (output_dir / f"{job['key']}.log").read_text()
                if "out of memory" in log.lower():
                    job["ready"] = time.monotonic() + 3600
                    jobs.append(job)
                else:
                    failures.append(job | {"exit_code": status})
            sys.stdout.write(f"completed {job['key']} exit={status}\n")
            sys.stdout.flush()
        for gpu, free, _utilization in sorted(gpu_status(), key=lambda row: (-row[1], row[2])):
            if gpu in running or free < args.min_free_mib:
                continue
            eligible = next((job for job in jobs if job["ready"] <= time.monotonic()), None)
            if eligible is None:
                continue
            jobs.remove(eligible)
            log = output_dir / f"{eligible['key']}.log"
            command = [
                sys.executable,
                "-m",
                "example.energy.evaluate",
                "--family",
                eligible["family"],
                "--model",
                eligible["model"],
                "--dataset",
                eligible["dataset"],
                "--preset",
                eligible["preset"],
                "--num-samples",
                str(args.num_samples),
                "--device",
                "cuda:0",
                "--output",
                str(output_dir / f"{eligible['key']}.json"),
            ]
            process = multiprocessing.get_context("spawn").Process(target=_worker, args=(command[3:], gpu, str(log)))
            process.start()
            running[gpu] = (process, eligible)
            sys.stdout.write(f"started {eligible['key']} gpu={gpu} free_mib={free}\n")
            sys.stdout.flush()
        (output_dir / "failures.json").write_text(json.dumps(failures, indent=2))
        if jobs or running:
            time.sleep(10)
    if failures:
        raise SystemExit(f"{len(failures)} jobs failed; see failures.json")


if __name__ == "__main__":
    main()
