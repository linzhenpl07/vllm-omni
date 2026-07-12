#!/usr/bin/env python3
"""Throughput-scaling benchmark for diffusion replica data parallelism.

Sends `--num-requests` identical text-to-video requests (up to `--concurrency`
in flight) to a running vLLM-Omni video server via `POST /v1/videos/sync`
(blocks until the clip is produced) and reports:

    completed / failed, makespan (wall clock), throughput (videos/min),
    per-request latency p50/p90/max, and an output-md5 dedup check.

Measuring replica scaling
-------------------------
This client is replica-agnostic. Vary the *server's* replica count
(`runtime.num_replicas` in the stage config; see run_server.sh) and keep the
client workload fixed, then compare throughput:

    N=1 -> baseline
    N=2 -> expect ~2x
    N=4 -> expect ~4x   (near-linear is the goal)

Pass the single-replica videos/min via `--baseline-tpm` to print a scaling
factor for the current run.

Isolation / correctness (optional)
----------------------------------
`--save-dir` writes each returned clip. With a fixed `--seed`, outputs should be
byte-identical across replica counts (replicas are isolated and must not
cross-talk); compare md5s between an N=1 and an N=4 run.

Requires: requests (`pip install requests`).
"""

import argparse
import concurrent.futures as cf
import hashlib
import statistics
import time
from pathlib import Path

import requests


def one_request(args, idx):
    """Fire one /v1/videos/sync request; return (ok, latency, nbytes, md5, err)."""
    # multipart form fields match POST /v1/videos/sync in the omni API server
    form = {
        "prompt": (None, args.prompt),
        "size": (None, args.size),  # e.g. "832x480"
        "num_frames": (None, str(args.num_frames)),
        "num_inference_steps": (None, str(args.steps)),
        "seed": (None, str(args.seed)),  # fixed seed -> reproducible + comparable
    }
    if args.model:
        form["model"] = (None, args.model)
    if args.negative_prompt:
        form["negative_prompt"] = (None, args.negative_prompt)

    t0 = time.perf_counter()
    try:
        r = requests.post(f"{args.url}/v1/videos/sync", files=form, timeout=args.timeout)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return (False, dt, 0, "", f"HTTP {r.status_code}: {r.text[:160]}")
        body = r.content
        md5 = hashlib.md5(body).hexdigest()
        if args.save_dir:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.save_dir) / f"req{idx:03d}_seed{args.seed}.mp4").write_bytes(body)
        return (True, dt, len(body), md5, "")
    except Exception as e:  # noqa: BLE001 - report any client-side failure
        return (False, time.perf_counter() - t0, 0, "", repr(e))


def main():
    ap = argparse.ArgumentParser(description="Diffusion replica-DP throughput benchmark")
    ap.add_argument("--url", default="http://127.0.0.1:8098", help="server base URL (no path)")
    ap.add_argument("--num-requests", type=int, default=16, help="total requests to send")
    ap.add_argument("--concurrency", type=int, default=4, help="max in-flight (set >= replica count)")
    ap.add_argument("--prompt", default="A cat playing piano, cinematic, high detail")
    ap.add_argument("--negative-prompt", default="")
    ap.add_argument("--size", default="832x480", help="WIDTHxHEIGHT")
    ap.add_argument("--num-frames", type=int, default=33)
    ap.add_argument("--steps", type=int, default=30, help="num_inference_steps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="", help="optional; defaults to the served model")
    ap.add_argument("--timeout", type=float, default=1800, help="per-request timeout (s)")
    ap.add_argument("--save-dir", default="", help="save returned clips (for the isolation check)")
    ap.add_argument("--label", default="", help="run label, e.g. 'replicas=2'")
    ap.add_argument(
        "--baseline-tpm", type=float, default=0.0, help="single-replica videos/min, to print a scaling factor"
    )
    args = ap.parse_args()

    print(
        f"-> {args.url}/v1/videos/sync | {args.num_requests} reqs "
        f"| concurrency {args.concurrency} | {args.size} {args.num_frames}f {args.steps}steps "
        f"| {args.label}"
    )

    lat, ok, fail, md5s, errs = [], 0, 0, [], []
    wall0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one_request, args, i) for i in range(args.num_requests)]
        for i, f in enumerate(cf.as_completed(futs), 1):
            good, dt, nbytes, md5, err = f.result()
            if good:
                ok += 1
                lat.append(dt)
                md5s.append(md5)
            else:
                fail += 1
                errs.append(err)
            print(f"  [{i}/{args.num_requests}] {'ok' if good else 'FAIL'} {dt:6.1f}s {(nbytes / 1e6):5.1f}MB {err}")
    makespan = time.perf_counter() - wall0

    print(f"\n===== result {args.label} =====")
    print(f"completed {ok} / failed {fail} | makespan {makespan:.1f}s")
    if ok:
        tpm = ok / makespan * 60.0
        print(f"throughput: {tpm:.2f} videos/min  ({ok / makespan:.4f} videos/s)")
        print(
            f"latency (s): p50={statistics.median(lat):.1f} "
            f"p90={sorted(lat)[int(0.9 * len(lat)) - 1]:.1f} "
            f"min={min(lat):.1f} max={max(lat):.1f}"
        )
        uniq = set(md5s)
        print(
            f"output md5 dedup: {len(uniq)} unique"
            + (
                "  (all identical: determinism OK)"
                if len(uniq) == 1
                else "  (differ -- check sampling / replica isolation)"
            )
        )
        if args.baseline_tpm > 0:
            print(f"scaling vs baseline ({args.baseline_tpm:.2f}/min): {tpm / args.baseline_tpm:.2f}x")
    if errs:
        print("sample failures:", errs[:3])


if __name__ == "__main__":
    main()
