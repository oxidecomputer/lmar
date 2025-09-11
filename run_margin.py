#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_margin.py
End-to-end orchestrator for:
  1) staging and running 'lmar' on a remote target,
  2) fetching produced archives (margin-*.zip / .tar.gz / .tgz / .tar),
  3) summarizing with margin_summary.py,
  4) collecting stats with collect_summaries.py.

Repo resolution:
  - Explicit: --lmar, --margin-summary, --collect-summaries
  - Single knob: --repo-dir (derives the three above)
  - Env: LMAR_REPO (same derivation)
  - Auto-discover: walk up from __file__ and CWD to find an lmar repo
  - PATH fallback for lmar if not found in a repo

Output layout (default, sessionized):
  <outdir>/<BOARD_SN>/
    raw/run-<N>/
    summaries/run-<N>/
    stats/run-<N>/

Default session is incremental run-<N> per board/outdir. Use --session to override.

Stats scope:
  --stats-scope board   -> stats over all runs under <SN>/summaries (default)
  --stats-scope run     -> stats using only this session summaries/run-<N>
                           If --combine-all is also set, we additionally write a
                           board-level combined file to <SN>/stats/.
"""

import argparse
import os
import sys
import shlex
import subprocess
import pathlib
import re
from typing import List, Tuple, Optional
from datetime import datetime

ARCH_PATTERNS = ("*.zip", "*.tar.gz", "*.tgz", "*.tar")


# ---------- small helpers ----------

def which_or_die(name: str) -> str:
    from shutil import which
    p = which(name)
    if not p:
        sys.stderr.write(f"ERROR: required command not found in PATH: {name}\n")
        sys.exit(2)
    return p

def run(
    cmd: List[str],
    *,
    check: bool = True,
    capture: bool = False,
    echo: bool = True,
    cwd: Optional[str] = None,
    merge_stderr: bool = True,
) -> subprocess.CompletedProcess:
    if echo:
        print("+", " ".join(shlex.quote(c) for c in cmd))
    try:
        if capture:
            return subprocess.run(
                cmd,
                check=check,
                text=True,
                stdout=subprocess.PIPE,
                stderr=(subprocess.STDOUT if merge_stderr else None),
                cwd=cwd,
            )
        else:
            return subprocess.run(cmd, check=check, cwd=cwd)
    except subprocess.CalledProcessError as e:
        out = getattr(e, "stdout", None)
        if out:
            sys.stderr.write("\n==== COMMAND OUTPUT BEGIN ====\n")
            sys.stderr.write(out)
            if not out.endswith("\n"):
                sys.stderr.write("\n")
            sys.stderr.write("==== COMMAND OUTPUT END ====\n")
        sys.stderr.write(f"FAILED: {' '.join(shlex.quote(c) for c in cmd)}\n")
        raise

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def joinp(*parts: str) -> str:
    return os.path.join(*parts)

def fail(msg: str) -> None:
    sys.stderr.write("ERROR: " + msg + "\n")
    sys.exit(2)

def info(msg: str) -> None:
    print(msg, flush=True)

def sh_c_quote(s: str) -> str:
    # Safely single-quote a string for 'sh -c ...'
    return "'" + s.replace("'", "'\"'\"'") + "'"


# ---------- board SN / hostname helpers ----------

def _sanitize_board_sn(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9\\-]+", "-", name).strip("-")
    if len(name) > 64:
        name = name[:64].rstrip("-")
    return name or "UNKNOWN"

def _looks_like_board(sn: str) -> bool:
    return bool(re.match(r"^(?=.*[A-Za-z])(?=.*\\d)[A-Za-z0-9\\-]+$", sn))

def ssh_target(remote: str, port: Optional[int], key: Optional[str], extra_opts: List[str], verbose: bool) -> List[str]:
    cmd = ["ssh"]
    if verbose:
        cmd += ["-v"]
    if port:
        cmd += ["-p", str(port)]
    if key:
        cmd += ["-i", key]
    if extra_opts:
        for opt in extra_opts:
            cmd += ["-o", opt]
    cmd.append(remote)
    return cmd

def remote_hostname(remote: str, port: Optional[int], key: Optional[str], extra_opts: List[str], verbose: bool) -> str:
    payload = "hostname -s 2>/dev/null || hostname 2>/dev/null || uname -n"
    cmd = ssh_target(remote, port, key, extra_opts, verbose) + ["sh", "-c", sh_c_quote(payload)]
    cp = run(cmd, check=True, capture=True, merge_stderr=False)
    host = (cp.stdout or "").strip().splitlines()[0].strip()
    if not host:
        fail("failed to read remote hostname")
    host = host.split(".")[0]
    sn = _sanitize_board_sn(host)
    if not _looks_like_board(sn):
        info(f"WARNING: remote hostname '{host}' sanitized to '{sn}', which may not look like a board SN.")
    else:
        info(f"remote hostname detected as board SN: {sn}")
    return sn


# ---------- repo/tool resolution ----------

def _is_repo_dir(path: str) -> bool:
    has_scripts = os.path.isfile(os.path.join(path, "margin_summary.py")) and \
                  os.path.isfile(os.path.join(path, "collect_summaries.py"))
    has_cargo = os.path.isfile(os.path.join(path, "Cargo.toml"))
    return has_scripts or has_cargo

def _discover_repo_candidates() -> List[str]:
    cands: List[str] = []
    try:
        here = os.path.abspath(os.path.dirname(__file__))
        cur = here
        while True:
            cands.append(cur)
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    except Exception:
        pass
    try:
        cur = os.path.abspath(os.getcwd())
        while cur not in cands:
            cands.append(cur)
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    except Exception:
        pass
    seen = set()
    uniq: List[str] = []
    for c in cands:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq

def _resolve_from_repo(repo_dir: str) -> Tuple[str, str, str]:
    repo_dir = os.path.abspath(repo_dir)
    lmar = os.path.join(repo_dir, "target", "release", "lmar")
    ms  = os.path.join(repo_dir, "margin_summary.py")
    cs  = os.path.join(repo_dir, "collect_summaries.py")
    return lmar, ms, cs

def resolve_tools(args) -> Tuple[str, str, str, Optional[str]]:
    lmar = args.lmar
    ms   = args.margin_summary
    cs   = args.collect_summaries
    repo_used: Optional[str] = None

    need_repo = (not lmar) or (not ms) or (not cs)

    repo_dir = None
    if args.repo_dir:
        repo_dir = os.path.abspath(args.repo_dir)
    elif os.environ.get("LMAR_REPO"):
        repo_dir = os.path.abspath(os.environ["LMAR_REPO"])

    if need_repo and repo_dir:
        if not _is_repo_dir(repo_dir):
            fail(f"--repo-dir looks wrong: {repo_dir}")
        rlmar, rms, rcs = _resolve_from_repo(repo_dir)
        lmar = lmar or rlmar
        ms   = ms   or rms
        cs   = cs   or rcs
        repo_used = repo_dir

    if need_repo and (not lmar or not ms or not cs):
        for cand in _discover_repo_candidates():
            if _is_repo_dir(cand):
                rlmar, rms, rcs = _resolve_from_repo(cand)
                if not lmar: lmar = rlmar
                if not ms:   ms   = rms
                if not cs:   cs   = rcs
                repo_used = cand
                break

    if not lmar:
        from shutil import which
        lmar = which("lmar")

    if not lmar:
        fail("cannot resolve path to 'lmar' binary (use --lmar, --repo-dir, LMAR_REPO, or ensure it is in PATH)")
    if not os.path.isfile(lmar):
        fail(f"lmar not found at: {lmar}")
    if not ms or not os.path.isfile(ms):
        fail("cannot resolve margin_summary.py (use --margin-summary, --repo-dir, or LMAR_REPO)")
    if not cs or not os.path.isfile(cs):
        fail("cannot resolve collect_summaries.py (use --collect-summaries, --repo-dir, or LMAR_REPO)")

    return os.path.abspath(lmar), os.path.abspath(ms), os.path.abspath(cs), repo_used


# ---------- remote ops ----------

def scp_to(remote: str, port: Optional[int], key: Optional[str], extra_opts: List[str], local_path: str, remote_path: str, verbose: bool) -> None:
    cmd = ["scp"]
    if verbose:
        cmd += ["-v"]
    if port:
        cmd += ["-P", str(port)]
    if key:
        cmd += ["-i", key]
    if extra_opts:
        for opt in extra_opts:
            cmd += ["-o", opt]
    cmd += [local_path, f"{remote}:{remote_path}"]
    run(cmd, check=True)

def scp_from(remote: str, port: Optional[int], key: Optional[str], extra_opts: List[str], remote_path: str, local_dir: str, verbose: bool) -> None:
    cmd = ["scp"]
    if verbose:
        cmd += ["-v"]
    if port:
        cmd += ["-P", str(port)]
    if key:
        cmd += ["-i", key]
    if extra_opts:
        for opt in extra_opts:
            cmd += ["-o", opt]
    cmd += [f"{remote}:{remote_path}", local_dir]
    run(cmd, check=True)

def remote_mktemp_dir(remote: str, port: Optional[int], key: Optional[str], extra_opts: List[str], verbose: bool) -> str:
    payload = (
        "d=''; "
        "if command -v mktemp >/dev/null 2>&1; then "
        "  d=$(mktemp -d -t lmar_run.XXXXXXXX 2>/dev/null) || true; "
        "  [ -n \"$d\" ] || d=$(mktemp -d /tmp/lmar_run.XXXXXX 2>/dev/null) || true; "
        "fi; "
        "[ -n \"$d\" ] || { d=/tmp/lmar_run_$$; mkdir -p \"$d\"; }; "
        "cd \"$d\" && pwd"
    )
    cmd = ssh_target(remote, port, key, extra_opts, verbose) + ["sh", "-c", sh_c_quote(payload)]
    cp = run(cmd, check=True, capture=True, merge_stderr=False)
    d = cp.stdout.strip().splitlines()[-1].strip()
    if not d:
        fail("failed to obtain a remote temp directory")
    return d

def remote_list_archives(remote: str, port: Optional[int], key: Optional[str], extra_opts: List[str], rdir: str, verbose: bool) -> List[str]:
    pats = " ".join(shlex.quote(p) for p in ARCH_PATTERNS)
    payload = f"cd {shlex.quote(rdir)} && for p in {pats}; do for f in $p; do [ -f \"$f\" ] && printf '%s\\n' \"$(pwd)/$f\"; done; done"
    cmd = ssh_target(remote, port, key, extra_opts, verbose) + ["sh", "-c", sh_c_quote(payload)]
    cp = run(cmd, check=False, capture=True, merge_stderr=False)
    out = (cp.stdout or "").strip()
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


# ---------- session helpers ----------

def _list_existing_run_numbers(base_root: str, board_sn: Optional[str]) -> List[int]:
    if board_sn:
        roots = [joinp(base_root, board_sn, "raw"),
                 joinp(base_root, board_sn, "summaries"),
                 joinp(base_root, board_sn, "stats")]
    else:
        roots = [joinp(base_root, "raw"),
                 joinp(base_root, "summaries"),
                 joinp(base_root, "stats")]

    nums = set()
    pat = re.compile(r"^run-(\d+)(?:$|-)")
    for r in roots:
        try:
            for name in os.listdir(r):
                m = pat.match(name)
                if m:
                    try:
                        nums.add(int(m.group(1)))
                    except Exception:
                        pass
        except FileNotFoundError:
            continue
    return sorted(nums)

def gen_incremental_session_label(base_root: str, board_sn: Optional[str]) -> str:
    existing = _list_existing_run_numbers(base_root, board_sn)
    next_n = (existing[-1] + 1) if existing else 1
    return f"run-{next_n}"

def _any_session_dir_exists(base_root: str, board_sn: Optional[str], session: str) -> bool:
    if board_sn:
        roots = [joinp(base_root, board_sn, "raw"),
                 joinp(base_root, board_sn, "summaries"),
                 joinp(base_root, board_sn, "stats")]
    else:
        roots = [joinp(base_root, "raw"),
                 joinp(base_root, "summaries"),
                 joinp(base_root, "stats")]
    for r in roots:
        if os.path.isdir(joinp(r, session)):
            return True
    return False

def ensure_unique_session(base_root: str, board_sn: Optional[str], session: str, policy: str) -> str:
    if policy not in ("timestamp", "increment", "error", "overwrite"):
        policy = "timestamp"
    if not _any_session_dir_exists(base_root, board_sn, session):
        return session
    if policy == "overwrite":
        return session
    if policy == "error":
        fail(f"session '{session}' already exists under {base_root}")
    if policy == "timestamp":
        uniq = session + "-" + datetime.now().strftime("%Y%m%d%H%M%S")
        if _any_session_dir_exists(base_root, board_sn, uniq):
            policy = "increment"
            session = uniq
        else:
            return uniq
    if policy == "increment":
        base = session
        i = 2
        while _any_session_dir_exists(base_root, board_sn, f"{base}-{i}"):
            i += 1
        return f"{base}-{i}"
    return session


# ---------- pipeline steps ----------

def stage_and_run_lmar(args, lmar_args: List[str], lmar_path: str) -> Tuple[str, List[str]]:
    which_or_die("ssh")
    which_or_die("scp")

    remote = args.remote
    port = args.ssh_port
    key = args.ssh_key
    extra = args.ssh_opt or []
    verbose = args.ssh_verbose

    if not args.board_sn:
        info("==> probing remote hostname for board SN")
        args.board_sn = remote_hostname(remote, port, key, extra, verbose)

    info("==> creating remote work dir")
    rdir = args.remote_cwd
    if rdir:
        payload = f"mkdir -p {shlex.quote(rdir)} && cd {shlex.quote(rdir)} && pwd"
        cmd = ssh_target(remote, port, key, extra, verbose) + ["sh", "-c", sh_c_quote(payload)]
        cp = run(cmd, check=True, capture=True, merge_stderr=False)
        rdir = cp.stdout.strip().splitlines()[-1].strip()
    else:
        rdir = remote_mktemp_dir(remote, port, key, extra, verbose)
    info(f"remote work dir: {rdir}")

    if not os.path.isfile(lmar_path):
        fail(f"lmar not found: {lmar_path}")
    info("==> copying lmar to remote")
    scp_to(remote, port, key, extra, lmar_path, f"{rdir}/lmar", verbose)

    if not lmar_args:
        fail("No lmar arguments provided. Append them after '--' or via --lmar-args (try '-- --help').")

    run_prefix = "sudo " if args.sudo else ""
    lmar_cmd = " ".join([f"{run_prefix}./lmar"] + [shlex.quote(x) for x in lmar_args])
    info(f"==> running lmar: {lmar_cmd}")
    payload = f"cd {shlex.quote(rdir)} && chmod +x ./lmar && {lmar_cmd}"
    cmd = ssh_target(remote, port, key, extra, verbose) + ["sh", "-c", sh_c_quote(payload)]
    run(cmd, check=True, capture=False)

    info("==> checking for archives")
    archives = remote_list_archives(remote, port, key, extra, rdir, verbose)
    if not archives:
        fail("no archives produced by lmar (expected .zip / .tar.gz / .tgz / .tar); check remote logs")
    for a in archives:
        info(f"found: {a}")
    return rdir, archives

def fetch_archives(args, remote_run_dir: str, remote_archives: List[str]) -> List[str]:
    raw_dir = args.raw_dir
    ensure_dir(raw_dir)
    out_paths: List[str] = []
    for apath in remote_archives:
        info(f"==> fetching {apath} -> {raw_dir}/")
        scp_from(args.remote, args.ssh_port, args.ssh_key, args.ssh_opt or [], apath, raw_dir, args.ssh_verbose)
        local_name = os.path.join(raw_dir, os.path.basename(apath))
        out_paths.append(local_name)
    return out_paths

def run_margin_summary(args, archives: List[str], margin_summary_script: str) -> List[str]:
    ensure_dir(args.summaries_dir)
    if not archives:
        return []

    script = margin_summary_script
    if not os.path.isfile(script):
        fail(f"margin_summary.py not found: {script}")

    # Build unified summary basename: margin_summary_<SN>_<run>
    session_label = os.path.basename(os.path.normpath(args.summaries_dir))  # e.g. "run-2"
    summary_base = f"margin_summary_{args.board_sn or 'UNKNOWN'}_{session_label}"

    written: List[str] = []
    total = len(archives)
    for idx, arch in enumerate(archives, start=1):
        # First (or only) dataset => no suffix; multiple datasets get _2, _3, ...
        out_name = f"{summary_base}.txt" if total == 1 else f"{summary_base}_{idx}.txt"
        out_path = os.path.join(args.summaries_dir, out_name)

        cmd = [sys.executable, script, arch, "--outdir", args.summaries_dir, "--out", out_name]
        if args.pass_count_required is not None:
            cmd += ["-c", str(args.pass_count_required)]
        info(f"==> summarizing {arch} -> {out_name}")
        cp = run(cmd, check=True, capture=True, echo=True)

        # Prefer the tool's "Wrote ..." message, but fall back to our constructed path.
        wrote = False
        for line in (cp.stdout or "").splitlines():
            s = line.strip()
            if s.startswith("Wrote "):
                p = s.split("Wrote ", 1)[1].strip()
                if p.endswith(".txt"):
                    written.append(p)
                    wrote = True
        if not wrote and os.path.exists(out_path):
            written.append(out_path)

    if not written:
        # Final fallback to match any margin_summary_*.txt in the run's summaries dir
        for p in pathlib.Path(args.summaries_dir).glob("margin_summary_*.txt"):
            written.append(str(p))

    return sorted(set(written))

def run_collect_summaries(args, scan_root: str, outdir: str, collect_summaries_script: str) -> List[str]:
    ensure_dir(outdir)
    script = collect_summaries_script
    if not os.path.isfile(script):
        fail(f"collect_summaries.py not found: {script}")

    written: List[str] = []

    base_cmd = [sys.executable, script, "--scan-root", scan_root, "--outdir", outdir]
    if args.combine_all:
        base_cmd.append("--combine-all")
    if args.only_combined:
        base_cmd.append("--only-combined")
    if args.omit_n7_9:
        base_cmd.append("--omit-n7-9")
    if args.omit_ports:
        base_cmd += ["--omit-ports", *args.omit_ports]
    if args.names and args.pci_ids:
        base_cmd += ["--names", "--pci-ids", args.pci_ids]
    if args.pass_width is not None:
        base_cmd += ["--pass-width", str(args.pass_width)]
    if args.pass_height is not None:
        base_cmd += ["--pass-height", str(args.pass_height)]
    if args.width_limit is not None:
        base_cmd += ["--width-limit", str(args.width_limit)]
    if args.height_limit is not None:
        base_cmd += ["--height-limit", str(args.height_limit)]

    info(f"==> collecting stats from {scan_root}")
    cp = run(base_cmd, check=True, capture=True, echo=True)
    for line in (cp.stdout or "").splitlines():
        s = line.strip()
        if s.startswith("Wrote "):
            written.append(s.split("Wrote ", 1)[1].strip())

    if not written:
        for p in pathlib.Path(outdir).glob("stats_*.txt"):
            written.append(str(p))
        p = pathlib.Path(outdir) / "stats_all_runs.txt"
        if p.exists():
            written.append(str(p))

    return sorted(set(written))


# ---------- output layout (sessionized) ----------

def build_output_dirs(outbase: str,
                      board_sn: Optional[str],
                      session: Optional[str],
                      flat_output: bool,
                      on_collision: str) -> tuple[str, str, str, str]:
    base = os.path.abspath(outbase)

    if board_sn:
        board_root = joinp(base, board_sn)
    else:
        board_root = base

    final_session = ""
    if not flat_output:
        if session:
            final_session = session
        else:
            final_session = gen_incremental_session_label(base, board_sn)
        final_session = ensure_unique_session(base, board_sn, final_session, on_collision)

    if board_sn:
        raw_root = joinp(base, board_sn, "raw")
        sum_root = joinp(base, board_sn, "summaries")
        sta_root = joinp(base, board_sn, "stats")
    else:
        raw_root = joinp(base, "raw")
        sum_root = joinp(base, "summaries")
        sta_root = joinp(base, "stats")

    if flat_output:
        raw_dir = raw_root
        sum_dir = sum_root
        sta_dir = sta_root
    else:
        raw_dir = joinp(raw_root, final_session)
        sum_dir = joinp(sum_root, final_session)
        sta_dir = joinp(sta_root, final_session)

    return board_root, raw_dir, sum_dir, sta_dir


def write_session_note(stats_dir: str,
                       board_sn: Optional[str],
                       remote: Optional[str],
                       lmar_args: List[str],
                       resolved_paths: Tuple[str, str, str, Optional[str]]) -> None:
    ensure_dir(stats_dir)
    note = os.path.join(stats_dir, "SESSION.txt")
    lmar_path, ms_path, cs_path, repo_used = resolved_paths
    lines = []
    lines.append(f"board_sn: {board_sn or ''}")
    lines.append(f"remote: {remote or ''}")
    lines.append(f"local_start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"lmar_args: {' '.join(shlex.quote(x) for x in lmar_args)}")
    lines.append(f"tools.lmar: {lmar_path}")
    lines.append(f"tools.margin_summary: {ms_path}")
    lines.append(f"tools.collect_summaries: {cs_path}")
    lines.append(f"repo_dir: {repo_used or ''}")
    with open(note, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    info(f"wrote session note: {note}")


# ---------- CLI wiring ----------

def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(
        description="Run lmar on a remote host, fetch archives, summarize, and collect stats.",
        epilog=(
            "Examples:\n"
            "  run_margin.py --remote root@mb-0 --repo-dir /staff/tom/git/lmar --outdir . -- --help\n"
            "  run_margin.py --remote root@mb-0 --repo-dir /staff/tom/git/lmar --outdir . --stats-scope run -- 60/1/3 downstream -l all\n"
            "  run_margin.py --skip-remote --board-sn BRM13250013 --repo-dir /staff/tom/git/lmar --outdir .\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Repo/tool resolution
    ap.add_argument("--repo-dir", default=None, help="Root of the lmar repo (derives tool paths).")
    ap.add_argument("--lmar", default=None, help="Path to lmar binary (overrides repo-dir).")
    ap.add_argument("--margin-summary", default=None, help="Path to margin_summary.py (overrides repo-dir).")
    ap.add_argument("--collect-summaries", default=None, help="Path to collect_summaries.py (overrides repo-dir).")
    ap.add_argument("--print-resolved", action="store_true", help="Print resolved tools and exit.")
    ap.add_argument("--cargo-build", action="store_true", help="Run 'cargo build --release' inside repo-dir before uploading lmar.")
    ap.add_argument("--cargo-args", default="", help="Extra args for cargo build (quoted string).")

    # Remote
    ap.add_argument("--remote", help="[user@]host for ssh/scp (can be an SSH config alias)")
    ap.add_argument("--ssh-port", type=int, default=None, help="SSH port (default: 22)")
    ap.add_argument("--ssh-key", default=None, help="SSH private key path")
    ap.add_argument("--ssh-opt", action="append", default=[], help="Additional -o options for ssh/scp (repeatable)")
    ap.add_argument("--ssh-verbose", dest="ssh_verbose", action="store_true", default=False, help="Add -v to ssh/scp for debugging (default: off).")
    ap.add_argument("--remote-cwd", default=None, help="Remote working dir (default: mktemp -d)")
    ap.add_argument("--sudo", action="store_true", help="Prefix remote lmar execution with sudo")
    ap.add_argument("--timeout", type=int, default=0, help="Reserved (no-op).")

    # Output layout
    ap.add_argument("--board-sn", default=None, help="Board serial number (e.g., BRM13250013). If omitted, we use the remote hostname.")
    ap.add_argument("--outdir", default=".", help="Base output directory (default: current dir)")

    # Sessionization
    ap.add_argument("--session", default=None, help="Session label; default is incrementing run-<N> per board/outdir.")
    ap.add_argument('--run-label', dest='session', default=None,
                    help='Alias for --session; session label like run-7 or custom tag.')

    ap.add_argument("--on-collision", choices=["timestamp", "increment", "error", "overwrite"], default="increment",
                    help="If session already exists, choose how to uniquify (default: increment).")
    ap.add_argument("--flat-output", action="store_true",
                    help="Do not create per-session subdirs; write directly into raw/summaries/stats roots.")
    ap.add_argument("--write-session-note", action="store_true", help="Write stats/<SESSION>/SESSION.txt with metadata.")

    # Optional overrides (literal; no session appended if you use these)
    ap.add_argument("--raw-subdir", default=None, help="Override raw dir path (literal).")
    ap.add_argument("--summaries-subdir", default=None, help="Override summaries dir path (literal).")
    ap.add_argument("--stats-subdir", default=None, help="Override stats dir path (literal).")

    # Control which stages to run
    ap.add_argument("--skip-remote", action="store_true", help="Skip remote run; only process local archives in raw dir")
    ap.add_argument("--skip-summary", action="store_true", help="Skip margin_summary.py")
    ap.add_argument("--skip-stats", action="store_true", help="Skip collect_summaries.py")

    # Stats scoping
    ap.add_argument("--stats-scope", choices=["board", "run"], default="board",
                    help="board = stats over all runs for this board (default); run = stats only from this session's summaries.")

    # collect_summaries passthrough
    ap.add_argument("--combine-all", action="store_true", help="Also write a combined file (usage depends on scope).")
    ap.add_argument("--only-combined", action="store_true", help="Write only combined file (skip per-summary outputs).")
    ap.add_argument("--omit-n7-9", action="store_true", help="Omit N7/N8/N9 from Vendor/Device stats")
    ap.add_argument("--omit-ports", nargs="+", default=[], help="Additional ports to omit (e.g. N4 N5)")
    ap.add_argument("--names", action="store_true", help="Append vendor/device names (requires --pci-ids)")
    ap.add_argument("--pci-ids", default=None, help="Path to pci.ids")
    ap.add_argument("--pass-width", type=float, default=None, help="PASS threshold for width in %%UI")
    ap.add_argument("--pass-height", type=float, default=None, help="PASS threshold for height in V")
    ap.add_argument("--width-limit", type=float, default=None, help="Dataset gating min width in %%UI")
    ap.add_argument("--height-limit", type=float, default=None, help="Dataset gating min height in V")

    # margin_summary passthrough
    ap.add_argument("-c", "--pass-count-required", type=int, default=None,
                    help="In summarize step: treat PASS as Pass==1 and Count==THIS")

    # lmar args
    ap.add_argument("--lmar-args", default=None,
                    help="Quoted string of arguments to pass to lmar (e.g. --lmar-args '60/1/3 downstream -l all').")
    ap.add_argument("--lmar-arg", action="append", default=[],
                    help="Repeatable single lmar argument (e.g. --lmar-arg 60/1/3 --lmar-arg downstream --lmar-arg -l --lmar-arg all).")

    # Everything after '--' is passed to lmar as-is
    args, tail_lmar_args = ap.parse_known_args(argv)

    # Merge lmar args
    lmar_args: List[str] = []
    if args.lmar_args:
        lmar_args += shlex.split(args.lmar_args)
    if args.lmar_arg:
        lmar_args += args.lmar_arg
    lmar_args += tail_lmar_args

    # Resolve tools
    lmar_path, ms_path, cs_path, repo_used = resolve_tools(args)

    if args.print_resolved:
        print("Resolved tools:")
        print(f"  repo_dir:             {repo_used or '(none)'}")
        print(f"  lmar:                 {lmar_path}")
        print(f"  margin_summary.py:    {ms_path}")
        print(f"  collect_summaries.py: {cs_path}")
        return

    if args.cargo_build:
        if not repo_used:
            fail("--cargo-build requires a valid --repo-dir or LMAR_REPO or auto-discovered repo")
        which_or_die("cargo")
        cargo_cmd = ["cargo", "build", "--release"]
        extra = args.cargo_args.strip()
        if extra:
            cargo_cmd += shlex.split(extra)
        info("==> cargo build (release)")
        run(cargo_cmd, check=True, echo=True, cwd=repo_used)
        lmar_path, _, _, _ = resolve_tools(args)

    # Board SN from remote if needed
    if not args.skip_remote and not args.board_sn:
        which_or_die("ssh")
        args.board_sn = remote_hostname(args.remote, args.ssh_port, args.ssh_key, args.ssh_opt or [], args.ssh_verbose)

    # Build sessionized output dirs
    board_root, raw_d, sum_d, sta_d = build_output_dirs(
        args.outdir, args.board_sn, args.session, args.flat_output, args.on_collision
    )

    # Literal overrides
    if args.raw_subdir: raw_d = os.path.abspath(args.raw_subdir)
    if args.summaries_subdir: sum_d = os.path.abspath(args.summaries_subdir)
    if args.stats_subdir: sta_d = os.path.abspath(args.stats_subdir)

    args.raw_dir = raw_d
    args.summaries_dir = sum_d
    args.stats_dir = sta_d

    # Stage 1: remote run
    fetched_archives: List[str] = []
    if not args.skip_remote:
        _, r_archives = stage_and_run_lmar(args, lmar_args, lmar_path)
        fetched_archives = fetch_archives(args, args.remote_cwd or "", r_archives)
    else:
        info("==> skipping remote stage; looking for existing archives in raw dir")
        ensure_dir(args.raw_dir)
        for pat in ARCH_PATTERNS:
            for p in pathlib.Path(args.raw_dir).glob(pat):
                fetched_archives.append(str(p))
        fetched_archives = sorted(set(fetched_archives))
        if not fetched_archives:
            fail(f"no archives found in {args.raw_dir}")

    # Stage 2: summaries
    summaries: List[str] = []
    if not args.skip_summary:
        summaries = run_margin_summary(args, fetched_archives, ms_path)
        if summaries:
            info("==> summaries written:")
            for s in summaries:
                info(f"    {s}")
        else:
            info("==> margin_summary did not report written files (check output)")
    else:
        info("==> skipping summary stage")

    # Stage 3: stats
    if not args.skip_stats:

        if args.stats_scope == "run":
            # Per-run stats: ONLY the summaries from this session directory.
            scan_root_run = args.summaries_dir
            ensure_dir(args.stats_dir)
    
            # Gather summary files in this run directory.
            summary_files = sorted(str(p) for p in pathlib.Path(scan_root_run).glob("margin_summary_*.txt"))
            if not summary_files:
                # Fallback: any *.txt (still scoped to the run dir).
                summary_files = sorted(str(p) for p in pathlib.Path(scan_root_run).glob("*.txt"))
            if not summary_files:
                fail(f"no summary files found in {scan_root_run}")

            # Build a clear per-run stats file name.
            session_label = os.path.basename(os.path.normpath(scan_root_run))
            out_name = f"stats_{args.board_sn or 'UNKNOWN'}_{session_label}.txt"
            out_path = os.path.join(args.stats_dir, out_name)

            # Build direct-mode command (NO --scan-root).
            cmd = [sys.executable, cs_path, "--outdir", args.stats_dir, "--out", out_name]

            # Pass-through knobs that make sense in direct mode.
            if args.omit_n7_9:
                cmd.append("--omit-n7-9")
            if args.omit_ports:
                cmd += ["--omit-ports", *args.omit_ports]
            if args.names and args.pci_ids:
                cmd += ["--names", "--pci-ids", args.pci_ids]
            if args.pass_width is not None:
                cmd += ["--pass-width", str(args.pass_width)]
            if args.pass_height is not None:
                cmd += ["--pass-height", str(args.pass_height)]
            if args.width_limit is not None:
                cmd += ["--width-limit", str(args.width_limit)]
            if args.height_limit is not None:
                cmd += ["--height-limit", str(args.height_limit)]

            # Positional inputs = summary files for this run.
            cmd += summary_files

            info(f"==> per-run stats from {scan_root_run}")
            cp = run(cmd, check=True, capture=True, echo=True)
            wrote_any = False
            for line in (cp.stdout or "").splitlines():
                s = line.strip()
                if s.startswith("Wrote "):
                    info("    " + s)
                    wrote_any = True
            if not wrote_any and os.path.exists(out_path):
                info(f"    Wrote {out_path}")

            # Optionally also write a combined board-level file across all runs.
            if args.combine_all:
                board_stats_root = joinp(board_root, "stats")
                # Build a single board-wide combined file with a self-describing name.
                out_all = f"stats_{args.board_sn or 'UNKNOWN'}_all.txt"

                # Collect all per-run summary files under this board (direct mode)
                summaries_root = joinp(board_root, "summaries")
                summary_files = sorted(
                    str(p) for p in pathlib.Path(summaries_root).glob("run-*/margin_summary_*.txt")
                )

                if not summary_files:
                    info(f"==> board-level combined stats: no summaries under {summaries_root}, skipping")
                else:
                    # Prepare a direct-mode call to collect_summaries.py
                    cmd2 = [sys.executable, cs_path,
                            "--outdir", board_stats_root,
                            "--out", out_all]

                    # Pass-through knobs
                    if args.omit_n7_9:
                        cmd2.append("--omit-n7-9")
                    if args.omit_ports:
                        cmd2 += ["--omit-ports", *args.omit_ports]
                    if args.names and args.pci_ids:
                        cmd2 += ["--names", "--pci-ids", args.pci_ids]
                    if args.pass_width is not None:
                        cmd2 += ["--pass-width", str(args.pass_width)]
                    if args.pass_height is not None:
                        cmd2 += ["--pass-height", str(args.pass_height)]
                    if args.width_limit is not None:
                        cmd2 += ["--width-limit", str(args.width_limit)]
                    if args.height_limit is not None:
                        cmd2 += ["--height-limit", str(args.height_limit)]

                    # Positional inputs = all per-run summary files for this board
                    cmd2 += summary_files

                    ensure_dir(board_stats_root)
                    info("==> board-level combined stats (all runs)")
                    cp2 = run(cmd2, check=True, capture=True, echo=True)
                    wrote_any2 = False
                    for line in (cp2.stdout or "").splitlines():
                        s = line.strip()
                        if s.startswith("Wrote "):
                            info("    " + s)
                            wrote_any2 = True
                    if not wrote_any2:
                        out_path2 = joinp(board_stats_root, out_all)
                        if os.path.exists(out_path2):
                            info(f"    Wrote {out_path2}")

        else:
            # Board scope: use all runs under this board
            written = run_collect_summaries(args, board_root, args.stats_dir, cs_path)
            if written:
                info("==> board-scope stats written:")
                for w in written:
                    info(f"    {w}")
    else:
        info("==> skipping stats stage")

    # Optional session note
    if args.write_session_note and not args.flat_output:
        write_session_note(args.stats_dir, args.board_sn, args.remote, lmar_args, (lmar_path, ms_path, cs_path, repo_used))


if __name__ == "__main__":
    main(sys.argv[1:])

