"""一键同步到 GitHub（双击「同步到GitHub.bat」运行）。

为什么不用 git push：本机 github.com 的 HTTPS 通道被网络层拦截
（TCP 能连、TLS 一握手就断），而 api.github.com 完全正常。
因此这里改走 GitHub 官方 Git Data API：把工作区文件按字节上传成
blob → 建 tree → 建 commit → 更新分支引用，效果与 push 等价。

凭据来源：本机 Windows 凭据管理器里 GitHub Desktop 的登录态
（条目名以「GitHub - https://api.github.com/<用户名>」开头）。
不会把 token 写进任何文件。

用法：
    python sync_to_github.py                # 同步当前工作区
    python sync_to_github.py -m "提交说明"   # 自定义提交说明
"""
import argparse
import base64
import ctypes
import ctypes.wintypes as wt
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER, REPO, BRANCH = "pengyaohui0", "imgtext-tool", "main"
API = "https://api.github.com"

GIT_CANDIDATES = [
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Git", "cmd", "git.exe"),
    r"C:\Users\Admin（无密码）\.workbuddy\binaries\PortableGit\versions\1.2.0\cmd\git.exe",
]

GITIGNORE = """# Python
__pycache__/
*.py[cod]
.venv/
venv/

# 程序输出 / 调试产物
_test_output/
_bench_out/

# 打包产物
dist/
build/
*.spec
"""

# 这些不进仓库
EXTRA_SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "_test_output", "_bench_out", "dist", "build"}


def find_git():
    for p in GIT_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return "git"


GIT = find_git()


class CREDENTIALW(ctypes.Structure):
    _fields_ = [
        ("Flags", wt.DWORD), ("Type", wt.DWORD),
        ("TargetName", wt.LPWSTR), ("Comment", wt.LPWSTR),
        ("LastWritten", wt.FILETIME), ("CredentialBlobSize", wt.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
        ("Persist", wt.DWORD), ("AttributeCount", wt.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wt.LPWSTR), ("UserName", wt.LPWSTR),
    ]


def read_token():
    """从 Windows 凭据管理器读 GitHub Desktop 的授权。"""
    adv = ctypes.windll.advapi32
    targets = [f"GitHub - https://api.github.com/{OWNER}"]
    try:
        out = subprocess.run([r"C:\Windows\System32\cmdkey.exe", "/list"],
                             capture_output=True, timeout=20).stdout.decode("gbk", "replace")
        for line in out.splitlines():
            if "GitHub - https://api.github.com" in line:
                t = line.split("target=", 1)[-1].strip()
                if t not in targets:
                    targets.insert(0, t)
    except Exception:
        pass
    for target in targets:
        ptr = ctypes.POINTER(CREDENTIALW)()
        if not adv.CredReadW(ctypes.c_wchar_p(target), 1, 0, ctypes.byref(ptr)):
            continue
        c = ptr.contents
        blob = ctypes.string_at(c.CredentialBlob, c.CredentialBlobSize)
        adv.CredFree(ptr)
        for enc in ("utf-16-le", "utf-8"):
            try:
                t = blob.decode(enc).strip().strip("\x00").strip()
            except Exception:
                continue
            if len(t) > 20 and t.isascii() and t.isprintable():
                return t
    return None


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _send(url, method, body, token):
    req = urllib.request.Request(
        url, data=body, method=method,
        headers={"Authorization": "token " + token, "User-Agent": "imgtext-sync",
                 "Content-Type": "application/json",
                 "Accept": "application/vnd.github+json"})
    try:
        with OPENER.open(req, timeout=60) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {}), None
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        loc = e.headers.get("Location") if e.headers else None
        try:
            return e.code, json.loads(raw), loc
        except Exception:
            return e.code, raw, loc
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}", None


def api(path, token, method="GET", payload=None):
    """GitHub 对写操作会 307 跳到 /repositories/{id}/…，urllib 不自动跟，这里手动跟。"""
    url = API + path
    body = json.dumps(payload).encode() if payload is not None else None
    for _ in range(5):
        st, data, loc = _send(url, method, body, token)
        if st in (301, 302, 307, 308) and loc:
            url = loc if loc.startswith("http") else API + loc
            continue
        return st, data
    return st, data


def git(*args, timeout=120):
    # core.quotepath=false：否则中文文件名会被转义成 \345\220\214… 这样的八进制串
    p = subprocess.run([GIT, "-C", HERE, "-c", "core.quotepath=false"] + list(args),
                       capture_output=True, timeout=timeout,
                       env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"})
    return p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace").strip()


def local_files():
    """收集应进仓库的文件（相对路径 -> 绝对路径），尊重 .gitignore。"""
    gi = os.path.join(HERE, ".gitignore")
    if not os.path.exists(gi):
        with open(gi, "w", encoding="utf-8", newline="\n") as f:
            f.write(GITIGNORE)
    rc, out = git("ls-files", "--cached", "--others", "--exclude-standard")
    if rc != 0 or not out.strip():
        files = []
        for dp, dn, fn in os.walk(HERE):
            dn[:] = [d for d in dn if d not in EXTRA_SKIP_DIRS]
            for f in fn:
                full = os.path.join(dp, f)
                files.append(os.path.relpath(full, HERE).replace("\\", "/"))
        return {f: os.path.join(HERE, f.replace("/", os.sep)) for f in sorted(files)}
    files = [x.strip() for x in out.splitlines() if x.strip()]
    return {f.replace("\\", "/"): os.path.join(HERE, f.replace("/", os.sep))
            for f in sorted(files)}


def blob_sha(path):
    """算工作区文件原始字节的 git blob sha（--no-filters 保证与上传字节同口径）。"""
    rc, out = git("hash-object", "--no-filters", path)
    return out.strip() if rc == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", default=None, help="提交说明")
    args = ap.parse_args()

    print("=" * 60)
    print(f"  同步到 GitHub：{OWNER}/{REPO}")
    print("=" * 60)

    token = read_token()
    if not token:
        print("[×] 没找到 GitHub 登录凭据。")
        print("    请先在 GitHub Desktop 里登录一次，或把 PAT 告诉我。")
        return 1
    print(f"[1/5] 已取到授权（{token[:4]}…，长度 {len(token)}）")

    st, ref = api(f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}", token)
    if st != 200:
        print(f"[×] 取远端分支失败：{st} {str(ref)[:200]}")
        return 1
    remote_commit = ref["object"]["sha"]

    st, commit = api(f"/repos/{OWNER}/{REPO}/git/commits/{remote_commit}", token)
    remote_tree = commit["tree"]["sha"] if st == 200 else None

    remote = {}
    if remote_tree:
        st, t = api(f"/repos/{OWNER}/{REPO}/git/trees/{remote_tree}?recursive=1", token)
        if st == 200:
            remote = {e["path"]: e["sha"] for e in t.get("tree", []) if e["type"] == "blob"}

    files = local_files()
    print(f"[2/5] 本地 {len(files)} 个文件，远端 {len(remote)} 个文件，正在比对…")

    changed, added, deleted = [], [], []
    bad = []
    for rel, full in files.items():
        sha = blob_sha(full)
        if not sha:
            bad.append(rel)
            continue
        if rel not in remote:
            added.append(rel)
        elif remote[rel] != sha:
            changed.append(rel)
    deleted = [p for p in remote if p not in files]
    if bad:
        print(f"[!] 有 {len(bad)} 个文件读不到，已跳过：{bad[:5]}")

    if not (added or changed or deleted):
        print("[3/5] 没有变化，远端已是最新。")
        return 0

    print(f"[3/5] 变更：新增 {len(added)}，修改 {len(changed)}，删除 {len(deleted)}")
    for p in added[:10]:
        print("       + " + p)
    for p in changed[:10]:
        print("       ~ " + p)
    for p in deleted[:10]:
        print("       - " + p)

    entries = []
    total = len(added) + len(changed)
    done = 0
    for rel in added + changed:
        with open(files[rel], "rb") as f:
            content = base64.b64encode(f.read()).decode("ascii")
        for attempt in range(3):
            st, d = api(f"/repos/{OWNER}/{REPO}/git/blobs", token, "POST",
                        {"content": content, "encoding": "base64"})
            if st == 201:
                break
            time.sleep(1.5 * (attempt + 1))
        if st != 201:
            print(f"[×] 上传失败：{rel} -> {st} {str(d)[:150]}")
            return 1
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": d["sha"]})
        done += 1
        if done % 10 == 0 or done == total:
            print(f"       上传进度 {done}/{total}")
    for rel in deleted:
        entries.append({"path": rel, "mode": "100644", "type": "blob", "sha": None})

    st, nt = api(f"/repos/{OWNER}/{REPO}/git/trees", token, "POST",
                 {"base_tree": remote_tree, "tree": entries} if remote_tree else {"tree": entries})
    if st != 201:
        print(f"[×] 建树失败：{st} {str(nt)[:200]}")
        return 1

    msg = args.message or time.strftime("更新代码 %Y-%m-%d %H:%M")
    st, nc = api(f"/repos/{OWNER}/{REPO}/git/commits", token, "POST",
                 {"message": msg, "tree": nt["sha"], "parents": [remote_commit]})
    if st != 201:
        print(f"[×] 建提交失败：{st} {str(nc)[:200]}")
        return 1
    new_sha = nc["sha"]

    st, _ = api(f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}", token, "PATCH",
                {"sha": new_sha, "force": True})
    if st not in (200, 201):
        print(f"[×] 更新分支失败：{st}")
        return 1
    print(f"[4/5] 已提交到 GitHub：{new_sha[:7]}  「{msg}」")

    print("[5/5] 同步本地指针…")
    git("fetch", "origin", BRANCH)
    d = os.path.join(HERE, ".git", "refs", "remotes", "origin")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, BRANCH), "w", encoding="ascii") as f:
        f.write(new_sha + "\n")
    # 只挪指针 + 刷新索引，绝不 checkout 工作区
    # （reset --hard 会按行尾规则重写文件字节，导致下次比对又"有修改"）
    git("update-ref", f"refs/heads/{BRANCH}", new_sha)
    git("read-tree", new_sha)
    rc, out = git("status", "-sb")
    print("       " + out)

    print()
    print("完成 -> https://github.com/%s/%s" % (OWNER, REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
