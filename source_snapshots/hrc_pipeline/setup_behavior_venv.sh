#!/bin/bash
# =============================================================================
# HRC Pipeline - BEHAVIOR venv Setup (开箱即用)
#
# 严格按 docs/BEHAVIOR_VENV_INSTALL_GUIDE.md 的栈：
#   - Python 3.10 venv (不是 conda)
#   - PyTorch 2.6.0 + CUDA 12.4
#   - Isaac Sim 4.5.0.0 (全套子包，含 [all,extscache])
#   - OmniGibson 3.7.2 + bddl 3.7.0 (从 monorepo 源码 editable 装)
#   - telemoma 0.3.0
#
# 设计原则:
#   1. 幂等: 重复跑不会失败，已装的会跳过
#   2. 断点续传: 任意一步中断，重跑会从那一步继续
#   3. 后台友好: 全部输出到日志，nohup 跑也行
#   4. 不依赖网络稳定: pip cache 命中后即使断网也能装完
#
# Usage:
#   # 前台跑 (看实时输出)
#   bash setup_behavior_venv.sh
#
#   # 后台跑 (推荐, 30~60 分钟)
#   nohup bash setup_behavior_venv.sh > setup.log 2>&1 &
#
#   # 跳过某些步骤 (调试/重装时)
#   bash setup_behavior_venv.sh --skip-apt --skip-isaacsim
#
# 选项:
#   --venv-dir DIR        venv 路径 (默认 $HOME/behavior-venv)
#   --omnigibson-dir DIR  OmniGibson monorepo 路径 (默认 <repo>/OmniGibson)
#   --skip-apt            跳过 apt 系统包安装 (无 root 时)
#   --skip-pytorch        跳过 PyTorch
#   --skip-isaacsim       跳过 Isaac Sim (~10GB)
#   --skip-omnigibson     跳过 omnigibson + bddl
#   --skip-hardware       跳过硬件外设包 (默认就跳过)
#   --with-hardware       装硬件外设 (RealSense / SpaceMouse / Dynamixel / Joycon)
#   --apply-patch         应用 patches/data_wrapper.patch (HRC 数据写入需要)
#   --download-data       装完后跑 omnigibson 数据集下载 (~9GB)
# =============================================================================

set -uo pipefail

# ---- 解析参数 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HOME}/behavior-venv"
OMNIGIBSON_DIR="${SCRIPT_DIR}/OmniGibson"
SKIP_APT=false
SKIP_PYTORCH=false
SKIP_ISAACSIM=false
SKIP_OMNIGIBSON=false
WITH_HARDWARE=false
APPLY_PATCH=false
DOWNLOAD_DATA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --venv-dir)        VENV_DIR="$2"; shift 2 ;;
        --omnigibson-dir)  OMNIGIBSON_DIR="$2"; shift 2 ;;
        --skip-apt)        SKIP_APT=true; shift ;;
        --skip-pytorch)    SKIP_PYTORCH=true; shift ;;
        --skip-isaacsim)   SKIP_ISAACSIM=true; shift ;;
        --skip-omnigibson) SKIP_OMNIGIBSON=true; shift ;;
        --skip-hardware)   WITH_HARDWARE=false; shift ;;
        --with-hardware)   WITH_HARDWARE=true; shift ;;
        --apply-patch)     APPLY_PATCH=true; shift ;;
        --download-data)   DOWNLOAD_DATA=true; shift ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- 工具函数 ----
log() { echo -e "\n\033[1;36m=== [$(date '+%H:%M:%S')] $* ===\033[0m"; }
warn() { echo -e "\033[1;33m[WARN] $*\033[0m"; }
err() { echo -e "\033[1;31m[ERR ] $*\033[0m"; }
ok() { echo -e "\033[1;32m[ OK ] $*\033[0m"; }

run() {
    # 跑命令并保留 exit code 给上层判断
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        err "command failed (exit $rc): $*"
    fi
    return $rc
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || { err "$1 not found"; exit 1; }
}

# ---- 启动横幅 ----
cat <<EOF

============================================================
  HRC Pipeline - BEHAVIOR venv Setup
============================================================
  repo dir       : $SCRIPT_DIR
  venv dir       : $VENV_DIR
  OmniGibson dir : $OMNIGIBSON_DIR
  with hardware  : $WITH_HARDWARE
  apply patch    : $APPLY_PATCH
  download data  : $DOWNLOAD_DATA
============================================================
EOF

# =============================================================================
# Step 0: 检查 Python 3.10
# =============================================================================
log "Step 0: 检查 Python 3.10"
if ! command -v python3.10 >/dev/null 2>&1; then
    err "python3.10 not found. Install via: apt install python3.10 python3.10-venv python3.10-dev"
    exit 1
fi
ok "$(python3.10 --version)"

# =============================================================================
# Step 1: 系统 apt 依赖
# =============================================================================
if [ "$SKIP_APT" = false ]; then
    log "Step 1: 安装系统依赖 (apt)"
    APT_PKGS=(
        python3.10-venv python3.10-dev
        build-essential cmake git
        libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
        libxkbcommon-x11-0 libegl1 libxcb-xinerama0
        libusb-1.0-0-dev libudev-dev
        libsndfile1 portaudio19-dev
        graphviz libgraphviz-dev
        libspatialindex-dev
        ffmpeg
        vulkan-tools
        # pymeshlab 编译需要
        libgmp-dev libmpfr-dev libboost-all-dev
    )

    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            SUDO="sudo"
        else
            warn "非 root 且无 sudo，跳过 apt 安装。如缺系统库会导致 import 失败。"
            SKIP_APT=true
        fi
    fi

    if [ "$SKIP_APT" = false ]; then
        $SUDO apt-get update -qq || warn "apt-get update failed, 继续尝试安装"
        DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y "${APT_PKGS[@]}" \
            && ok "apt 包就绪" \
            || warn "部分 apt 包安装失败，继续，可能后面 pip 编译报错"
    fi
else
    log "Step 1: 跳过 apt"
fi

# =============================================================================
# Step 2: 创建 / 检查 venv
# =============================================================================
log "Step 2: venv $VENV_DIR"
if [ ! -f "$VENV_DIR/bin/python" ]; then
    python3.10 -m venv "$VENV_DIR" || { err "venv 创建失败"; exit 1; }
    ok "venv 已创建"
else
    ok "venv 已存在，复用"
fi

PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# 升级 pip / setuptools (一次就够，下次跑会 already-up-to-date)
"$PIP" install -q --upgrade 'pip>=24' 'setuptools>=68' wheel 2>&1 | tail -3 || true
ok "$("$PY" --version) | $("$PIP" --version | head -1)"

# 确认是 3.10
PY_VER=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PY_VER" != "3.10" ]; then
    err "venv Python 是 $PY_VER, 必须 3.10. 删掉 $VENV_DIR 重跑."
    exit 1
fi

# =============================================================================
# Step 3: PyTorch 2.6.0 + CUDA 12.4
# =============================================================================
if [ "$SKIP_PYTORCH" = false ]; then
    log "Step 3: PyTorch 2.6.0 + cu124"
    TORCH_VER=$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "")
    if [[ "$TORCH_VER" == "2.6.0+cu124" ]]; then
        ok "PyTorch $TORCH_VER 已就绪"
    else
        "$PIP" install \
            torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
            --index-url https://download.pytorch.org/whl/cu124 \
            && ok "PyTorch 安装完成" \
            || { err "PyTorch 安装失败"; exit 1; }
    fi

    # numpy<2 (omnigibson / isaacsim 都强依赖 numpy 1.x)
    NP_VER=$("$PY" -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo "")
    if [[ "$NP_VER" =~ ^2\. ]]; then
        warn "numpy 当前是 $NP_VER, 降到 <2"
        "$PIP" install --upgrade 'numpy<2'
    elif [ -z "$NP_VER" ]; then
        "$PIP" install 'numpy<2'
    else
        ok "numpy $NP_VER (<2) 已就绪"
    fi
else
    log "Step 3: 跳过 PyTorch"
fi

# =============================================================================
# Step 4: Isaac Sim 4.5.0.0 全套子包
# =============================================================================
if [ "$SKIP_ISAACSIM" = false ]; then
    log "Step 4: Isaac Sim 4.5.0.0 [all,extscache] (~10GB, 慢)"
    export OMNI_KIT_ACCEPT_EULA=YES

    # 用某个核心子包是否存在判断完整性 (光有 isaacsim meta 包不够)
    HAS_CORE=$("$PIP" list 2>/dev/null | grep -ci '^isaacsim-core ')
    HAS_EXTS=$("$PIP" list 2>/dev/null | grep -ci '^isaacsim-extscache-kit ')
    if [ "$HAS_CORE" -ge 1 ] && [ "$HAS_EXTS" -ge 1 ]; then
        ok "isaacsim 全套子包已就绪 (跳过)"
    else
        "$PIP" install 'isaacsim[all,extscache]==4.5.0.0' \
            --extra-index-url https://pypi.nvidia.com \
            && ok "Isaac Sim 安装完成" \
            || { err "Isaac Sim 安装失败"; exit 1; }
    fi
else
    log "Step 4: 跳过 Isaac Sim"
fi

# =============================================================================
# Step 5: OmniGibson 3.7.2 + bddl 3.7.0 (从 monorepo 源码)
# =============================================================================
if [ "$SKIP_OMNIGIBSON" = false ]; then
    log "Step 5: OmniGibson monorepo (3.7.2)"

    # 5.1 clone monorepo
    if [ ! -d "$OMNIGIBSON_DIR/.git" ]; then
        if [ -d "$OMNIGIBSON_DIR" ]; then
            warn "$OMNIGIBSON_DIR 存在但不是 git repo, 移走重 clone"
            mv "$OMNIGIBSON_DIR" "${OMNIGIBSON_DIR}.bak.$(date +%s)"
        fi
        git clone --depth 50 https://github.com/StanfordVL/OmniGibson.git "$OMNIGIBSON_DIR" \
            && (cd "$OMNIGIBSON_DIR" && git fetch --tags --depth 200) \
            || { err "git clone 失败"; exit 1; }
    fi

    # 5.2 checkout v3.7.2 (即使已经在了也无副作用)
    pushd "$OMNIGIBSON_DIR" >/dev/null
    CUR_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "")
    if [ "$CUR_TAG" != "v3.7.2" ]; then
        # 先看这个 tag 在不在本地
        if ! git rev-parse v3.7.2 >/dev/null 2>&1; then
            git fetch --tags --depth 500 origin
        fi
        git checkout v3.7.2 || { err "checkout v3.7.2 失败"; popd >/dev/null; exit 1; }
    fi
    ok "OmniGibson 已在 v3.7.2"
    popd >/dev/null

    # 5.3 校验 monorepo 结构
    if [ ! -f "$OMNIGIBSON_DIR/bddl3/setup.py" ]; then
        err "找不到 $OMNIGIBSON_DIR/bddl3/setup.py"; exit 1
    fi
    if [ ! -f "$OMNIGIBSON_DIR/OmniGibson/setup.py" ]; then
        err "找不到 $OMNIGIBSON_DIR/OmniGibson/setup.py"; exit 1
    fi

    # 5.4 装 bddl3 (必须在 omnigibson 之前)
    if "$PY" -c 'import bddl, sys; sys.exit(0 if bddl.__version__.startswith("3.7") else 1)' 2>/dev/null; then
        ok "bddl 3.7.x 已就绪"
    else
        "$PIP" install -e "$OMNIGIBSON_DIR/bddl3" \
            && ok "bddl3 已装" \
            || { err "bddl3 安装失败"; exit 1; }
    fi

    # 5.5 装 omnigibson
    if "$PY" -c 'import omnigibson, sys; sys.exit(0 if omnigibson.__version__=="3.7.2" else 1)' 2>/dev/null; then
        ok "omnigibson 3.7.2 已就绪"
    else
        "$PIP" install -e "$OMNIGIBSON_DIR/OmniGibson" --no-build-isolation \
            && ok "omnigibson 已装" \
            || { err "omnigibson 安装失败 (常见原因: pymeshlab 编译缺 libgmp/libmpfr/libboost)"; exit 1; }
    fi

    # 5.6 telemoma (guide 要求, 可选)
    if ! "$PY" -c 'import telemoma' 2>/dev/null; then
        "$PIP" install telemoma==0.3.0 || warn "telemoma 安装失败 (非必须)"
    fi

    # 5.7 post-install 修复 (Isaac Sim 自带的 cffi/websockets 会跟环境冲突)
    log "Step 5.7: post-install 修复"
    "$PIP" install -q --force-reinstall cffi==1.17.1 2>&1 | tail -3 || true
    "$PIP" install -q --force-reinstall 'websockets>=15.0.1' 2>&1 | tail -3 || true

    # 5.8 再次确保 numpy<2 (omnigibson 装的过程可能引入 numpy 2.x 依赖)
    NP_VER=$("$PY" -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo "")
    if [[ "$NP_VER" =~ ^2\. ]]; then
        warn "numpy 又变成 $NP_VER, 强制降到 <2"
        "$PIP" install --force-reinstall 'numpy<2'
    fi
else
    log "Step 5: 跳过 OmniGibson"
fi

# =============================================================================
# Step 6: 硬件外设 (可选)
# =============================================================================
if [ "$WITH_HARDWARE" = true ]; then
    log "Step 6: 硬件外设包"
    "$PIP" install \
        pyrealsense2==2.56.5.9235 \
        pyspacemouse==2.0.0 \
        dynamixel-sdk==4.0.3 \
        joycon-python==0.2.4 \
        || warn "硬件包部分失败 (没硬件本来也用不到)"
else
    log "Step 6: 跳过硬件外设 (用 --with-hardware 启用)"
fi

# =============================================================================
# Step 7: 应用 patches/data_wrapper.patch (HRC HDF5 写入需要)
# =============================================================================
if [ "$APPLY_PATCH" = true ] && [ -d "$OMNIGIBSON_DIR" ]; then
    log "Step 7: 应用 data_wrapper patch"
    PATCH_TARGET="$OMNIGIBSON_DIR/OmniGibson/omnigibson/envs/data_wrapper.py"
    if [ ! -f "$PATCH_TARGET" ]; then
        warn "$PATCH_TARGET 不存在, 跳过"
    else
        # 直接 sed 改两处，比 git apply 稳 (上游行号会变)
        if grep -q 'h5py.File(input_path, "r+")' "$PATCH_TARGET" 2>/dev/null; then
            ok "patch 已生效 (r+ 已存在)"
        else
            sed -i 's/h5py\.File(input_path, "r")/h5py.File(input_path, "r+")/g' "$PATCH_TARGET"
            COUNT=$(grep -c 'h5py.File(input_path, "r+")' "$PATCH_TARGET")
            if [ "$COUNT" -ge 1 ]; then
                ok "patch 应用成功 ($COUNT 处)"
            else
                warn "patch 似乎没生效, 请手动检查 $PATCH_TARGET"
            fi
        fi
    fi
fi

# =============================================================================
# Step 8: 数据集下载 (可选, ~9GB)
# =============================================================================
if [ "$DOWNLOAD_DATA" = true ]; then
    log "Step 8: 下载 BEHAVIOR-1K 数据集 (~9GB)"
    export OMNI_KIT_ACCEPT_EULA=YES
    "$PY" - <<'PY' || warn "数据集下载失败, 可稍后手动重跑"
from omnigibson.utils.asset_utils import (
    download_omnigibson_robot_assets,
    download_behavior_1k_assets,
    download_2025_challenge_task_instances,
)
print("[1/3] robot assets...")
download_omnigibson_robot_assets()
print("[2/3] behavior-1k assets...")
download_behavior_1k_assets(accept_license=True)
print("[3/3] 2025 challenge task instances...")
download_2025_challenge_task_instances()
print("done")
PY
fi

# =============================================================================
# Step 9: 验证
# =============================================================================
log "Step 9: 最终验证"
export OMNI_KIT_ACCEPT_EULA=YES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# 防止 AutoDL 镜像默认 OMP_NUM_THREADS=0 触发 libgomp warn
[ "$OMP_NUM_THREADS" = "0" ] && export OMP_NUM_THREADS=1
"$PY" - <<'PY'
import sys, importlib
import importlib.metadata as md
print(f"  python : {sys.version.split()[0]}")
# 包名 -> pip 包名 (有的 import 名和 pip 名不一样)
PIP_NAME = {"cv2": "opencv-python"}
results = []
def chk(name):
    try:
        importlib.import_module(name)
    except Exception as e:
        print(f"  [FAIL] {name:14s} {type(e).__name__}: {e}")
        results.append(False)
        return
    try:
        v = md.version(PIP_NAME.get(name, name))
    except Exception:
        v = "?"
    print(f"  [ OK ] {name:14s} {v}")
    results.append(True)
for pkg in ["torch", "numpy", "isaacsim", "bddl", "omnigibson",
            "cv2", "h5py", "imageio", "telemoma"]:
    chk(pkg)
import torch
print(f"  cuda available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda devices   : {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
sys.exit(0 if all(results[:5]) else 1)  # 前 5 个 (torch/numpy/isaacsim/bddl/omnigibson) 必须 OK
PY
RC=$?

cat <<EOF

============================================================
  $([ $RC -eq 0 ] && echo "Setup Complete!" || echo "Setup FAILED.")
============================================================

  激活环境:
    source $VENV_DIR/bin/activate
    export OMNI_KIT_ACCEPT_EULA=YES

  运行 HRC:
    bash $SCRIPT_DIR/cmd.sh

  下次重跑此脚本是安全的 (幂等), 已装的会跳过.
============================================================
EOF

exit $RC
