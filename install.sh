#!/usr/bin/env bash
# HybAligner — One-command installer for Linux
# Usage: curl -fsSL https://.../install.sh | bash
#    or: bash install.sh

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

INSTALL_DIR="${HOME}/.local/share/hybaligner"
BIN_DIR="${HOME}/.local/bin"

echo "╔══════════════════════════════════════════╗"
echo "║   HybAligner Installer v0.5.0           ║"
echo "║   GPU Sequence Aligner for DGX Spark    ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# --- Check prerequisites ---
echo "Checking prerequisites..."

command -v nvcc >/dev/null 2>&1 || error "CUDA toolkit not found. Install: sudo apt install nvidia-cuda-toolkit"
CUDA_VER=$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')
info "CUDA $CUDA_VER found"

command -v cmake >/dev/null 2>&1 || error "cmake not found. Install: sudo apt install cmake"
info "cmake $(cmake --version | head -1 | awk '{print $3}') found"

command -v gcc >/dev/null 2>&1 || error "gcc not found. Install: sudo apt install build-essential"
info "gcc $(gcc --version | head -1 | awk '{print $4}') found"

command -v python3 >/dev/null 2>&1 || error "python3 not found"
info "python3 $(python3 --version | awk '{print $2}') found"

command -v nvidia-smi >/dev/null 2>&1 || warn "nvidia-smi not found — GPU may not be available"
[ -n "$(command -v nvidia-smi)" ] && info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# --- Create directories ---
mkdir -p "$INSTALL_DIR" "$BIN_DIR"

# --- Build CUDA kernels ---
echo ""
echo "Building CUDA kernels..."
cd "$INSTALL_DIR"
mkdir -p build && cd build
cmake .. 2>/dev/null || cmake "$SCRIPT_DIR/.." 2>/dev/null || {
    warn "Cloning HybAligner..."
    cd /tmp
    rm -rf hybaligner-src
    git clone --depth 1 https://github.com/your-org/hybaligner.git hybaligner-src 2>/dev/null || {
        # Fallback: copy from current directory
        cp -r "$(dirname "$0")" hybaligner-src 2>/dev/null || error "Cannot find HybAligner source"
    }
    cd hybaligner-src
    mkdir -p build && cd build
    cmake ..
}
make -j$(nproc)
info "CUDA kernels built: $(du -sh libcuda_kernels.so 2>/dev/null | cut -f1)"

# --- Install Python deps ---
echo ""
echo "Installing Python dependencies..."
pip install numpy psutil tqdm 2>/dev/null | tail -1
info "Python packages installed"

# --- Create wrapper ---
WRAPPER="$BIN_DIR/hyb-align"
cat > "$WRAPPER" << 'WRAPPER'
#!/usr/bin/env bash
# HybAligner launcher
HYB_HOME="${HOME}/.local/share/hybaligner"
export PYTHONPATH="${HYB_HOME}:${PYTHONPATH}"
exec python3 -m hyb_align "$@"
WRAPPER
chmod +x "$WRAPPER"

# --- Update PATH hint ---
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo ""
    warn "Add to your shell config (~/.bashrc):"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Installation Complete!                 ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Run: hyb-align reads.fastq ref.fasta -o results/"
echo ""
