.PHONY: all build install clean test bench

PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin
LIBDIR ?= $(PREFIX)/lib/hybaligner

all: build

# Build CUDA shared library
build:
	@mkdir -p build && cd build && cmake .. && make -j$$(nproc)
	@echo "CUDA kernels built: build/libcuda_kernels.so"

# Install everything
install: build
	@mkdir -p $(BINDIR) $(LIBDIR)
	@cp build/libcuda_kernels.so $(LIBDIR)/
	@cp -r gpu cpu runtime obs $(LIBDIR)/
	@cp hyb_align.py $(LIBDIR)/
	@printf '#!/usr/bin/env bash\nexport PYTHONPATH="%s:$$PYTHONPATH"\nexec python3 -m hyb_align "$$@"\n' "$(LIBDIR)" > $(BINDIR)/hyb-align
	@chmod +x $(BINDIR)/hyb-align
	@echo "Installed: $(BINDIR)/hyb-align"

# Uninstall
uninstall:
	@rm -f $(BINDIR)/hyb-align
	@rm -rf $(LIBDIR)
	@echo "Uninstalled"

# Run tests
test:
	python -m pytest tests/ -v

# Interactive TUI
tui:
	python hyb_align_tui.py

# Benchmark
bench:
	python benchmark/bench.py -n 5000 -l 150 -r 20000

# Clean build artifacts
clean:
	@rm -rf build .pytest_cache
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned"
