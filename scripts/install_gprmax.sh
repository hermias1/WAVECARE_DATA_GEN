#!/usr/bin/env bash
# Install gprMax with macOS ARM (Apple Silicon) patches.
# Requires: Homebrew GCC (brew install gcc)
#
# Usage: bash scripts/install_gprmax.sh [install_dir]

set -e

INSTALL_DIR="${1:-./vendor/gprMax}"
REPO_URL="https://github.com/gprMax/gprMax.git"
BRANCH="master"

echo "=== gprMax installer for macOS ARM ==="

# Check GCC
GCC_PATH=$(ls /opt/homebrew/bin/gcc-1[0-9]* 2>/dev/null | tail -1)
if [ -z "$GCC_PATH" ]; then
    echo "ERROR: Homebrew GCC not found. Install with: brew install gcc"
    exit 1
fi
GCC_NAME=$(basename "$GCC_PATH")
GCC_VER=$(echo "$GCC_NAME" | sed 's/gcc-//')
echo "Found: $GCC_NAME (version $GCC_VER)"

# Clone
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Cloning gprMax..."
    git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
else
    echo "gprMax directory exists, skipping clone"
fi

cd "$INSTALL_DIR"

# Apply macOS ARM patch to setup.py
echo "Applying macOS ARM build patch..."
python3 -c "
import re

with open('setup.py', 'r') as f:
    content = f.read()

# Check if already patched
if '_filter_flags' in content:
    print('  Already patched, skipping')
else:
    # Insert patch after imports
    patch = '''
# --- Patch for macOS system Python + Homebrew GCC ---
import sys
if sys.platform == 'darwin':
    _bad_flags = ['-iwithsysroot', '-arch x86_64',
                  '-Wno-error=cast-function-type-mismatch',
                  '-Wno-unknown-warning-option']

    def _filter_flags(flags_str):
        if not flags_str:
            return flags_str
        tokens = flags_str.split()
        filtered = []
        skip_next = False
        for tok in tokens:
            if skip_next:
                skip_next = False
                continue
            if any(tok.startswith(bf) for bf in _bad_flags):
                continue
            if tok == '-arch':
                skip_next = True
                continue
            filtered.append(tok)
        filtered.extend(['-arch', 'arm64'])
        return ' '.join(filtered)

    import distutils.sysconfig as _dsysconfig
    _orig_gcv = _dsysconfig.get_config_vars

    def _patched_gcv(*args):
        result = _orig_gcv(*args)
        if isinstance(result, dict):
            for key in ['CFLAGS', 'CCSHARED', 'LDSHARED', 'LDFLAGS',
                        'BLDSHARED', 'PY_CFLAGS', 'PY_LDFLAGS',
                        'OPT', 'BASECFLAGS', 'PY_CORE_CFLAGS']:
                if key in result and isinstance(result[key], str):
                    result[key] = _filter_flags(result[key])
        return result

    _dsysconfig.get_config_vars = _patched_gcv
# --- End patch ---
'''
    # Insert after the last import line
    insert_pos = content.find(\"import sysconfig\")
    if insert_pos == -1:
        insert_pos = content.find(\"import sys\")
    end_of_line = content.find('\n', insert_pos)
    content = content[:end_of_line+1] + patch + content[end_of_line+1:]

    # Fix GCC detection for version >= 10
    content = content.replace(
        \"glob.glob('/opt/homebrew/bin/gcc-[10-11]*')\",
        \"glob.glob('/opt/homebrew/bin/gcc-1[0-9]*')\"
    )

    # Fix library: gomp instead of iomp5
    content = content.replace(\"'iomp5'\", \"'gomp'\")

    with open('setup.py', 'w') as f:
        f.write(content)
    print('  Patch applied')
"

# Build
echo "Building gprMax with $GCC_NAME..."
python3 setup.py build

# Install
echo "Installing in editable mode..."
pip3 install --user -e .

echo ""
echo "=== gprMax installed successfully ==="
echo "Test with: python3 -c 'import gprMax; print(gprMax.__version__)'"
