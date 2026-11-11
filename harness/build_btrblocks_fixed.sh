#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${BTRBLOCKS_ROOT:-}"
build_dir="${BTRBLOCKS_FIXED_BUILD_DIR:-}"
python="${PYTHON_EXE:-python3}"
[[ "$(uname -s)" == Linux ]] || { echo "Run this native build on Linux." >&2; exit 2; }
[[ "$source_dir" = /* && "$build_dir" = /* ]] || {
  echo "Set absolute BTRBLOCKS_ROOT and BTRBLOCKS_FIXED_BUILD_DIR paths." >&2
  exit 2
}
[[ -f "$source_dir/btrblocks/btrblocks.hpp" ]] || { echo "Missing BTRBLOCKS_ROOT: $source_dir" >&2; exit 2; }
cmake_options=()
if [[ -n "${BTRBLOCKS_OFFLINE_ARCHIVE_DIR:-}" ]]; then
  "$python" "$repo/tools/prepare_btrblocks_offline.py" \
    --source "$source_dir" --build-dir "$build_dir/offline" \
    --archives-dir "$BTRBLOCKS_OFFLINE_ARCHIVE_DIR" \
    --cache-dir "${BTRBLOCKS_VENDOR_CACHE:-$build_dir/vendor_cache}"
  source_dir="$build_dir/offline/offline_source"
  cmake_options=(-C "$build_dir/offline/offline_paths.cmake")
fi
"$python" "$repo/tools/prepare_btrblocks_fixed.py" \
  --source "$source_dir" --build-dir "$build_dir/native"
cmake "${cmake_options[@]}" -S "$build_dir/native/fixed_source" -B "$build_dir/native" \
  -DCMAKE_BUILD_TYPE=Release -DWITH_LOGGING=OFF -DSAMPLING_TEST_MODE=OFF -DENABLE_FOR_SCHEME=OFF \
  -DCMAKE_PROJECT_BtrBlocks_INCLUDE="$repo/harness/external_baselines/btrblocks_fixed/CMakeLists.txt"
cmake --build "$build_dir/native" --target BtrBlocksFixed -j "${BUILD_JOBS:-1}"
ldd "$build_dir/native/BtrBlocksFixed" > "$build_dir/native/runtime_libraries.txt"
if grep -q 'not found' "$build_dir/native/runtime_libraries.txt"; then
  cat "$build_dir/native/runtime_libraries.txt" >&2
  exit 2
fi
"$build_dir/native/BtrBlocksFixed" --self-test
printf 'Fixed primitive adapter: %s\n' "$build_dir/native/BtrBlocksFixed"
