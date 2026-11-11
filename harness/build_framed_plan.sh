#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build="${FRAMED_PLAN_BUILD_DIR:-$repo/build/frozen-plan}"
alp_root="${ALP_ROOT:-}"
alp_library="${ALP_LIBRARY:-}"
streaming_root="${STREAMING_HEADERS_ROOT:-}"
command -v cmake >/dev/null || { echo "Missing cmake >=3.22" >&2; exit 2; }
if [[ -z "${CXX:-}" ]] && command -v g++-13 >/dev/null; then
  export CXX="$(command -v g++-13)"
fi
[[ -n "$alp_root" && -f "$alp_root/include/alp.hpp" ]] || {
  echo "Set ALP_ROOT to the external ALP source directory" >&2; exit 2;
}
[[ -n "$alp_library" && -f "$alp_library" ]] || {
  echo "Missing existing ALP library. Set ALP_LIBRARY=/absolute/path/to/libALP.so" >&2
  exit 2
}
[[ -n "$streaming_root" && -f "$streaming_root/Gorilla/CompressorGorilla.hpp" \
   && -f "$streaming_root/lib/BitStream.hpp" ]] || {
  echo "Set STREAMING_HEADERS_ROOT to the external NeaTS benchmark header directory" >&2
  exit 2
}
echo "Frozen-plan replay uses ALP library: $alp_library"
cmake -S "$repo/harness/framed" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DALP_ROOT="$alp_root" -DALP_LIBRARY="$alp_library" \
  -DSTREAMING_HEADERS_ROOT="$streaming_root"
cmake --build "$build" --target FrozenPlanFile -j "${BUILD_JOBS:-1}"
echo "Frozen-plan serializer: $build/FrozenPlanFile"
