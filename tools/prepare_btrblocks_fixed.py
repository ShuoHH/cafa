"""Create a checked, private BtrBlocks build copy with no adaptive scheme picker.

The original checkout and complete-BtrBlocks executable are never modified.
This is a fixed-composition adapter, not an unmodified BtrBlocks baseline.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from prepare_btrblocks_offline import source_files, tree_hashes


SCHEMA = "btrblocks-fixed-primitives-v1"
PICKER = "btrblocks/compression/SchemePicker.hpp"
DECIMAL = "btrblocks/scheme/double/Pseudodecimal.cpp"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Unexpected upstream source; refusing blind patch: " + old[:90])
    return text.replace(old, new, 1)


def fixed_picker(text):
    # Replace BOTH complete method bodies, not just the SAMPLE branch. In the
    # native picker, even force_scheme may be bypassed by ONE_VALUE/FOR or an
    # expansion fallback, so setting override_scheme is not sufficient.
    if "#define BTR_FIXED_POLICY_V1" in text:
        raise ValueError("Input is already a fixed-policy overlay, not an original source")
    if text.count("  static SchemeType& chooseScheme(") != 1 or text.count("  static void compress(") != 1:
        raise ValueError("Unexpected upstream picker declarations")
    for expected in ("SchemeSelection::SAMPLE", "SchemeSelection::TRY_ALL",
                     "using MyTypeWrapper = TypeWrapper<SchemeType, SchemeCodeType>;"):
        if expected not in text:
            raise ValueError("Unexpected upstream picker structure: " + expected)
    start = text.index("  static SchemeType& chooseScheme(")
    middle = text.index("  static void compress(", start)
    end = text.index("\n};", middle)
    replacement = '''  static SchemeType& chooseScheme(StatsType&, u8) {
    throw std::logic_error("adaptive selection disabled in fixed primitive build");
  }
  static void compress(const Type* src, const BITMAP* nullmap, u8* dest,
                       u32 tuple_count, u8 allowed_cascading_level,
                       u32& after_size, u8& scheme_code,
                       u8 force_scheme = autoScheme(), const string& = "?") {
    // Fixed substreams: integer BP, raw binary64. No estimation, sampling,
    // alternative encoder trial, ONE_VALUE special case or expansion fallback.
    u8 fixed_code = CB(SchemeCodeType::UNCOMPRESSED);
    if constexpr (std::is_same_v<Type, INTEGER>)
      fixed_code = CB(IntegerSchemeType::BP);
    if (force_scheme != autoScheme() && force_scheme != fixed_code)
      throw std::logic_error("substream requested a different fixed encoder");
    auto& schemes = MyTypeWrapper::getSchemes();
    const auto found = schemes.find(static_cast<SchemeCodeType>(fixed_code));
    if (found == schemes.end() || !found->second)
      throw std::logic_error("required fixed encoder is not registered");
    scheme_code = fixed_code;
    if (tuple_count == 0) { after_size = 0; return; }
    auto stats = StatsType::generateStats(src, nullmap, tuple_count);
    after_size = found->second->compress(src, nullmap, dest, stats,
                                         allowed_cascading_level);
  }'''
    result = text[:start] + replacement + text[end:]
    result = replace_once(result, "#pragma once", '''#pragma once
#define BTR_FIXED_POLICY_V1 1
#include <stdexcept>
#include <type_traits>
#if defined(BTR_FLAG_SAMPLING_TEST_MODE) && BTR_FLAG_SAMPLING_TEST_MODE
#error "sampling-test mode is forbidden in the fixed primitive build"
#endif
#if defined(BTR_FLAG_ENABLE_FOR_SCHEME) && BTR_FLAG_ENABLE_FOR_SCHEME
#error "adaptive FOR insertion is forbidden in the fixed primitive build"
#endif''')
    for forbidden in ("SchemeSelection::SAMPLE", "SchemeSelection::TRY_ALL", "ThreadCache::get().compression_level++", "chooseScheme(stats"):
        if forbidden in result:
            raise ValueError("Adaptive picker path survived: " + forbidden)
    return result


def guarded_decimal(text):
    # Upstream casts rounded doubles to u64 before checking representability.
    # Guard the fixed build against NaN/Inf, negatives and out-of-range casts.
    # All conversions the native encoder can actually store (31 bits) remain
    # byte-identical. Other values continue through its existing patch path.
    return replace_once(text, "          converted_number = static_cast<u64>(cd);", '''          if (!std::isfinite(cd) || cd < 0.0 || cd > 2147483647.0) {
            continue; // private fixed build: do not execute undefined casts
          }
          converted_number = static_cast<u64>(cd);''')


def prepare(source, build):
    source, build = Path(source).resolve(), Path(build).resolve()
    if source == build or source in build.parents or build in source.parents:
        raise ValueError("Private build and source must be disjoint")
    original = source_files(source)
    for name in (PICKER, DECIMAL):
        if name not in original:
            raise ValueError("Required native source missing: " + name)
    replacements = {
        PICKER: fixed_picker((source / PICKER).read_text(encoding="utf-8")),
        DECIMAL: guarded_decimal((source / DECIMAL).read_text(encoding="utf-8")),
    }
    signature = dict(schema=SCHEMA, source=str(source), source_sha256=original,
                     replacements_sha256={k: hashlib.sha256(v.encode()).hexdigest()
                                          for k, v in replacements.items()})
    target, manifest = build / "fixed_source", build / "fixed_overlay_manifest.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("inputs") != signature or previous.get("private_source_sha256") != tree_hashes(target):
            raise ValueError("Existing fixed build changed; use a new build directory")
        print("Verified unchanged fixed primitive build copy")
        return target
    if target.exists() or (build / "CMakeCache.txt").exists():
        raise ValueError("Refusing existing/partial private build; use a new directory")
    build.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    for name in original:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    for name, text in replacements.items():
        (target / name).write_text(text, encoding="utf-8", newline="\n")
    if source_files(source) != original:
        raise ValueError("Original source changed concurrently")
    private_hashes = tree_hashes(target)
    changed = [name for name, value in original.items() if private_hashes.get(name) != value]
    if set(changed) != {PICKER, DECIMAL}:
        raise ValueError("Unexpected private source differences: " + repr(changed))
    report = dict(inputs=signature, private_source_sha256=private_hashes,
                  original_sources_unchanged=True, fixed_substreams={
                      "dictionary_codes": "BP", "rle_values": "UNCOMPRESSED",
                      "rle_counts": "BP", "frequency_exceptions": "UNCOMPRESSED",
                      "decimal_numbers": "BP", "decimal_exponents": "BP",
                      "decimal_patches": "UNCOMPRESSED"},
                  native_changes=["deterministic picker entry; adaptive choice disabled",
                                  "decimal defined-domain guard before floating-to-u64 cast"],
                  adaptive_selection=False, codec_expansion_fallback=False)
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Prepared checked fixed primitive source: " + str(target))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--build-dir", required=True)
    args = parser.parse_args()
    prepare(args.source, args.build_dir)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print("BtrBlocks fixed preparation: " + str(error), file=sys.stderr)
        sys.exit(2)
