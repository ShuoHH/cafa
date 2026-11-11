"""Prepare an offline *build copy*, never patch downloaded baseline sources.

Use the two pinned GitHub ZIPs and the already configured Boost/CRoaring cache.
Only transport and unused test/executable targets change. Codec source files,
library definitions and optimization flags are preserved byte for byte.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import zipfile


PINS = {
    "fastpfor": ("FastPFOR-bc375056b803af9b3ca5a6c98e6b38aeca80d825",
                 "91e8be377b408f6d5345eb6180b02958a69a6e46b6f7ba132c83e42520483bde"),
    "fsst": ("fsst-0f0f9057048412da1ee48e35d516155cb7edd155",
             "c12a81d7169c22d2f8300955cf7c8860279845c4b5c73c92e0903c2b81654b2b"),
}
BOOST_PIN = "b6928ae5c92e21a04bbe17a558e6e066dbe632f6"
ROARING_PIN = "b88b002407b42fafaea23ea5009a54a24d1c1ed4"
BOOST_LIBS = ("dynamic_bitset", "assert", "config", "container_hash", "describe",
              "mp11", "type_traits", "core", "integer", "move", "static_assert",
              "throw_exception")
SOURCE_DIRS = ("btrblocks", "btrfiles", "cmake", "test", "bench", "tools")
SCHEMA = "btrblocks-offline-build-v1"


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def require_file(path):
    if not path.is_file():
        raise ValueError("Required file missing: " + str(path))
    return path


def revision(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True,
                            env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    return result.stdout.strip()


def verify_checkout(path, submodule_paths=()):
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    subprocess.run(["git", "-C", str(path), "diff", "--quiet", "HEAD", "--"],
                   check=True, capture_output=True, env=environment)
    if submodule_paths:
        result = subprocess.run(["git", "-C", str(path), "submodule", "status", "--recursive", "--"]
                                + list(submodule_paths), check=True, capture_output=True,
                                text=True, env=environment)
        lines = result.stdout.splitlines()
        if not lines or any(not line.startswith(" ") for line in lines):
            raise ValueError("Boost submodules are missing or differ from their pinned commits")


def check_cache(cache, get_revision=revision):
    boost = cache / "vendor/boost/boost_src-src"
    require_file(boost / "CMakeLists.txt")
    require_file(boost / "tools/cmake/include/BoostRoot.cmake")
    for library in BOOST_LIBS:
        require_file(boost / "libs" / library / "CMakeLists.txt")
        if not (boost / "libs" / library / "include").is_dir():
            raise ValueError("Incomplete Boost submodule: " + library)
    if get_revision(boost) != BOOST_PIN:
        raise ValueError("Cached Boost revision differs from BtrBlocks' pinned revision")
    verify_checkout(boost, ["libs/" + name for name in BOOST_LIBS] + ["tools/cmake"])
    roaring = cache / "vendor/croaring"
    require_file(roaring / "include/roaring/roaring.hh")
    require_file(roaring / "include/roaring/roaring.h")
    library = require_file(roaring / "lib/libroaring.so")
    try:
        library.resolve().relative_to(roaring.resolve())
    except ValueError:
        raise ValueError("CRoaring library symlink points outside its original cache")
    if get_revision(roaring / "src/croaring_src") != ROARING_PIN:
        raise ValueError("Cached CRoaring revision differs from BtrBlocks' pinned revision")
    verify_checkout(roaring / "src/croaring_src")
    return boost, roaring, library


def archive_path(folder, name):
    # GitHub canonical archive spelling; Windows is case insensitive but Linux isn't.
    matches = [p for p in folder.iterdir() if p.is_file()
               and p.name.lower() == (name + ".zip").lower()]
    if len(matches) != 1:
        raise ValueError("Need exactly one " + name + ".zip in " + str(folder))
    return matches[0]


def extract_zip(archive, destination, expected_root):
    if destination.exists():
        raise ValueError("Refusing to overwrite extracted dependency: " + str(destination))
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) > 20000 or sum(p.file_size for p in entries) > 512 * 1024 * 1024:
            raise ValueError("Dependency archive exceeds extraction limits")
        seen = set()
        for entry in entries:
            path = PurePosixPath(entry.filename)
            mode = (entry.external_attr >> 16) & 0xFFFF
            if (path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                    or not path.parts or path.parts[0] != expected_root
                    or stat.S_ISLNK(mode) or entry.filename in seen
                    or any(":" in p for p in path.parts)):
                raise ValueError("Unsafe dependency archive member: " + entry.filename)
            seen.add(entry.filename)
        destination.mkdir(parents=True)
        for entry in entries:
            relative = PurePosixPath(entry.filename).parts[1:]
            if not relative:
                continue
            target = destination.joinpath(*relative)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def read_text(path):
    return path.read_text(encoding="utf-8")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Unexpected upstream CMake recipe; refusing fuzzy patch: " + old)
    return text.replace(old, new, 1)


def patch_build_copy(source, fastpfor):
    # The stock pinned library downloads floating GoogleTest main unconditionally.
    # Wrap only executable/test declarations, preserving the FastPFOR target/flags.
    recipe = fastpfor / "CMakeLists.txt"
    contents = replace_once(read_text(recipe), "# other executables",
                            "if(NOT BTRBLOCKS_LIBRARY_ONLY)\n# other executables")
    contents = replace_once(contents, "include(GNUInstallDirs)",
                            "endif() # BTRBLOCKS_LIBRARY_ONLY\n\ninclude(GNUInstallDirs)")
    recipe.write_text(contents, encoding="utf-8", newline="\n")
    for name, repo, pin, variable, target in [
        ("fastpfor", "lemire/FastPFor", PINS["fastpfor"][0].split("-", 1)[1],
         "BTR_LOCAL_FASTPFOR_SOURCE", "FastPFOR"),
        ("fsst", "cwida/fsst", PINS["fsst"][0].split("-", 1)[1],
         "BTR_LOCAL_FSST_SOURCE", "fsst"),
    ]:
        recipe = source / "cmake" / (name + ".cmake")
        contents = read_text(recipe)
        contents = replace_once(contents, 'GIT_REPOSITORY "https://github.com/' + repo + '.git"',
                                'SOURCE_DIR "${' + variable + '}"\n    DOWNLOAD_COMMAND ""')
        contents = replace_once(contents, "GIT_TAG " + pin, "# Pinned local archive: " + pin)
        contents = replace_once(contents, "CMAKE_ARGS", "BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target "
                                + target + " --parallel 1\n    CMAKE_ARGS")
        if name == "fastpfor":
            contents = replace_once(contents, "CMAKE_ARGS", "CMAKE_ARGS\n    -DBTRBLOCKS_LIBRARY_ONLY:BOOL=ON")
        recipe.write_text(contents, encoding="utf-8", newline="\n")
    # Reuse the successful installed native library from the old cache, read only.
    (source / "cmake/croaring.cmake").write_text('''# Offline reuse of the pinned, already-built CRoaring (original cache unchanged).
set(CROARING_INCLUDE_DIR "${BTR_CACHED_CROARING_PREFIX}/include")
set(CROARING_LIBRARY_PATH "${BTR_CACHED_CROARING_PREFIX}/lib/libroaring.so")
add_library(croaring SHARED IMPORTED)
set_property(TARGET croaring PROPERTY IMPORTED_LOCATION "${CROARING_LIBRARY_PATH}")
set_property(TARGET croaring APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CROARING_INCLUDE_DIR}")
''', encoding="utf-8", newline="\n")


def tree_hashes(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Source/build-copy symlinks are not supported: " + str(path))
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest(path)
    return result


def source_files(root):
    if any(p.is_symlink() for p in root.iterdir()):
        raise ValueError("BtrBlocks source symlinks are not supported")
    paths = [p for p in root.iterdir() if p.is_file()]
    for name in SOURCE_DIRS:
        directory = root / name
        if not directory.is_dir():
            raise ValueError("Incomplete BtrBlocks source tree: " + str(directory))
        entries = list(directory.rglob("*"))
        if any(p.is_symlink() for p in entries):
            raise ValueError("BtrBlocks source symlinks are not supported")
        paths.extend(p for p in entries if p.is_file())
    if any(p.is_symlink() for p in paths):
        raise ValueError("BtrBlocks source symlinks are not supported")
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}


def cmake_set(name, value, kind="PATH"):
    text = str(value).replace("\\", "/")
    if any(char in text for char in [';', '\n', '\r', '"', '$']):
        raise ValueError("Unsupported special character in CMake path: " + text)
    return 'set({} "{}" CACHE {} "Offline dependency" FORCE)\n'.format(name, text, kind)


def prepare(args, get_revision=revision):
    source, build, cache, archives = [Path(v).resolve() for v in
                                     [args.source, args.build_dir, args.cache_dir, args.archives_dir]]
    if build == cache or source == build or source in build.parents or build in source.parents:
        raise ValueError("Use a NEW build directory outside the original source and vendor cache")
    # Never create or write anything under the source or cache trees.
    if cache in build.parents or build in cache.parents:
        raise ValueError("Offline build and original vendor cache must be disjoint")
    require_file(source / "btrblocks/btrblocks.hpp")
    boost, roaring, library = check_cache(cache, get_revision)
    bundles = {name: archive_path(archives, pin[0]) for name, pin in PINS.items()}
    for name, path in bundles.items():
        if digest(path) != PINS[name][1]:
            raise ValueError("SHA256 mismatch; preserve and re-download the pinned ZIP: " + str(path))
    original = source_files(source)
    signature = dict(schema=SCHEMA, source=str(source), source_sha256=original,
                     archives={name: dict(path=str(p), sha256=digest(p)) for name, p in bundles.items()},
                     boost=str(boost), boost_revision=BOOST_PIN, croaring=str(roaring),
                     croaring_revision=ROARING_PIN, croaring_library_sha256=digest(library),
                     croaring_header_sha256=tree_hashes(roaring / "include"))
    staged, vendor = build / "offline_source", build / "offline_vendor"
    manifest_path = build / "offline_manifest.json"
    config_path = build / "offline_paths.cmake"
    config = (cmake_set("BTR_LOCAL_FASTPFOR_SOURCE", vendor / "fastpfor")
              + cmake_set("BTR_LOCAL_FSST_SOURCE", vendor / "fsst")
              + cmake_set("BTR_CACHED_CROARING_PREFIX", roaring)
              + cmake_set("FETCHCONTENT_SOURCE_DIR_BOOST_SRC", boost)
              + cmake_set("FETCHCONTENT_FULLY_DISCONNECTED", "ON", "BOOL"))
    if manifest_path.exists():
        previous = json.loads(read_text(manifest_path))
        if (previous.get("inputs") != signature or previous.get("source_copy_sha256") != tree_hashes(staged)
                or previous.get("vendor_copy_sha256") != tree_hashes(vendor)
                or not config_path.is_file() or read_text(config_path) != config):
            raise ValueError("Existing offline build inputs/copies changed; use a new build directory")
        print("Verified unchanged offline build copy and pinned dependency inputs", flush=True)
        return staged
    if staged.exists() or vendor.exists() or config_path.exists() or (build / "CMakeCache.txt").exists():
        raise ValueError("Refusing to replace an existing/partial build; use a new offline build directory")
    build.mkdir(parents=True, exist_ok=True)
    staged.mkdir()
    for relative in original:
        target = staged / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    for name, archive in bundles.items():
        extract_zip(archive, vendor / name, PINS[name][0])
    patch_build_copy(staged, vendor / "fastpfor")
    allowed_recipe_changes = {"cmake/fastpfor.cmake", "cmake/fsst.cmake", "cmake/croaring.cmake"}
    if any(digest(staged / name) != checksum for name, checksum in original.items()
           if name not in allowed_recipe_changes):
        raise ValueError("Build preparation unexpectedly changed a non-approved source file")
    if source_files(source) != original:
        raise ValueError("Original BtrBlocks source changed concurrently; do not use this build")
    config_path.write_text(config, encoding="utf-8", newline="\n")
    report = dict(inputs=signature, source_copy_sha256=tree_hashes(staged),
                  vendor_copy_sha256=tree_hashes(vendor), original_sources_unchanged=True,
                  codec_source_edits=False, skipped="unused FastPFOR tests/executables and their GoogleTest download")
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Prepared offline build copy: " + str(staged), flush=True)
    print("Original BtrBlocks, ZIPs, Boost and CRoaring cache were not modified", flush=True)
    return staged


def main():
    if sys.version_info < (3, 10):
        raise ValueError("Offline BtrBlocks preparation requires Python >= 3.10")
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ["source", "build-dir", "cache-dir", "archives-dir"]:
        parser.add_argument("--" + argument, required=True)
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        print("BtrBlocks offline preparation: " + str(error), file=sys.stderr)
        sys.exit(2)
