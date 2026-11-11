# Injected into the PRIVATE upstream project by CMAKE_PROJECT_BtrBlocks_INCLUDE.
# No linkage to the original full-BtrBlocks executable/build is performed.
if(NOT UNIX)
  message(FATAL_ERROR "BtrBlocksFixed requires UNIX, matching its native library")
endif()
if(SAMPLING_TEST_MODE OR ENABLE_FOR_SCHEME)
  message(FATAL_ERROR "Fixed primitives require SAMPLING_TEST_MODE=OFF and ENABLE_FOR_SCHEME=OFF")
endif()
# This project include runs before upstream creates its btrblocks library.
# Directory options are inherited by that later target as well as our wrapper;
# target_compile_options(wrapper) alone would not protect Decimal's arithmetic.
add_compile_options(-fno-fast-math -ffp-contract=off)
if(NOT TARGET BtrBlocksFixed)
  add_executable(BtrBlocksFixed "${CMAKE_CURRENT_LIST_DIR}/fixed_btr.cpp")
  set_target_properties(BtrBlocksFixed PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
  target_compile_options(BtrBlocksFixed PRIVATE -march=native -fno-fast-math -ffp-contract=off)
  target_link_libraries(BtrBlocksFixed PRIVATE btrblocks)
endif()
