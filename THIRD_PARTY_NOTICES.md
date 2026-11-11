# Third-party integration notices

This repository does **not** contain a ClaSPy or BtrBlocks source tree. It
contains two narrow integrations for the CACF bit-change detector and one
fixed-composition BtrBlocks adapter. CACF's own license has not yet been chosen.

## ClaSPy

`tools/class_bit_hamming.py` and `tools/class_xor_change.py` adapt the internal
streaming nearest-neighbour/control-flow behavior of
[ClaSPy 0.2.8](https://github.com/ermshaua/claspy/tree/v0.2.8) for binary64
Hamming and adjacent-XOR-change inputs. They import ClaSPy at runtime; the
upstream package is not copied here. The upstream BSD 3-Clause copyright,
conditions and disclaimer are retained in
[`third_party_licenses/ClaSPy-BSD-3-Clause.txt`](third_party_licenses/ClaSPy-BSD-3-Clause.txt).

## BtrBlocks

`harness/external_baselines/btrblocks_fixed/fixed_btr.cpp` is a CACF-side
stored-file wrapper that calls externally obtained
[BtrBlocks](https://github.com/maxi-k/btrblocks) primitives. Its decimal
fraction lookup constants follow BtrBlocks's native `Pseudodecimal.cpp`
table. The upstream MIT copyright and permission notice are retained in
[`third_party_licenses/BtrBlocks-MIT.txt`](third_party_licenses/BtrBlocks-MIT.txt).
The private build-copy script modifies two upstream files during the external
build; it does **not** modify a user's original BtrBlocks checkout or include
those files in this repository. `BtrBlocksFixed` is not the unmodified full
BtrBlocks compressor.

Other named codecs and baselines are external dependencies only. Their source
and executable files are not part of this draft.
