# Decoding benchmark

Single-threaded decoding of `tests/data/executable.exe` at every preset, the way
the crate's README measures it, by five decoders over the same compressed bytes:

- **lzma-rust2 master**: the master branch this branch was made from, as a git
  dependency.
- **lzma-rust2 asm**: this branch, whose `optimization` feature decodes LZMA
  on aarch64 through one inline assembly kernel that owns the symbol loop: the
  range coder, the window position, the state and the repeat distance stay in
  registers from one symbol to the next, both children of a tree node are
  loaded before the bit that chooses between them is known (the trick of the
  LZMA SDK's arm64 decoder), and a matched literal's candidates are loaded two
  bits ahead, where the SDK loads after each bit. Its `LzmaReader` reads its
  input through a buffer.
- **liblzma**: the `liblzma` crate over the C library of the same name.
- **7-Zip (C)**: the LZMA SDK's `LzmaDec.c` and `Lzma2Dec.c` as shipped in
  7-Zip 26.02 (`lzma-sdk/`, public domain).
- **7-Zip (asm)**: the same with `LzmaDecOpt.S` as the inner loop, which is
  what 7-Zip's own arm64 binary runs. On other targets only the C build is
  made.

Each stream is compressed once by this branch's encoder, and each decoder's
output is checked against the input before it is timed. The LZMA streams carry
the `.lzma` header, the LZMA2 streams are raw.

```
cd benchmark
cargo bench --bench decoding
python3 report.py      # the tables below and the charts in assets/
```

The package is a workspace of its own so that it leaves the crate's lock file
alone. `build.rs` compiles the SDK decoder twice through `cc`, with the
functions of each build renamed (`lzma-sdk/rename_*.h`) so that both link into
the one benchmark binary.

## Results

Apple M3 Max, macOS, rustc 1.98, 7-Zip's SDK 26.02, liblzma 5.8.3 (the crate's
bundled copy), lzma-rust2 master at 5614fd5. MiB/s of output, Criterion's mean
of 20 samples.

### decompression lzma2

This branch decodes 1.5 times faster than master, 1.4 times faster than
liblzma, and about 1.1 times faster than 7-Zip's own arm64 build, whose whole
decoding loop is the SDK's assembly.

| preset | lzma-rust2 master | lzma-rust2 asm | liblzma | 7-Zip (C) | 7-Zip (asm) |
|---|---:|---:|---:|---:|---:|
| 0 | 86 | 131 | 92 | 95 | 120 |
| 1 | 93 | 140 | 97 | 101 | 128 |
| 2 | 96 | 146 | 101 | 105 | 133 |
| 3 | 100 | 150 | 104 | 108 | 137 |
| 4 | 99 | 149 | 104 | 108 | 136 |
| 5 | 102 | 153 | 107 | 111 | 140 |
| 6 | 102 | 154 | 108 | 112 | 141 |
| 7 | 103 | 154 | 109 | 113 | 142 |
| 8 | 103 | 154 | 109 | 114 | 142 |
| 9 | 103 | 155 | 109 | 112 | 141 |

![decompression lzma2](./assets/decompression_lzma2.svg)

### decompression lzma

The same through `LzmaReader`, which on this branch reads its input through a
buffer and so decodes through the kernel too.

| preset | lzma-rust2 master | lzma-rust2 asm | liblzma | 7-Zip (C) | 7-Zip (asm) |
|---|---:|---:|---:|---:|---:|
| 0 | 87 | 132 | 92 | 95 | 121 |
| 1 | 93 | 140 | 97 | 101 | 128 |
| 2 | 96 | 146 | 101 | 105 | 134 |
| 3 | 100 | 150 | 104 | 108 | 138 |
| 4 | 99 | 149 | 104 | 108 | 136 |
| 5 | 102 | 153 | 107 | 111 | 139 |
| 6 | 102 | 154 | 108 | 112 | 140 |
| 7 | 102 | 154 | 108 | 111 | 140 |
| 8 | 103 | 154 | 108 | 112 | 140 |
| 9 | 103 | 153 | 108 | 111 | 140 |

![decompression lzma](./assets/decompression_lzma.svg)
