# The LZMA SDK's decoder

`7zTypes.h`, `Compiler.h`, `Precomp.h`, `LzmaDec.c`, `LzmaDec.h`, `Lzma2Dec.c`,
`Lzma2Dec.h` and `arm64/7zAsm.S`, `arm64/LzmaDecOpt.S` are the LZMA SDK's files
as shipped in 7-Zip 26.02 (2026-06-25), unchanged. Igor Pavlov placed the LZMA SDK
in the public domain.

`shim.c` calls the SDK's one-shot decoders for the benchmark. `rename_c.h` and
`rename_asm.h` rename every function of the decoder, so that the plain C build
and the build with the assembly inner loop (which is what 7-Zip's own arm64
binary runs) can be linked into one benchmark and compared. `build.rs` compiles
the two.
