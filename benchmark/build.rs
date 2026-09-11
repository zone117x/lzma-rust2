//! Builds the LZMA SDK's decoder twice: as plain C, and on aarch64 with the
//! assembly inner loop, as 7-Zip's own arm64 build has it. Each build gets
//! its functions renamed through a header so that both link into one binary.

use std::path::PathBuf;

fn sdk(name: &str, rename: &str, asm: bool) {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let sdk = manifest.join("lzma-sdk");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join(name);
    std::fs::create_dir_all(&out).unwrap();
    let mut build = cc::Build::new();
    build
        .include(&sdk)
        .files([
            sdk.join("LzmaDec.c"),
            sdk.join("Lzma2Dec.c"),
            sdk.join("shim.c"),
        ])
        .flag("-include")
        .flag(sdk.join(rename).to_str().unwrap())
        .opt_level(2)
        .warnings(false)
        .out_dir(&out);
    if asm {
        // 7zip_gcc.mak compiles LzmaDec.c with -DZ7_LZMA_DEC_OPT when the
        // assembly decoder is in.
        build
            .define("Z7_LZMA_DEC_OPT", None)
            .include(sdk.join("arm64"))
            .file(sdk.join("arm64").join("LzmaDecOpt.S"));
    }
    build.compile(name);
}

fn main() {
    println!("cargo:rerun-if-changed=lzma-sdk");
    println!("cargo:rerun-if-changed=build.rs");
    sdk("sdk_c", "rename_c.h", false);
    if std::env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "aarch64" {
        sdk("sdk_asm", "rename_asm.h", true);
    }
}
