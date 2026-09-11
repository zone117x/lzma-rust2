//! The LZMA SDK's decoders as 7-Zip builds them, for the benchmark: plain C,
//! and on aarch64 the build with the assembly inner loop that 7-Zip's own
//! arm64 binary runs. `build.rs` compiles them from `lzma-sdk/`.

use core::ffi::c_int;

unsafe extern "C" {
    fn SdkC_bench_lzma_decode(
        dest: *mut u8,
        dest_len: *mut usize,
        src: *const u8,
        src_len: *mut usize,
        props: *const u8,
    ) -> c_int;
    fn SdkC_bench_lzma2_decode(
        dest: *mut u8,
        dest_len: *mut usize,
        src: *const u8,
        src_len: *mut usize,
        prop: u8,
    ) -> c_int;
    #[cfg(target_arch = "aarch64")]
    fn SdkAsm_bench_lzma_decode(
        dest: *mut u8,
        dest_len: *mut usize,
        src: *const u8,
        src_len: *mut usize,
        props: *const u8,
    ) -> c_int;
    #[cfg(target_arch = "aarch64")]
    fn SdkAsm_bench_lzma2_decode(
        dest: *mut u8,
        dest_len: *mut usize,
        src: *const u8,
        src_len: *mut usize,
        prop: u8,
    ) -> c_int;
}

/// Which build of the SDK decodes.
#[derive(Clone, Copy, Debug)]
pub enum Sdk {
    /// `LzmaDec.c` alone.
    C,
    /// `LzmaDec.c` with `LzmaDecOpt.S` as its inner loop.
    #[cfg(target_arch = "aarch64")]
    Asm,
}

impl Sdk {
    /// Every build this target has.
    pub fn all() -> &'static [Sdk] {
        #[cfg(target_arch = "aarch64")]
        {
            &[Sdk::C, Sdk::Asm]
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            &[Sdk::C]
        }
    }

    /// The name the benchmark reports it under.
    pub fn name(self) -> &'static str {
        match self {
            Sdk::C => "7-Zip (C)",
            #[cfg(target_arch = "aarch64")]
            Sdk::Asm => "7-Zip (asm)",
        }
    }
}

/// Decodes a raw LZMA stream with its five property bytes into `dest`, which
/// must be exactly the uncompressed size. The error is the SDK's `SRes`.
pub fn lzma_decode(which: Sdk, dest: &mut [u8], src: &[u8], props: &[u8; 5]) -> Result<(), i32> {
    let mut dest_len = dest.len();
    let mut src_len = src.len();
    // SAFETY: the pointers and lengths describe live slices, and the SDK
    // writes at most `dest_len` bytes and reads at most `src_len`.
    let result = unsafe {
        match which {
            Sdk::C => SdkC_bench_lzma_decode(
                dest.as_mut_ptr(),
                &mut dest_len,
                src.as_ptr(),
                &mut src_len,
                props.as_ptr(),
            ),
            #[cfg(target_arch = "aarch64")]
            Sdk::Asm => SdkAsm_bench_lzma_decode(
                dest.as_mut_ptr(),
                &mut dest_len,
                src.as_ptr(),
                &mut src_len,
                props.as_ptr(),
            ),
        }
    };
    if result != 0 {
        return Err(result);
    }
    if dest_len != dest.len() {
        return Err(-1);
    }
    Ok(())
}

/// Decodes a raw LZMA2 stream with its dictionary property byte, likewise.
pub fn lzma2_decode(which: Sdk, dest: &mut [u8], src: &[u8], prop: u8) -> Result<(), i32> {
    let mut dest_len = dest.len();
    let mut src_len = src.len();
    // SAFETY: as above.
    let result = unsafe {
        match which {
            Sdk::C => SdkC_bench_lzma2_decode(
                dest.as_mut_ptr(),
                &mut dest_len,
                src.as_ptr(),
                &mut src_len,
                prop,
            ),
            #[cfg(target_arch = "aarch64")]
            Sdk::Asm => SdkAsm_bench_lzma2_decode(
                dest.as_mut_ptr(),
                &mut dest_len,
                src.as_ptr(),
                &mut src_len,
                prop,
            ),
        }
    };
    if result != 0 {
        return Err(result);
    }
    if dest_len != dest.len() {
        return Err(-1);
    }
    Ok(())
}

/// The LZMA2 dictionary property byte for a dictionary of at least `dict_size`
/// bytes: the smallest of the forty sizes the format can name that is large
/// enough.
pub fn lzma2_dict_prop(dict_size: u32) -> u8 {
    (0..40u8)
        .find(|&prop| lzma2_dict_size(prop) >= dict_size)
        .unwrap_or(40)
}

/// The dictionary size an LZMA2 property byte names.
pub fn lzma2_dict_size(prop: u8) -> u32 {
    if prop >= 40 {
        u32::MAX
    } else {
        (2 | u32::from(prop & 1)) << (prop / 2 + 11)
    }
}
