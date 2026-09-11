//! Single-threaded decoding of `executable.exe` at every preset, the way the
//! crate's README measures it: the master branch of lzma-rust2, this branch,
//! liblzma and 7-Zip's decoders (the LZMA SDK's C decoder, and on aarch64 the
//! same with its assembly inner loop, which is what 7-Zip's arm64 binary runs),
//! all over the same compressed bytes. Each decoder's output is checked against
//! the input once before it is timed.

use std::{
    hint::black_box,
    io::{Read, Write},
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use liblzma::{bufread::XzDecoder, stream};
use lzma_rust2_benchmark::{Sdk, lzma_decode, lzma2_decode, lzma2_dict_prop};

static TEST_DATA: &[u8] = include_bytes!("../../tests/data/executable.exe");

/// The `.lzma` header: five property bytes and the uncompressed size.
const LZMA_HEADER: usize = 13;

fn lzma2_streams() -> Vec<(Vec<u8>, u32)> {
    (0..=9)
        .map(|level| {
            let options = lzma_rust2::Lzma2Options::with_preset(level);
            let dict_size = options.lzma_options.dict_size;
            let mut compressed = Vec::new();
            let mut writer = lzma_rust2::Lzma2Writer::new(&mut compressed, options);
            writer.write_all(TEST_DATA).unwrap();
            writer.finish().unwrap();
            (compressed, dict_size)
        })
        .collect()
}

fn lzma_streams() -> Vec<Vec<u8>> {
    (0..=9)
        .map(|level| {
            let options = lzma_rust2::LzmaOptions::with_preset(level);
            let mut compressed = Vec::new();
            let mut writer = lzma_rust2::LzmaWriter::new_use_header(
                &mut compressed,
                &options,
                Some(TEST_DATA.len() as u64),
            )
            .unwrap();
            writer.write_all(TEST_DATA).unwrap();
            writer.finish().unwrap();
            compressed
        })
        .collect()
}

fn bench_decompression_lzma2(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompression lzma2");
    group.throughput(Throughput::Bytes(TEST_DATA.len() as u64));
    group.sample_size(20);

    let streams = lzma2_streams();
    let mut out = vec![0u8; TEST_DATA.len()];

    for (level, (compressed, dict_size)) in streams.iter().enumerate() {
        let prop = lzma2_dict_prop(*dict_size);

        // Every decoder gives the input back.
        {
            let mut decoded = Vec::with_capacity(TEST_DATA.len());
            lzma_rust2_master::Lzma2Reader::new(compressed.as_slice(), *dict_size, None)
                .read_to_end(&mut decoded)
                .unwrap();
            assert!(decoded == TEST_DATA, "lzma-rust2 master, level {level}");
            decoded.clear();
            lzma_rust2::Lzma2Reader::new(compressed.as_slice(), *dict_size, None)
                .read_to_end(&mut decoded)
                .unwrap();
            assert!(decoded == TEST_DATA, "lzma-rust2 asm, level {level}");
            decoded.clear();
            XzDecoder::new_stream(compressed.as_slice(), liblzma_lzma2(prop))
                .read_to_end(&mut decoded)
                .unwrap();
            assert!(decoded == TEST_DATA, "liblzma, level {level}");
            for &sdk in Sdk::all() {
                out.fill(0);
                lzma2_decode(sdk, &mut out, compressed, prop).unwrap();
                assert!(out == TEST_DATA, "{}, level {level}", sdk.name());
            }
        }

        group.bench_with_input(
            BenchmarkId::new("lzma-rust2 master", level),
            compressed,
            |b, compressed| {
                b.iter(|| {
                    let mut decoded = Vec::with_capacity(TEST_DATA.len());
                    let mut reader = lzma_rust2_master::Lzma2Reader::new(
                        black_box(compressed.as_slice()),
                        *dict_size,
                        None,
                    );
                    reader.read_to_end(black_box(&mut decoded)).unwrap();
                    black_box(decoded)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lzma-rust2 asm", level),
            compressed,
            |b, compressed| {
                b.iter(|| {
                    let mut decoded = Vec::with_capacity(TEST_DATA.len());
                    let mut reader = lzma_rust2::Lzma2Reader::new(
                        black_box(compressed.as_slice()),
                        *dict_size,
                        None,
                    );
                    reader.read_to_end(black_box(&mut decoded)).unwrap();
                    black_box(decoded)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("liblzma", level),
            compressed,
            |b, compressed| {
                b.iter(|| {
                    let mut decoded = Vec::with_capacity(TEST_DATA.len());
                    let mut reader = XzDecoder::new_stream(
                        black_box(compressed.as_slice()),
                        liblzma_lzma2(prop),
                    );
                    reader.read_to_end(black_box(&mut decoded)).unwrap();
                    black_box(decoded)
                });
            },
        );

        for &sdk in Sdk::all() {
            group.bench_with_input(
                BenchmarkId::new(sdk.name(), level),
                compressed,
                |b, compressed| {
                    b.iter(|| {
                        let mut decoded = vec![0u8; TEST_DATA.len()];
                        lzma2_decode(sdk, black_box(&mut decoded), black_box(compressed), prop)
                            .unwrap();
                        black_box(decoded)
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_decompression_lzma(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompression lzma");
    group.throughput(Throughput::Bytes(TEST_DATA.len() as u64));
    group.sample_size(20);

    let streams = lzma_streams();
    let mut out = vec![0u8; TEST_DATA.len()];

    for (level, compressed) in streams.iter().enumerate() {
        let props: [u8; 5] = compressed[..5].try_into().unwrap();
        let body = &compressed[LZMA_HEADER..];

        {
            let mut decoded = Vec::with_capacity(TEST_DATA.len());
            lzma_rust2_master::LzmaReader::new_mem_limit(compressed.as_slice(), u32::MAX, None)
                .unwrap()
                .read_to_end(&mut decoded)
                .unwrap();
            assert!(decoded == TEST_DATA, "lzma-rust2 master, level {level}");
            decoded.clear();
            lzma_rust2::LzmaReader::new_mem_limit(compressed.as_slice(), u32::MAX, None)
                .unwrap()
                .read_to_end(&mut decoded)
                .unwrap();
            assert!(decoded == TEST_DATA, "lzma-rust2 asm, level {level}");
            decoded.clear();
            XzDecoder::new_stream(compressed.as_slice(), liblzma_lzma())
                .read_to_end(&mut decoded)
                .unwrap();
            assert!(decoded == TEST_DATA, "liblzma, level {level}");
            for &sdk in Sdk::all() {
                out.fill(0);
                lzma_decode(sdk, &mut out, body, &props).unwrap();
                assert!(out == TEST_DATA, "{}, level {level}", sdk.name());
            }
        }

        group.bench_with_input(
            BenchmarkId::new("lzma-rust2 master", level),
            compressed,
            |b, compressed| {
                b.iter(|| {
                    let mut decoded = Vec::with_capacity(TEST_DATA.len());
                    let mut reader = lzma_rust2_master::LzmaReader::new_mem_limit(
                        black_box(compressed.as_slice()),
                        u32::MAX,
                        None,
                    )
                    .unwrap();
                    reader.read_to_end(black_box(&mut decoded)).unwrap();
                    black_box(decoded)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lzma-rust2 asm", level),
            compressed,
            |b, compressed| {
                b.iter(|| {
                    let mut decoded = Vec::with_capacity(TEST_DATA.len());
                    let mut reader = lzma_rust2::LzmaReader::new_mem_limit(
                        black_box(compressed.as_slice()),
                        u32::MAX,
                        None,
                    )
                    .unwrap();
                    reader.read_to_end(black_box(&mut decoded)).unwrap();
                    black_box(decoded)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("liblzma", level),
            compressed,
            |b, compressed| {
                b.iter(|| {
                    let mut decoded = Vec::with_capacity(TEST_DATA.len());
                    let mut reader =
                        XzDecoder::new_stream(black_box(compressed.as_slice()), liblzma_lzma());
                    reader.read_to_end(black_box(&mut decoded)).unwrap();
                    black_box(decoded)
                });
            },
        );

        for &sdk in Sdk::all() {
            group.bench_with_input(
                BenchmarkId::new(sdk.name(), level),
                compressed,
                |b, compressed| {
                    b.iter(|| {
                        let mut decoded = vec![0u8; TEST_DATA.len()];
                        lzma_decode(
                            sdk,
                            black_box(&mut decoded),
                            black_box(&compressed[LZMA_HEADER..]),
                            &props,
                        )
                        .unwrap();
                        black_box(decoded)
                    });
                },
            );
        }
    }

    group.finish();
}

/// liblzma's raw LZMA2 decoder for the dictionary the property byte names.
fn liblzma_lzma2(prop: u8) -> stream::Stream {
    let mut filters = stream::Filters::new();
    filters.lzma2_properties(&[prop]).unwrap();
    stream::Stream::new_raw_decoder(&filters).unwrap()
}

/// liblzma's `.lzma` decoder.
fn liblzma_lzma() -> stream::Stream {
    stream::Stream::new_lzma_decoder(u64::MAX).unwrap()
}

criterion_group!(benches, bench_decompression_lzma2, bench_decompression_lzma);
criterion_main!(benches);
