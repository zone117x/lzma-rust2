//! The LZMA decoder for aarch64 with the `optimization` feature: the same
//! algorithm as [`decoder`](crate::decoder), with its probabilities in one
//! array in the layout of 7-Zip's `LzmaDec.c`, so that the bit trees, the
//! literal and the matched literal are runs of adjacent probabilities a
//! kernel can walk.
//!
//! Over buffered input the symbols are decoded by [`Coder`], which holds the
//! range coder's state in locals for the whole run, so that it stays in
//! registers, and decodes those runs of bits with inline assembly kernels
//! that load both children of a tree node before the bit that chooses
//! between them is known. That is the trick of the LZMA SDK's arm64 decoder,
//! and it is worth a third over the same loops in Rust, where the next
//! probability is loaded only once the bit has been decided. Over input that
//! arrives a byte at a time, and for the last bytes of a buffer, the same
//! symbols are decoded through the [`RangeDecoder`] as [`Bits`]. The choice
//! of symbol, the state machine, the distances and the copies are Rust
//! either way.

use alloc::{vec, vec::Vec};

use crate::{
    BIT_MODEL_TOTAL, BIT_MODEL_TOTAL_BITS, MOVE_BITS, SHIFT_BITS, TOP_VALUE, error_other,
    lz::{LzDecoder, WindowParts},
    lzma_reader::IN_REQUIRED,
    range_dec::{RangeCoderState, RangeDecoder, RangeReader},
};

// The layout of 7-Zip's `LzmaDec.c`: every probability in one array, the
// position-state tables indexed as `(pos_state << 4) + state`.
const NUM_POS_BITS_MAX: usize = 4;
const NUM_POS_STATES_MAX: usize = 1 << NUM_POS_BITS_MAX;
const LEN_NUM_LOW_BITS: u32 = 3;
const LEN_NUM_LOW_SYMBOLS: usize = 1 << LEN_NUM_LOW_BITS;
const LEN_NUM_HIGH_BITS: u32 = 8;
const LEN_NUM_HIGH_SYMBOLS: usize = 1 << LEN_NUM_HIGH_BITS;
const LEN_LOW: usize = 0;
const LEN_HIGH: usize = LEN_LOW + 2 * (NUM_POS_STATES_MAX << LEN_NUM_LOW_BITS);
const NUM_LEN_PROBS: usize = LEN_HIGH + LEN_NUM_HIGH_SYMBOLS;
const LEN_CHOICE: usize = LEN_LOW;
const LEN_CHOICE2: usize = LEN_LOW + (1 << LEN_NUM_LOW_BITS);
const NUM_STATES: usize = 12;
const NUM_STATES2: usize = 16;
const NUM_LIT_STATES: usize = 7;
const START_POS_MODEL_INDEX: u32 = 4;
const END_POS_MODEL_INDEX: u32 = 14;
const NUM_FULL_DISTANCES: usize = 1 << (END_POS_MODEL_INDEX >> 1);
const NUM_POS_SLOT_BITS: u32 = 6;
const NUM_LEN_TO_POS_STATES: usize = 4;
const NUM_ALIGN_BITS: u32 = 4;
const ALIGN_TABLE_SIZE: usize = 1 << NUM_ALIGN_BITS;
const MATCH_MIN_LEN: u32 = 2;
const SPEC_POS: usize = 0;
const IS_REP0_LONG: usize = SPEC_POS + NUM_FULL_DISTANCES;
const REP_LEN_CODER: usize = IS_REP0_LONG + (NUM_STATES2 << NUM_POS_BITS_MAX);
const LEN_CODER: usize = REP_LEN_CODER + NUM_LEN_PROBS;
const IS_MATCH: usize = LEN_CODER + NUM_LEN_PROBS;
const ALIGN: usize = IS_MATCH + (NUM_STATES2 << NUM_POS_BITS_MAX);
const IS_REP: usize = ALIGN + ALIGN_TABLE_SIZE;
const IS_REP_G0: usize = IS_REP + NUM_STATES;
const IS_REP_G1: usize = IS_REP_G0 + NUM_STATES;
const IS_REP_G2: usize = IS_REP_G1 + NUM_STATES;
const POS_SLOT: usize = IS_REP_G2 + NUM_STATES;
const LITERAL: usize = POS_SLOT + (NUM_LEN_TO_POS_STATES << NUM_POS_SLOT_BITS);
const NUM_BASE_PROBS: usize = LITERAL;
const LIT_SIZE: usize = 0x300;
const PROB_INIT: u16 = 1024;

/// The probability update in one form: `prob - ((prob - OFFSET) >> 5)` is
/// `prob + ((2048 - prob) >> 5)` for a 0, the offset making the rounding
/// come out right, and with `prob` in place of the difference it is
/// `prob - (prob >> 5)` for a 1.
const BIT_MODEL_OFFSET: u32 = BIT_MODEL_TOTAL - (1 << MOVE_BITS) + 1;

/// What the symbol decoder needs of a range coder: single bits and the runs
/// of bits the layout keeps adjacent.
trait Bits {
    fn bit(&mut self, prob: &mut u16) -> u32;

    /// A bit tree of `probs.len()` leaves, a power of two; the leaf. Index 0
    /// is not used.
    fn tree(&mut self, probs: &mut [u16]) -> u32;

    /// The eight-bit tree of a literal, over the first 0x100 of a literal
    /// coder's probabilities.
    fn literal(&mut self, probs: &mut [u16]) -> u32;

    /// A literal against `match_byte`, over the 0x300 probabilities of one
    /// literal coder, the way `LzmaDec.c` indexes them.
    fn matched_literal(&mut self, probs: &mut [u16], match_byte: u32) -> u32;

    /// `count` bits of a reverse bit tree from node `start`, walked as 7-Zip's
    /// `REV_BIT` walks it: each bit moves to `node + step` for a 0 or `node +
    /// 2 * step` for a 1, and the step doubles. Returns the node reached; the
    /// value is that less `1 << count`.
    fn reverse(&mut self, probs: &mut [u16], start: u32, count: u32) -> u32;

    fn direct_bits(&mut self, count: u32) -> u32;
}

/// The loops in Rust, over any reader.
impl<R: RangeReader> Bits for RangeDecoder<R> {
    #[inline(always)]
    fn bit(&mut self, prob: &mut u16) -> u32 {
        self.decode_bit(prob) as u32
    }

    #[inline(always)]
    fn tree(&mut self, probs: &mut [u16]) -> u32 {
        let limit = probs.len();
        let mut i = 1usize;
        while i < limit {
            // The mask is a no-op on the index and lets the bounds check go.
            i = (i << 1) | self.decode_bit(&mut probs[i & (limit - 1)]) as usize;
        }
        (i - limit) as u32
    }

    #[inline(always)]
    fn literal(&mut self, probs: &mut [u16]) -> u32 {
        self.tree(probs)
    }

    #[inline(always)]
    fn matched_literal(&mut self, probs: &mut [u16], match_byte: u32) -> u32 {
        let mut match_byte = match_byte;
        let mut offs = 0x100u32;
        let mut symbol = 1u32;
        while symbol < 0x100 {
            match_byte += match_byte;
            let bit = offs;
            offs &= match_byte;
            let decoded = self.decode_bit(&mut probs[(offs + bit + symbol) as usize]) as u32;
            symbol = (symbol << 1) | decoded;
            if decoded == 0 {
                offs ^= bit;
            }
        }
        symbol - 0x100
    }

    #[inline(always)]
    fn reverse(&mut self, probs: &mut [u16], start: u32, count: u32) -> u32 {
        let mut node = start;
        let mut step = 1u32;
        for _ in 0..count {
            let bit = self.decode_bit(&mut probs[node as usize]) as u32;
            step += step;
            node += if bit == 0 { step >> 1 } else { step };
        }
        node
    }

    #[inline(always)]
    fn direct_bits(&mut self, count: u32) -> u32 {
        self.decode_direct_bits(count) as u32
    }
}

/// The range coder over a buffer, its state in locals: the kernels take it
/// through registers and give it back the same way, and nothing between them
/// has to go through memory.
struct Coder<'a> {
    range: u32,
    code: u32,
    pos: usize,
    buf: &'a [u8],
}

impl Coder<'_> {
    /// Whether a whole symbol, twenty bytes at most, is sure to be in the
    /// buffer from `pos`. The kernels read past `pos` freely up to that many.
    #[inline(always)]
    fn has_symbol(&self) -> bool {
        self.pos + IN_REQUIRED <= self.buf.len()
    }
}

// The one normalisation every kernel does before a bit: pull a byte in when
// the range's top byte is clear. The read is clamped to the buffer's last
// byte, so that a symbol running past the input reads that byte again and
// the caller, seeing the position past the end, reports the truncation.
macro_rules! normalize {
    () => {
        concat!(
            "tst    {range:w}, #0xFF000000\n",
            "b.ne   3f\n",
            "lsl    {range:w}, {range:w}, #{shift_bits}\n",
            "cmp    {pos}, {last}\n",
            "csel   {t}, {last}, {pos}, hi\n",
            "ldrb   {t:w}, [{buf}, {t}]\n",
            "add    {pos}, {pos}, #1\n",
            "orr    {code:w}, {t:w}, {code:w}, lsl #{shift_bits}\n",
            "3:\n",
        )
    };
}

// One bit against the probability in `prob`, with the range and code
// updated and the flags left saying which bit it was: `hs` for a 1. The new
// probability is left in `u`.
macro_rules! decide {
    () => {
        concat!(
            "lsr    {t:w}, {range:w}, #{total_bits}\n",
            "mul    {t:w}, {t:w}, {prob:w}\n",
            "sub    {u:w}, {prob:w}, #{offset}\n",
            "subs   {v:w}, {code:w}, {t:w}\n",
            "sub    {range:w}, {range:w}, {t:w}\n",
            "csel   {range:w}, {t:w}, {range:w}, lo\n",
            "csel   {code:w}, {v:w}, {code:w}, hs\n",
            "csel   {u:w}, {prob:w}, {u:w}, hs\n",
            "sub    {u:w}, {prob:w}, {u:w}, asr #{move_bits}\n",
        )
    };
}

// One step of a bit tree, unrolled: the children of the node loaded before
// the bit is known, the bit decided, the probability stored, and the walk
// taken down to the child chosen.
macro_rules! tree_step {
    () => {
        concat!(
            normalize!(),
            "add    {t}, {probs}, {sym}, lsl #2\n",
            "ldrh   {p0:w}, [{t}]\n",
            "ldrh   {p1:w}, [{t}, #2]\n",
            decide!(),
            "strh   {u:w}, [{probs}, {sym}, lsl #1]\n",
            "csel   {prob:w}, {p1:w}, {p0:w}, hs\n",
            "adc    {sym:w}, {sym:w}, {sym:w}\n",
        )
    };
}

// The last step of a tree, which has no children to load.
macro_rules! tree_last_step {
    () => {
        concat!(
            normalize!(),
            decide!(),
            "strh   {u:w}, [{probs}, {sym}, lsl #1]\n",
            "adc    {sym:w}, {sym:w}, {sym:w}\n",
        )
    };
}

// A step of a tree kernel for each token given, so that a kernel can be
// unrolled to its depth.
macro_rules! tree_step_for {
    ($_:tt) => {
        tree_step!()
    };
}

// An unrolled tree kernel: one step per token in the list, then the last
// step, over a tree of that many bits plus one.
macro_rules! unrolled_tree {
    ($name:ident, $bits:literal, [$($step:tt),*]) => {
        /// A bit tree of this many bits, unrolled.
        #[inline(always)]
        fn $name(&mut self, probs: &mut [u16]) -> u32 {
            debug_assert_eq!(probs.len(), 1 << $bits);
            let mut sym: u64 = 1;

            // SAFETY: the node visited at each step is `sym`, below the number
            // of leaves, and the children loaded ahead by every step but the
            // last are below it too. Every input read is clamped to the
            // buffer's last byte. The assembly touches no stack and only the
            // memory named.
            unsafe {
                core::arch::asm!(
                    $(tree_step_for!($step),)*
                    tree_last_step!(),
                    range = inout(reg) self.range,
                    code = inout(reg) self.code,
                    pos = inout(reg) self.pos,
                    sym = inout(reg) sym,
                    probs = in(reg) probs.as_mut_ptr(),
                    buf = in(reg) self.buf.as_ptr(),
                    last = in(reg) self.buf.len() - 1,
                    prob = inout(reg) u32::from(probs[1]) => _,
                    p0 = out(reg) _,
                    p1 = out(reg) _,
                    t = out(reg) _,
                    u = out(reg) _,
                    v = out(reg) _,
                    shift_bits = const SHIFT_BITS,
                    total_bits = const BIT_MODEL_TOTAL_BITS,
                    move_bits = const MOVE_BITS,
                    offset = const BIT_MODEL_OFFSET,
                    options(nostack),
                );
            }
            sym as u32 - (1 << $bits)
        }
    };
}

// A step of a reverse tree kernel: the nodes a 0 and a 1 lead to and their
// probabilities loaded before the bit is known, the bit decided, the
// probability stored, and the walk taken to the node chosen.
macro_rules! reverse_step {
    () => {
        concat!(
            normalize!(),
            "add    {n0}, {node}, {step}\n",
            "add    {step}, {step}, {step}\n",
            "add    {n1}, {node}, {step}\n",
            "ldrh   {p0:w}, [{probs}, {n0}, lsl #1]\n",
            "ldrh   {p1:w}, [{probs}, {n1}, lsl #1]\n",
            decide!(),
            "strh   {u:w}, [{probs}, {node}, lsl #1]\n",
            "csel   {node}, {n1}, {n0}, hs\n",
            "csel   {prob:w}, {p1:w}, {p0:w}, hs\n",
        )
    };
}

// The last step of a reverse tree, which loads no children.
macro_rules! reverse_last_step {
    () => {
        concat!(
            normalize!(),
            decide!(),
            "strh   {u:w}, [{probs}, {node}, lsl #1]\n",
            "add    {n0}, {node}, {step}\n",
            "add    {step}, {step}, {step}\n",
            "add    {n1}, {node}, {step}\n",
            "csel   {node}, {n1}, {n0}, hs\n",
        )
    };
}

impl Coder<'_> {
    unrolled_tree!(tree3, 3, [a, a]);
    unrolled_tree!(tree6, 6, [a, a, a, a, a]);
    unrolled_tree!(tree8, 8, [a, a, a, a, a, a, a]);

    /// The four align bits of a distance, a reverse tree from node 1,
    /// unrolled; returns the node reached, the value plus sixteen.
    #[inline(always)]
    fn align(&mut self, probs: &mut [u16]) -> u32 {
        debug_assert_eq!(probs.len(), ALIGN_TABLE_SIZE);
        let mut node: u64 = 1;

        // SAFETY: the nodes visited are 1, then one of 2 and 3, of 4 to 7, of
        // 8 to 15, all inside the sixteen; the last step loads none. Every
        // input read is clamped to the buffer's last byte. The assembly
        // touches no stack and only the memory named.
        unsafe {
            core::arch::asm!(
                reverse_step!(),
                reverse_step!(),
                reverse_step!(),
                reverse_last_step!(),
                range = inout(reg) self.range,
                code = inout(reg) self.code,
                pos = inout(reg) self.pos,
                node = inout(reg) node,
                step = inout(reg) 1u64 => _,
                probs = in(reg) probs.as_mut_ptr(),
                buf = in(reg) self.buf.as_ptr(),
                last = in(reg) self.buf.len() - 1,
                prob = inout(reg) u32::from(probs[1]) => _,
                p0 = out(reg) _,
                p1 = out(reg) _,
                n0 = out(reg) _,
                n1 = out(reg) _,
                t = out(reg) _,
                u = out(reg) _,
                v = out(reg) _,
                shift_bits = const SHIFT_BITS,
                total_bits = const BIT_MODEL_TOTAL_BITS,
                move_bits = const MOVE_BITS,
                offset = const BIT_MODEL_OFFSET,
                options(nostack),
            );
        }
        node as u32
    }
}

impl Bits for Coder<'_> {
    #[inline(always)]
    fn literal(&mut self, probs: &mut [u16]) -> u32 {
        self.tree8(probs)
    }

    /// A single bit, the ones the decoder branches on, in Rust: the same
    /// arithmetic as [`RangeDecoder::decode_bit`], with the input read
    /// clamped as in the kernels.
    #[inline(always)]
    fn bit(&mut self, prob: &mut u16) -> u32 {
        if self.range < TOP_VALUE {
            let b = u32::from(self.buf[self.pos.min(self.buf.len() - 1)]);
            self.pos += 1;
            self.code = (self.code << SHIFT_BITS) | b;
            self.range <<= SHIFT_BITS;
        }
        let p = u32::from(*prob);
        let bound = (self.range >> BIT_MODEL_TOTAL_BITS) * p;
        // 0 for a 0, all ones for a 1.
        let mask = 0u32.wrapping_sub(u32::from(self.code >= bound));
        self.range = (bound & !mask) | ((self.range - bound) & mask);
        self.code -= bound & mask;
        let offset = BIT_MODEL_OFFSET & !mask;
        *prob = p.wrapping_sub(p.wrapping_sub(offset) >> MOVE_BITS) as u16;
        mask & 1
    }

    #[inline(always)]
    fn tree(&mut self, probs: &mut [u16]) -> u32 {
        debug_assert!(probs.len() >= 2 && probs.len().is_power_of_two());
        // The sizes the format has, unrolled; the length is a constant at
        // every call, so this is no branch.
        match probs.len() {
            8 => return self.tree3(probs),
            64 => return self.tree6(probs),
            256 => return self.tree8(probs),
            _ => {}
        }
        let count = probs.len().trailing_zeros();
        let mut sym: u64 = 1;

        // SAFETY: the node visited at each step is `sym`, below the number of
        // leaves, and the children loaded ahead are `2 * sym` and `2 * sym +
        // 1`, below it too except on the last step, which loads indices 0 and
        // 1 instead. Every input read is clamped to the buffer's last byte.
        // The assembly touches no stack and only the memory named.
        unsafe {
            core::arch::asm!(
                "2:",
                normalize!(),
                // The children of this node, before the bit is known; the
                // last step has none and reads the unused first pair instead.
                "subs   {count:w}, {count:w}, #1",
                "add    {t}, {probs}, {sym}, lsl #2",
                "csel   {t}, {probs}, {t}, eq",
                "ldrh   {p0:w}, [{t}]",
                "ldrh   {p1:w}, [{t}, #2]",
                decide!(),
                "strh   {u:w}, [{probs}, {sym}, lsl #1]",
                // Down to the child the bit chose: the carry is the bit.
                "csel   {prob:w}, {p1:w}, {p0:w}, hs",
                "adc    {sym:w}, {sym:w}, {sym:w}",
                "cbnz   {count:w}, 2b",
                range = inout(reg) self.range,
                code = inout(reg) self.code,
                pos = inout(reg) self.pos,
                sym = inout(reg) sym,
                count = inout(reg) count => _,
                probs = in(reg) probs.as_mut_ptr(),
                buf = in(reg) self.buf.as_ptr(),
                last = in(reg) self.buf.len() - 1,
                prob = inout(reg) u32::from(probs[1]) => _,
                p0 = out(reg) _,
                p1 = out(reg) _,
                t = out(reg) _,
                u = out(reg) _,
                v = out(reg) _,
                shift_bits = const SHIFT_BITS,
                total_bits = const BIT_MODEL_TOTAL_BITS,
                move_bits = const MOVE_BITS,
                offset = const BIT_MODEL_OFFSET,
                options(nostack),
            );
        }
        sym as u32 - probs.len() as u32
    }

    #[inline(always)]
    fn matched_literal(&mut self, probs: &mut [u16], match_byte: u32) -> u32 {
        debug_assert_eq!(probs.len(), LIT_SIZE);
        let mut sym: u64 = 1;

        // SAFETY: the index is `offs + bit + sym` with `offs` and `bit` each 0
        // or 0x100 and `sym` below 0x100, so below 0x300. Every input read is
        // clamped to the buffer's last byte. The assembly touches no stack and
        // only the memory named.
        unsafe {
            core::arch::asm!(
                "2:",
                normalize!(),
                // The match byte's next bit picks the coder: the index is offs
                // + bit + sym, where bit is the offset so far and offs keeps it
                // only while the match byte's bit is set.
                "lsl    {mb:w}, {mb:w}, #1",
                "mov    {bit:w}, {offs:w}",
                "and    {offs:w}, {offs:w}, {mb:w}",
                "add    {idx:w}, {offs:w}, {bit:w}",
                "add    {idx:w}, {idx:w}, {sym:w}",
                "ldrh   {prob:w}, [{probs}, {idx}, lsl #1]",
                decide!(),
                "strh   {u:w}, [{probs}, {idx}, lsl #1]",
                // A decoded 0 that disagrees with the match byte drops the
                // offset: offs ^= bit.
                "eor    {t:w}, {offs:w}, {bit:w}",
                "csel   {offs:w}, {offs:w}, {t:w}, hs",
                "adc    {sym:w}, {sym:w}, {sym:w}",
                "subs   {count:w}, {count:w}, #1",
                "b.ne   2b",
                range = inout(reg) self.range,
                code = inout(reg) self.code,
                pos = inout(reg) self.pos,
                sym = inout(reg) sym,
                offs = inout(reg) 0x100u64 => _,
                mb = inout(reg) match_byte => _,
                count = inout(reg) 8u32 => _,
                probs = in(reg) probs.as_mut_ptr(),
                buf = in(reg) self.buf.as_ptr(),
                last = in(reg) self.buf.len() - 1,
                prob = out(reg) _,
                bit = out(reg) _,
                idx = out(reg) _,
                t = out(reg) _,
                u = out(reg) _,
                v = out(reg) _,
                shift_bits = const SHIFT_BITS,
                total_bits = const BIT_MODEL_TOTAL_BITS,
                move_bits = const MOVE_BITS,
                offset = const BIT_MODEL_OFFSET,
                options(nostack),
            );
        }
        sym as u32 - 0x100
    }

    #[inline(always)]
    fn reverse(&mut self, probs: &mut [u16], start: u32, count: u32) -> u32 {
        debug_assert!(count >= 1 && probs.len() >= 2);
        if probs.len() == ALIGN_TABLE_SIZE && start == 1 && count == NUM_ALIGN_BITS {
            return self.align(probs);
        }
        let mut node: u64 = u64::from(start);

        // SAFETY: the caller keeps every node visited inside `probs`; the two
        // children loaded ahead are the nodes the next step would visit, and
        // the last step loads index 0 instead. Every input read is clamped to
        // the buffer's last byte. The assembly touches no stack and only the
        // memory named.
        unsafe {
            core::arch::asm!(
                "2:",
                normalize!(),
                // The nodes a 0 and a 1 lead to, and their probabilities,
                // before the bit is known.
                "subs   {count:w}, {count:w}, #1",
                "add    {n0}, {node}, {step}",
                "add    {step}, {step}, {step}",
                "add    {n1}, {node}, {step}",
                "csel   {t}, xzr, {n0}, eq",
                "ldrh   {p0:w}, [{probs}, {t}, lsl #1]",
                "csel   {t}, xzr, {n1}, eq",
                "ldrh   {p1:w}, [{probs}, {t}, lsl #1]",
                decide!(),
                "strh   {u:w}, [{probs}, {node}, lsl #1]",
                "csel   {node}, {n1}, {n0}, hs",
                "csel   {prob:w}, {p1:w}, {p0:w}, hs",
                "cbnz   {count:w}, 2b",
                range = inout(reg) self.range,
                code = inout(reg) self.code,
                pos = inout(reg) self.pos,
                node = inout(reg) node,
                step = inout(reg) 1u64 => _,
                count = inout(reg) count => _,
                probs = in(reg) probs.as_mut_ptr(),
                buf = in(reg) self.buf.as_ptr(),
                last = in(reg) self.buf.len() - 1,
                prob = inout(reg) u32::from(probs[start as usize]) => _,
                p0 = out(reg) _,
                p1 = out(reg) _,
                n0 = out(reg) _,
                n1 = out(reg) _,
                t = out(reg) _,
                u = out(reg) _,
                v = out(reg) _,
                shift_bits = const SHIFT_BITS,
                total_bits = const BIT_MODEL_TOTAL_BITS,
                move_bits = const MOVE_BITS,
                offset = const BIT_MODEL_OFFSET,
                options(nostack),
            );
        }
        node as u32
    }

    #[inline(always)]
    fn direct_bits(&mut self, count: u32) -> u32 {
        let mut result: u32 = 0;

        // SAFETY: every input read is clamped to the buffer's last byte, and
        // the assembly touches no memory else and no stack.
        unsafe {
            core::arch::asm!(
                "2:",
                normalize!(),
                // Halve the range; the bit is 1 when the code is at or past it.
                "lsl    {result:w}, {result:w}, #1",
                "orr    {one:w}, {result:w}, #1",
                "lsr    {range:w}, {range:w}, #1",
                "subs   {v:w}, {code:w}, {range:w}",
                "csel   {code:w}, {v:w}, {code:w}, hs",
                "csel   {result:w}, {one:w}, {result:w}, hs",
                "subs   {count:w}, {count:w}, #1",
                "b.ne   2b",
                range = inout(reg) self.range,
                code = inout(reg) self.code,
                pos = inout(reg) self.pos,
                result = inout(reg) result,
                count = inout(reg) count => _,
                buf = in(reg) self.buf.as_ptr(),
                last = in(reg) self.buf.len() - 1,
                one = out(reg) _,
                t = out(reg) _,
                v = out(reg) _,
                shift_bits = const SHIFT_BITS,
                options(nostack, readonly),
            );
        }
        result
    }
}

/// The state of a run of symbols, in locals: the state as 7-Zip counts it
/// (0 to 11), the repeat distances as 7-Zip keeps them (the distance plus
/// one), and the masks of the properties.
struct Symbols {
    state: u32,
    reps: [u32; 4],
    /// The last byte of the output, the literal context; 0 before the first.
    previous: u32,
    lc: u32,
    lp_mask: u32,
    pb_mask: u32,
    end_marker: bool,
}

/// The decoder: the probabilities in 7-Zip's one array, the state as 7-Zip
/// counts it (0 to 11), the repeat distances as 7-Zip keeps them (the
/// distance plus one).
pub(crate) struct LzmaDecoder {
    probs: Vec<u16>,
    lc: u32,
    lp: u32,
    pb: u32,
    state: u32,
    reps: [u32; 4],
    end_marker: bool,
}

impl LzmaDecoder {
    pub(crate) fn new(lc: u32, lp: u32, pb: u32) -> Self {
        let probs = vec![PROB_INIT; NUM_BASE_PROBS + (LIT_SIZE << (lc + lp))];
        Self {
            probs,
            lc,
            lp,
            pb,
            state: 0,
            reps: [1; 4],
            end_marker: false,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.probs.fill(PROB_INIT);
        self.state = 0;
        self.reps = [1; 4];
        self.end_marker = false;
    }

    /// Whether the last `decode` stopped at the end of payload marker. The
    /// marker surfaces as a "dist overflow" error, as it always has, and
    /// this says whether that error was the marker.
    pub(crate) fn end_marker_detected(&self) -> bool {
        self.end_marker
    }

    /// Decodes into `lz` until its output is full or `rc` cannot start
    /// another symbol.
    pub(crate) fn decode<R: RangeReader>(
        &mut self,
        lz: &mut LzDecoder,
        rc: &mut RangeDecoder<R>,
    ) -> crate::Result<()> {
        lz.repeat_pending()?;
        if self.end_marker {
            return Err(error_other("dist overflow"));
        }
        // The state of the run in locals, where the kernels leave it in
        // registers; the decoder and the window take it back at the end.
        let mut w = lz.parts();
        let mut st = Symbols {
            state: self.state,
            reps: self.reps,
            previous: if w.full() != 0 {
                u32::from(w.get_byte(0))
            } else {
                0
            },
            lc: self.lc,
            lp_mask: (0x100u32 << self.lp) - (0x100u32 >> self.lc),
            pb_mask: (1u32 << self.pb) - 1,
            end_marker: false,
        };
        let probs = &mut self.probs[..];
        let mut result = Ok(());
        if rc.inner().is_buffer() {
            // The symbols a whole one of which is sure to be in the buffer,
            // through the kernels; the last few bytes are left to the loop
            // below.
            let state = rc.state();
            let mut coder = Coder {
                range: state.range,
                code: state.code,
                pos: rc.inner().pos(),
                buf: rc.inner().buf(),
            };
            while w.has_space() && coder.has_symbol() {
                if let Err(error) = Self::decode_symbol(probs, &mut w, &mut st, &mut coder) {
                    result = Err(error);
                    break;
                }
            }
            let (range, code, pos) = (coder.range, coder.code, coder.pos);
            rc.set_state(RangeCoderState { range, code });
            rc.inner_mut().set_pos(pos);
        }
        if result.is_ok() {
            while w.has_space() && rc.can_start_symbol() {
                if let Err(error) = Self::decode_symbol(probs, &mut w, &mut st, rc) {
                    result = Err(error);
                    break;
                }
            }
        }
        let space = w.has_space();
        let counters = w.counters();
        lz.set_counters(counters);
        self.state = st.state;
        self.reps = st.reps;
        self.end_marker = st.end_marker;
        result?;
        // Normalise only if we stopped because the output is full. If we
        // stopped because the input ran out, `rc` has to stay as it is, so
        // that the next call can pick up where this one left off.
        if !space && rc.can_normalize() {
            rc.normalize();
        }
        Ok(())
    }

    /// One LZMA symbol, as 7-Zip's `LzmaDec_DecodeReal` decodes it, into
    /// the window.
    #[inline(always)]
    fn decode_symbol<B: Bits>(
        probs: &mut [u16],
        lz: &mut WindowParts<'_>,
        st: &mut Symbols,
        rc: &mut B,
    ) -> crate::Result<()> {
        let pos_state = (lz.get_pos() as u32 & st.pb_mask) as usize;
        let mut state = st.state as usize;
        if rc.bit(&mut probs[IS_MATCH + (pos_state << NUM_POS_BITS_MAX) + state]) == 0 {
            let context = (((lz.get_pos() as u32) << 8) + st.previous) & st.lp_mask;
            let base = LITERAL + 3 * (context << st.lc) as usize;
            let symbol = if state < NUM_LIT_STATES {
                state -= if state < 4 { state } else { 3 };
                rc.literal(&mut probs[base..base + 0x100])
            } else {
                let match_byte = u32::from(lz.get_byte((st.reps[0] - 1) as usize));
                state -= if state < 10 { 3 } else { 6 };
                rc.matched_literal(&mut probs[base..base + LIT_SIZE], match_byte)
            };
            st.state = state as u32;
            st.previous = symbol;
            lz.put_byte(symbol as u8);
            return Ok(());
        }
        let len_base = if rc.bit(&mut probs[IS_REP + state]) == 0 {
            state += NUM_STATES;
            LEN_CODER
        } else {
            if rc.bit(&mut probs[IS_REP_G0 + state]) == 0 {
                if rc.bit(&mut probs[IS_REP0_LONG + (pos_state << NUM_POS_BITS_MAX) + state]) == 0 {
                    // A short repeat: one byte from the last distance.
                    if lz.full() == 0 {
                        return Err(error_other("dist overflow"));
                    }
                    st.state = if state < NUM_LIT_STATES { 9 } else { 11 };
                    lz.repeat((st.reps[0] - 1) as usize, 1)?;
                    st.previous = u32::from(lz.get_byte(0));
                    return Ok(());
                }
            } else {
                let distance = if rc.bit(&mut probs[IS_REP_G1 + state]) == 0 {
                    st.reps[1]
                } else {
                    let distance = if rc.bit(&mut probs[IS_REP_G2 + state]) == 0 {
                        st.reps[2]
                    } else {
                        let distance = st.reps[3];
                        st.reps[3] = st.reps[2];
                        distance
                    };
                    st.reps[2] = st.reps[1];
                    distance
                };
                st.reps[1] = st.reps[0];
                st.reps[0] = distance;
            }
            state = if state < NUM_LIT_STATES { 8 } else { 11 };
            REP_LEN_CODER
        };
        let mut len = Self::decode_len(probs, rc, len_base, pos_state);
        if state >= NUM_STATES {
            let len_state = (len as usize).min(NUM_LEN_TO_POS_STATES - 1);
            let slot = POS_SLOT + (len_state << NUM_POS_SLOT_BITS);
            let mut distance = rc.tree(&mut probs[slot..slot + (1 << NUM_POS_SLOT_BITS)]);
            if distance >= START_POS_MODEL_INDEX {
                let pos_slot = distance;
                let mut direct_bits = (distance >> 1) - 1;
                distance = 2 | (distance & 1);
                if pos_slot < END_POS_MODEL_INDEX {
                    distance <<= direct_bits;
                    let node = rc.reverse(
                        &mut probs[SPEC_POS..SPEC_POS + NUM_FULL_DISTANCES],
                        distance + 1,
                        direct_bits,
                    );
                    distance = node - (1 << direct_bits);
                } else {
                    direct_bits -= NUM_ALIGN_BITS;
                    distance = (distance << direct_bits) | rc.direct_bits(direct_bits);
                    distance <<= NUM_ALIGN_BITS;
                    let node = rc.reverse(
                        &mut probs[ALIGN..ALIGN + ALIGN_TABLE_SIZE],
                        1,
                        NUM_ALIGN_BITS,
                    );
                    distance |= node - (1 << NUM_ALIGN_BITS);
                    if distance == 0xFFFF_FFFF {
                        st.state = (state - NUM_STATES) as u32;
                        st.end_marker = true;
                        return Err(error_other("dist overflow"));
                    }
                }
            }
            st.reps[3] = st.reps[2];
            st.reps[2] = st.reps[1];
            st.reps[1] = st.reps[0];
            st.reps[0] = distance.wrapping_add(1);
            state = if state < NUM_STATES + NUM_LIT_STATES {
                NUM_LIT_STATES
            } else {
                NUM_LIT_STATES + 3
            };
            st.state = state as u32;
            if distance as usize >= lz.full() {
                return Err(error_other("dist overflow"));
            }
        } else {
            st.state = state as u32;
        }
        len += MATCH_MIN_LEN;
        lz.repeat((st.reps[0] - 1) as usize, len as usize)?;
        st.previous = u32::from(lz.get_byte(0));
        Ok(())
    }

    /// A length, 7-Zip's layout: the two choice bits at the head of the low
    /// table, low and mid trees of three bits per position state, a high
    /// tree of eight.
    #[inline(always)]
    fn decode_len<B: Bits>(probs: &mut [u16], rc: &mut B, base: usize, pos_state: usize) -> u32 {
        let len_state = pos_state << (LEN_NUM_LOW_BITS + 1);
        if rc.bit(&mut probs[base + LEN_CHOICE]) == 0 {
            let low = base + LEN_LOW + len_state;
            rc.tree(&mut probs[low..low + LEN_NUM_LOW_SYMBOLS])
        } else if rc.bit(&mut probs[base + LEN_CHOICE2]) == 0 {
            let mid = base + LEN_LOW + len_state + LEN_NUM_LOW_SYMBOLS;
            LEN_NUM_LOW_SYMBOLS as u32 + rc.tree(&mut probs[mid..mid + LEN_NUM_LOW_SYMBOLS])
        } else {
            let high = base + LEN_HIGH;
            2 * LEN_NUM_LOW_SYMBOLS as u32 + rc.tree(&mut probs[high..high + LEN_NUM_HIGH_SYMBOLS])
        }
    }
}
