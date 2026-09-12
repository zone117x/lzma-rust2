//! The LZMA decoder for aarch64 with the `optimization` feature: the same
//! algorithm as [`decoder`](crate::decoder), with its probabilities in one
//! array in the layout of 7-Zip's `LzmaDec.c`, so that every bit tree, the
//! literal and the matched literal are runs of adjacent probabilities.
//!
//! Over buffered input the symbols are decoded by one inline assembly
//! kernel that owns the loop, [`Coder::run`]: the range coder, the window's
//! position, the state and the repeat distances stay in registers from one
//! symbol to the next, the probabilities of both children of a tree node
//! are loaded before the bit that chooses between them is known, and the
//! decision bits are taken with branches. That is how the LZMA SDK's arm64
//! decoder is built, and a kernel called per run of bits from Rust, tried
//! first, gave a third of the gain away in the state it had to store and
//! reload around every call. The kernel is written from the step macros
//! below, so that each phase reads as the tree, the literal or the copy it
//! is. It hands a copy that wraps the window or runs past the output limit
//! to Rust, and the last bytes of a buffer, and input that arrives a byte at
//! a time, are decoded by the same symbol decoder in Rust.

use alloc::{vec, vec::Vec};

use crate::{
    BIT_MODEL_TOTAL, BIT_MODEL_TOTAL_BITS, MOVE_BITS, SHIFT_BITS, error_other,
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
/// The stride of the literal coders: a power of two, so that the coder of a
/// byte is a shift and an add from it, with a quarter of each unused.
const LIT_STRIDE: usize = 0x400;
const PROB_INIT: u16 = 1024;

/// The probability update in one form: `prob - ((prob - OFFSET) >> 5)` is
/// `prob + ((2048 - prob) >> 5)` for a 0, the offset making the rounding
/// come out right, and with `prob` in place of the difference it is
/// `prob - (prob >> 5)` for a 1.
const BIT_MODEL_OFFSET: u32 = BIT_MODEL_TOTAL - (1 << MOVE_BITS) + 1;

/// The state of a run of symbols, in locals: the state as 7-Zip counts it
/// (0 to 11), the repeat distances as 7-Zip keeps them (the distance plus
/// one), and the masks of the properties.
struct Symbols {
    state: u32,
    reps: [u32; 4],
    /// The last byte of the output, the literal context; 0 before the first.
    previous: u32,
    lc: u32,
    lp: u32,
    lp_mask: u32,
    pb_mask: u32,
    end_marker: bool,
    /// The length of a match the kernel left for Rust to copy.
    pending: u32,
}

// The one normalisation before every bit: pull a byte in when the range's
// top byte is clear. The read is not checked: a symbol starts only with
// twenty bytes left in the buffer and takes at most twenty, so it never
// reaches past the last.
macro_rules! normalize {
    () => {
        concat!(
            "tst    {range:w}, #0xFF000000\n",
            "b.ne   3f\n",
            "lsl    {range:w}, {range:w}, #{shift_bits}\n",
            "ldrb   {t:w}, [{pos}], #1\n",
            "orr    {code:w}, {t:w}, {code:w}, lsl #{shift_bits}\n",
            "3:\n",
        )
    };
}

// One bit against the probability in `prob`, without a branch: the range
// and the code updated, the flags left saying which bit it was, `hs` for a
// 1, and the new probability left in `u`.
macro_rules! decide {
    () => {
        concat!(
            "lsr    {t:w}, {range:w}, #{total_bits}\n",
            "mul    {t:w}, {t:w}, {prob:w}\n",
            "subs   {u:w}, {code:w}, {t:w}\n",
            "sub    {range:w}, {range:w}, {t:w}\n",
            "csel   {range:w}, {t:w}, {range:w}, lo\n",
            "csel   {code:w}, {u:w}, {code:w}, hs\n",
            "sub    {u:w}, {prob:w}, #{offset}\n",
            "csel   {u:w}, {prob:w}, {u:w}, hs\n",
            "sub    {u:w}, {prob:w}, {u:w}, asr #{move_bits}\n",
        )
    };
}

// A bit the decoder branches on, decoded with a branch as the SDK does, so
// that the range's update does not wait on the compare when the branch is
// predicted right: the probability at the offset from the table named, the
// range narrowed on the fall-through path for a 0, and the jump taken to
// the label for a 1, where `branch_bit_one` must follow.
macro_rules! branch_bit {
    ($tab:literal, $off:literal, $one:literal) => {
        concat!(
            "ldrh   {prob:w}, [",
            $tab,
            ", #",
            $off,
            "]\n",
            normalize!(),
            "lsr    {t:w}, {range:w}, #{total_bits}\n",
            "mul    {t:w}, {t:w}, {prob:w}\n",
            "cmp    {code:w}, {t:w}\n",
            "b.hs   ",
            $one,
            "f\n",
            "mov    {range:w}, {t:w}\n",
            "sub    {u:w}, {prob:w}, #{offset}\n",
            "sub    {u:w}, {prob:w}, {u:w}, asr #{move_bits}\n",
            "strh   {u:w}, [",
            $tab,
            ", #",
            $off,
            "]\n",
        )
    };
}

// The is-match bit, whose probability is loaded already, into `prob`, from
// the state's row `step` of the table at the position state's column `n1`:
// a jump to the match label named for a match, the fall-through a literal.
macro_rules! is_match_bit {
    ($one:literal) => {
        concat!(
            normalize!(),
            "mov    {t:w}, {range:w}\n",
            "lsr    {range:w}, {range:w}, #{total_bits}\n",
            "mul    {range:w}, {range:w}, {prob:w}\n",
            "cmp    {code:w}, {range:w}\n",
            "b.hs   ",
            $one,
            "\n",
            "sub    {u:w}, {prob:w}, #{offset}\n",
            "sub    {u:w}, {prob:w}, {u:w}, asr #{move_bits}\n",
            "strh   {u:w}, [{step}, {n1}]\n",
        )
    };
}

// What the next symbol needs from the position and the state as they will
// be, computed while the current symbol is still being decoded: the
// is-match row and column into `step` and `n1`, whose probability is loaded
// into `prob` as soon as `prob` is free, and the literal coders of the
// position's `lp` bits into `n0`, so that the coder of the next literal is
// found from the last byte alone. `masks` holds `8 - lc` in its low byte,
// the position bits' mask in its third, and the `lp` bits' mask shifted by
// `lc` in its high word.
macro_rules! next_symbol {
    ($pos:literal) => {
        concat!(
            "and    {n1}, {pbm}, ",
            $pos,
            ", lsl #5\n",
            "add    {step}, {pim}, {state}, lsl #1\n",
            "lsl    {n0}, ",
            $pos,
            ", #8\n",
            "lsr    {n0}, {n0}, {masks}\n",
            "and    {n0}, {n0}, {masks}, lsr #32\n",
            "ldr    {t}, [{p}, #168]\n",
            "add    {n0}, {t}, {n0}, lsl #11\n",
        )
    };
}

// The coder of a literal, into `tab`: the top `lc` bits of the last byte,
// `sym`, over the coders of the position in `n0`, each coder 0x400
// probabilities apart. This is the chain from the last bit of one literal
// to the first of the next, so it is as short as it can be: `sym` carries
// bit 8 above the byte, the leading one of its tree, which the shift
// leaves at bit `lc`, and the coders' base in `n0` is a coder `1 << lc`
// back to make up for it.
macro_rules! literal_coder {
    () => {
        concat!(
            "lsr    {tab}, {sym}, {masks}\n",
            "add    {tab}, {n0}, {tab}, lsl #11\n",
        )
    };
}

// The 1 path of `branch_bit`: the range and the code both lose the bound,
// the probability falls.
macro_rules! branch_bit_one {
    ($tab:literal, $off:literal) => {
        concat!(
            "sub    {range:w}, {range:w}, {t:w}\n",
            "sub    {code:w}, {code:w}, {t:w}\n",
            "sub    {u:w}, {prob:w}, {prob:w}, lsr #{move_bits}\n",
            "strh   {u:w}, [",
            $tab,
            ", #",
            $off,
            "]\n",
        )
    };
}

// A whole tree of three bits over the table named, from node 1, with the
// first probability loaded here.
macro_rules! tree3 {
    ($tab:literal) => {
        concat!(
            "mov    {sym:w}, #1\n",
            "ldrh   {prob:w}, [",
            $tab,
            ", #2]\n",
            "add    {tab2}, ",
            $tab,
            ", #2\n",
            sdk_step!($tab),
            sdk_step!($tab),
            sdk_last_step!($tab),
        )
    };
}

// A whole tree of six bits.
macro_rules! tree6 {
    ($tab:literal) => {
        concat!(
            "mov    {sym:w}, #1\n",
            "ldrh   {prob:w}, [",
            $tab,
            ", #2]\n",
            "add    {tab2}, ",
            $tab,
            ", #2\n",
            sdk_step!($tab),
            sdk_step!($tab),
            sdk_step!($tab),
            sdk_step!($tab),
            sdk_step!($tab),
            sdk_last_step!($tab),
        )
    };
}

// One step of a bit tree over the table named, in the shape of the SDK's,
// which decodes measurably faster than a step of the same instructions in
// another order: the node index kept doubled, the children loaded by
// register offset from the table and from the table plus one probability
// (`tab2`), the code's remainder and the probability update in temporaries
// of their own. The children are loaded before the bit is known, so that
// the load is off the critical path.
macro_rules! sdk_step {
    ($tab:literal) => {
        concat!(
            normalize!(),
            "lsr    {t:w}, {range:w}, #{total_bits}\n",
            "add    {sym:w}, {sym:w}, {sym:w}\n",
            "sub    {u:w}, {prob:w}, #{offset}\n",
            "mul    {t:w}, {t:w}, {prob:w}\n",
            "ldrh   {p1:w}, [",
            $tab,
            ", {sym}, lsl #1]\n",
            "subs   {v:w}, {code:w}, {t:w}\n",
            "sub    {range:w}, {range:w}, {t:w}\n",
            "csel   {range:w}, {t:w}, {range:w}, lo\n",
            "csel   {u:w}, {prob:w}, {u:w}, hs\n",
            "ldrh   {t:w}, [{tab2}, {sym}, lsl #1]\n",
            "sub    {u:w}, {prob:w}, {u:w}, asr #{move_bits}\n",
            "csel   {prob:w}, {p1:w}, {t:w}, lo\n",
            "csel   {code:w}, {v:w}, {code:w}, hs\n",
            "strh   {u:w}, [",
            $tab,
            ", {sym}]\n",
            "adc    {sym:w}, {sym:w}, wzr\n",
        )
    };
}

// The SDK's last step: no children.
macro_rules! sdk_last_step {
    ($tab:literal) => {
        concat!(
            normalize!(),
            "lsr    {t:w}, {range:w}, #{total_bits}\n",
            "add    {sym:w}, {sym:w}, {sym:w}\n",
            "sub    {u:w}, {prob:w}, #{offset}\n",
            "mul    {t:w}, {t:w}, {prob:w}\n",
            "subs   {v:w}, {code:w}, {t:w}\n",
            "sub    {range:w}, {range:w}, {t:w}\n",
            "csel   {range:w}, {t:w}, {range:w}, lo\n",
            "csel   {u:w}, {prob:w}, {u:w}, hs\n",
            "csel   {code:w}, {v:w}, {code:w}, hs\n",
            "sub    {u:w}, {prob:w}, {u:w}, asr #{move_bits}\n",
            "strh   {u:w}, [",
            $tab,
            ", {sym}]\n",
            "adc    {sym:w}, {sym:w}, wzr\n",
        )
    };
}

// A whole tree of eight bits.
macro_rules! tree8 {
    ($tab:literal) => {
        concat!(
            "mov    {sym:w}, #1\n",
            "ldrh   {prob:w}, [",
            $tab,
            ", #2]\n",
            "add    {tab2}, ",
            $tab,
            ", #2\n",
            sdk_step!($tab),
            sdk_step!($tab),
            sdk_step!($tab),
            sdk_step!($tab),
            sdk_step!($tab),
            sdk_step!($tab),
            sdk_step!($tab),
            sdk_last_step!($tab),
        )
    };
}

// One direct bit of a distance, into `tab`: the range halves, and the code
// keeps or loses the half. The last bit leaves for `43`.
macro_rules! direct_bit {
    () => {
        concat!(
            "lsr    {range:w}, {range:w}, #1\n",
            "subs   {u:w}, {code:w}, {range:w}\n",
            "add    {tab:w}, {tab:w}, {tab:w}\n",
            "csel   {code:w}, {code:w}, {u:w}, mi\n",
            "csinc  {tab:w}, {tab:w}, {tab:w}, mi\n",
            "subs   {cnt:w}, {cnt:w}, #1\n",
            "b.eq   43f\n",
        )
    };
}

// The same, after a check of the range: one whose top byte is clear goes
// to `44`, which normalises it and decodes eight bits before the next
// check.
macro_rules! direct_checked {
    () => {
        concat!(
            "tst    {range:w}, #0xFF000000\n",
            "b.eq   44f\n",
            direct_bit!(),
        )
    };
}

// A step of a reverse tree over the table named: the nodes a 0 and a 1 lead
// to and their probabilities loaded before the bit is known, the bit
// decided, the probability stored, and the walk taken to the node chosen.
// The node is `sym`, the step `step`.
macro_rules! reverse_step {
    ($tab:literal) => {
        concat!(
            normalize!(),
            "add    {n0}, {sym}, {step}\n",
            "add    {step}, {step}, {step}\n",
            "add    {n1}, {sym}, {step}\n",
            "ldrh   {v:w}, [",
            $tab,
            ", {n0}, lsl #1]\n",
            "ldrh   {p1:w}, [",
            $tab,
            ", {n1}, lsl #1]\n",
            decide!(),
            "strh   {u:w}, [",
            $tab,
            ", {sym}, lsl #1]\n",
            "csel   {sym}, {n1}, {n0}, hs\n",
            "csel   {prob:w}, {p1:w}, {v:w}, hs\n",
        )
    };
}

// The last step of a reverse tree, which loads no children.
macro_rules! reverse_last_step {
    ($tab:literal) => {
        concat!(
            normalize!(),
            decide!(),
            "strh   {u:w}, [",
            $tab,
            ", {sym}, lsl #1]\n",
            "add    {n0}, {sym}, {step}\n",
            "add    {step}, {step}, {step}\n",
            "add    {n1}, {sym}, {step}\n",
            "csel   {sym}, {n1}, {n0}, hs\n",
        )
    };
}

// The matched literal, decoded two bits ahead. Its coder has three rows of
// 0x100 probabilities: while the bits decoded so far agree with the match
// byte's, the offset `n0` is 0x100 and the row is the offset plus the match
// byte's next bit as 0x100, `n1`, and once a bit disagrees the offset is 0
// for the rest of the byte and the row is 0. The probability of a bit is at
// the row plus the tree node `sym`. The SDK loads it after the bit before
// is known, since the row depends on that bit, which puts the load's
// latency on the path between one bit and the next. The kernel loads both
// probabilities a bit may lead to before the bit is decoded, as it does for
// a tree, and for that it loads the four a pair of bits may lead to two
// bits ahead: the four in the row 0 at the four nodes two levels down,
// which are two words, and the one in the row the offset leads to if both
// bits agree with the match byte, `tab2`. When the first of the two bits is
// known it picks a word, whose halves the second bit will pick between, and
// the one probability of the agreeing row takes the half it belongs to if
// the offset is still up. The match byte is `cnt` shifted up by 8, so that
// bit `k` of the byte is bit `15 - k` of `cnt`; the shifts are the step's.
// Two pairs of registers hold the words in turn, one pair loaded while the
// other is picked from.

// The match byte's bit of this step as 0x100, masked by the offset.
macro_rules! ml_bit {
    ($shift:literal) => {
        concat!("and    {n1}, {n0}, {cnt}, lsr #", $shift, "\n")
    };
}

// The word the bit before picked, its halves as the next bit's two
// probabilities, and the agreeing row's probability into the half it
// belongs to if the offset is still up.
macro_rules! ml_pair {
    ($a:literal, $b:literal) => {
        concat!(
            "csel   ",
            $a,
            ", ",
            $b,
            ", ",
            $a,
            ", hs\n",
            "lsr    ",
            $b,
            ", ",
            $a,
            ", #16\n",
            "and    ",
            $a,
            ", ",
            $a,
            ", #0xFFFF\n",
            "cmp    {n0}, {n1}\n",
            "csel   ",
            $a,
            ", {tab2:w}, ",
            $a,
            ", ne\n",
            "cmp    {n1}, #0\n",
            "csel   ",
            $b,
            ", {tab2:w}, ",
            $b,
            ", ne\n",
        )
    };
}

// The loads two bits ahead: the agreeing row's probability at the node the
// match byte's next two bits lead to, and the two words of row 0 at the
// four nodes two levels down.
macro_rules! ml_loads {
    ($pair_shift:literal, $third_shift:literal, $a:literal, $b:literal) => {
        concat!(
            "ubfx   {t}, {cnt}, #",
            $pair_shift,
            ", #2\n",
            "and    {u}, {n0}, {cnt}, lsr #",
            $third_shift,
            "\n",
            "add    {u}, {u}, {n0}\n",
            "add    {t}, {t}, {u}\n",
            "add    {t}, {t}, {sym}, lsl #2\n",
            "ldrh   {tab2:w}, [{tab}, {t}, lsl #1]\n",
            "add    {u}, {tab}, {sym}, lsl #3\n",
            "ldr    ",
            $a,
            ", [{u}]\n",
            "ldr    ",
            $b,
            ", [{u}, #4]\n",
        )
    };
}

// The bit itself, with the SDK's range and code updates, the probability
// updated in its row, and the offset kept if the bit agrees with the match
// byte's.
macro_rules! ml_decide {
    () => {
        concat!(
            normalize!(),
            "lsr    {t:w}, {range:w}, #{total_bits}\n",
            "eor    {u:w}, {n0:w}, {n1:w}\n",
            "mul    {t:w}, {t:w}, {prob:w}\n",
            "subs   {v:w}, {code:w}, {t:w}\n",
            "sub    {range:w}, {range:w}, {t:w}\n",
            "csel   {range:w}, {t:w}, {range:w}, lo\n",
            "add    {t}, {tab}, {n0}, lsl #1\n",
            "add    {t}, {t}, {n1}, lsl #1\n",
            "csel   {n0:w}, {n1:w}, {u:w}, hs\n",
            "csel   {code:w}, {v:w}, {code:w}, hs\n",
            "sub    {u:w}, {prob:w}, #{offset}\n",
            "csel   {u:w}, {prob:w}, {u:w}, hs\n",
            "sub    {u:w}, {prob:w}, {u:w}, asr #{move_bits}\n",
            "strh   {u:w}, [{t}, {sym}, lsl #1]\n",
        )
    };
}

// The next bit's probability from the two the bit picks between, and the
// node.
macro_rules! ml_next {
    ($a:literal, $b:literal) => {
        concat!(
            "csel   {prob:w}, ",
            $a,
            ", ",
            $b,
            ", lo\n",
            "adc    {sym:w}, {sym:w}, {sym:w}\n",
        )
    };
}

/// What [`Coder::run`] stopped for.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Exit {
    /// The output is full, or the input is down to its last bytes.
    Limit,
    /// A match the kernel does not copy itself: one that wraps the window
    /// or runs past the output limit. `pending` and the first repeat
    /// distance say what to copy.
    Copy,
    /// A distance reaches before the window's data.
    Overflow,
    /// The end of payload marker.
    EndMarker,
}

/// The state the kernel loads at entry and stores at exit, in the order it
/// reads it. Do not reorder.
#[repr(C)]
struct Run {
    probs: *mut u16,
    /// The input's base, its position, its last byte and the last position
    /// a symbol may start at, as pointers.
    buf: *const u8,
    pos: *const u8,
    last: *const u8,
    stop: *const u8,
    range: u64,
    code: u64,
    dic: *mut u8,
    dic_pos: u64,
    dic_limit: u64,
    dic_full: u64,
    dic_size: u64,
    state: u64,
    reps: [u64; 4],
    previous: u64,
    /// `8 - lc` in the low byte, the position state mask in the third, and
    /// the literal position mask shifted by `lc` in the high half, so that
    /// one shift by the register and one and with it shifted down give the
    /// literal coder.
    masks: u64,
    len: u64,
    exit: u64,
    /// The literal coders' base, a coder `1 << lc` back, for the leading one
    /// the byte carries.
    lit: u64,
}

const _: () = {
    assert!(core::mem::offset_of!(Run, pos) == 16);
    assert!(core::mem::offset_of!(Run, last) == 24);
    assert!(core::mem::offset_of!(Run, stop) == 32);
    assert!(core::mem::offset_of!(Run, range) == 40);
    assert!(core::mem::offset_of!(Run, dic) == 56);
    assert!(core::mem::offset_of!(Run, dic_limit) == 72);
    assert!(core::mem::offset_of!(Run, dic_full) == 80);
    assert!(core::mem::offset_of!(Run, dic_size) == 88);
    assert!(core::mem::offset_of!(Run, state) == 96);
    assert!(core::mem::offset_of!(Run, reps) == 104);
    assert!(core::mem::offset_of!(Run, previous) == 136);
    assert!(core::mem::offset_of!(Run, masks) == 144);
    assert!(core::mem::offset_of!(Run, len) == 152);
    assert!(core::mem::offset_of!(Run, exit) == 160);
    assert!(core::mem::offset_of!(Run, lit) == 168);
};

/// The range coder over a buffer, for the kernel.
struct Coder<'a> {
    range: u32,
    code: u32,
    pos: usize,
    buf: &'a [u8],
}

impl Coder<'_> {
    /// Whether a whole symbol, twenty bytes at most, is sure to be in the
    /// buffer from `pos`.
    #[inline(always)]
    fn has_symbol(&self) -> bool {
        self.pos + IN_REQUIRED <= self.buf.len()
    }

    /// Decodes symbols into the window until the output is full, the input
    /// is down to its last bytes, a copy needs Rust, or the stream ends or
    /// fails, and says which.
    #[inline(never)]
    fn run(&mut self, probs: &mut [u16], w: &mut WindowParts<'_>, st: &mut Symbols) -> Exit {
        debug_assert!(probs.len() >= NUM_BASE_PROBS);
        let (dic, dic_pos, dic_limit, dic_full, dic_size) = w.raw_parts();
        let mut run = Run {
            probs: probs.as_mut_ptr(),
            buf: self.buf.as_ptr(),
            pos: self.buf[self.pos..].as_ptr(),
            last: self.buf[self.buf.len() - 1..].as_ptr(),
            stop: self.buf[self.buf.len() - IN_REQUIRED..].as_ptr(),
            range: u64::from(self.range),
            code: u64::from(self.code),
            dic,
            dic_pos: dic_pos as u64,
            dic_limit: dic_limit as u64,
            dic_full: dic_full as u64,
            dic_size: dic_size as u64,
            state: u64::from(st.state),
            reps: st.reps.map(u64::from),
            previous: u64::from(st.previous),
            masks: u64::from(8 - st.lc)
                | (u64::from(st.pb_mask) << 16)
                | ((((1u64 << st.lp) - 1) << st.lc) << 32),
            len: 0,
            exit: 0,
            lit: (probs.as_mut_ptr() as u64)
                .wrapping_add((LITERAL * 2) as u64)
                .wrapping_sub(((LIT_STRIDE * 2) as u64) << st.lc),
        };

        // SAFETY: the kernel reads and writes only `run`, the probabilities
        // of the layout inside `probs` (whose length covers every literal
        // coder the properties can address), the input buffer up to twenty
        // bytes past a position it only starts a symbol at with twenty bytes
        // left, and the window between the position and
        // the limit, itself below the window's size, with every source byte
        // read behind the position by a distance below the window's fill,
        // which it checks, and a copy that would reach before the window's
        // start or past its end left to Rust. The stack is used below the
        // pointer and restored.
        unsafe {
            core::arch::asm!(
                // The state into registers. The fourth repeat distance
                // stays in memory, used too rarely to hold a register.
                "str    {p}, [sp, #-16]!",
                "ldr    {probs}, [{p}]",
                "ldp    {pos}, {t}, [{p}, #16]",
                "ldp    {stop}, {range}, [{p}, #32]",
                "ldp    {code}, {dic}, [{p}, #48]",
                "ldp    {dpos}, {dlim}, [{p}, #64]",
                "ldr    {state}, [{p}, #96]",
                "ldr    {rep0}, [{p}, #104]",
                "ldp    {sym}, {masks}, [{p}, #136]",
                "orr    {sym}, {sym}, #0x100",
                "ubfx   {pbm}, {masks}, #16, #4",
                "lsl    {pbm}, {pbm}, #5",
                "add    {pim}, {probs}, #{o_is_match}",
                // ---- The first symbol's is-match row, column and
                // probability; every later symbol has them ready from the one
                // before. The state says which head it starts at: the one
                // after a match, after a matched literal, or after a
                // literal, which know the state's next value without asking.
                // ----
                next_symbol!("{dpos}"),
                "ldrh   {prob:w}, [{step}, {n1}]",
                "cmp    {state:w}, #7",
                "b.hs   16f",
                "cmp    {state:w}, #4",
                "b.hs   15f",
                ".p2align 6",
                // ---- After a literal: the state was below 4, and a literal
                // takes it to 0. ----
                "14:",
                "cmp    {dpos}, {dlim}",
                "b.hs   90f",
                "cmp    {pos}, {stop}",
                "b.hi   90f",
                is_match_bit!("20f"),
                "mov    {state:w}, wzr",
                // ---- A literal: the coder of the last byte, which is `sym`,
                // and the position. ----
                "11:",
                literal_coder!(),
                "add    {n0}, {dpos}, #1",
                next_symbol!("{n0}"),
                tree8!("{tab}"),
                "13:",
                // The byte, which is the next literal's context.
                "strb   {sym:w}, [{dic}, {dpos}]",
                "add    {dpos}, {dpos}, #1",
                "ldrh   {prob:w}, [{step}, {n1}]",
                "b      14b",
                // ---- After a matched literal: the state is 4 to 6, and a
                // literal takes it down by 3. ----
                "15:",
                "cmp    {dpos}, {dlim}",
                "b.hs   90f",
                "cmp    {pos}, {stop}",
                "b.hi   90f",
                is_match_bit!("20f"),
                "sub    {state:w}, {state:w}, #3",
                "b      11b",
                // ---- A match: the is-rep and repeat bits sit at the state. ----
                "20:",
                "sub    {code:w}, {code:w}, {range:w}",
                "sub    {range:w}, {t:w}, {range:w}",
                "sub    {u:w}, {prob:w}, {prob:w}, lsr #{move_bits}",
                "strh   {u:w}, [{step}, {n1}]",
                // The state's other tables lie at fixed offsets from its
                // is-match row.
                branch_bit!("{step}", "{r_is_rep}", "30"),
                // A fresh match: the state is marked, the length coder is
                // the match one.
                "orr    {state}, {state}, #16",
                "add    {tab}, {probs}, #{o_len_coder}",
                "b      40f",
                "30:",
                branch_bit_one!("{step}", "{r_is_rep}"),
                branch_bit!("{step}", "{r_is_rep_g0}", "31"),
                // The last distance again: one byte, or a length.
                "add    {cnt}, {probs}, {state}, lsl #1",
                "add    {cnt}, {cnt}, {n1}",
                branch_bit!("{cnt}", "{o_is_rep0_long}", "32"),
                "cmp    {state:w}, #7",
                "mov    {t:w}, #9",
                "mov    {u:w}, #11",
                "csel   {state:w}, {t:w}, {u:w}, lo",
                "mov    {cnt:w}, #1",
                "b      50f",
                "32:",
                branch_bit_one!("{cnt}", "{o_is_rep0_long}"),
                "b      33f",
                "31:",
                branch_bit_one!("{step}", "{r_is_rep_g0}"),
                branch_bit!("{step}", "{r_is_rep_g1}", "35"),
                // The second distance, moved to the front. The distances
                // after the first live in memory, used too rarely to hold
                // registers.
                "ldr    {tab}, [{p}, #112]",
                "str    {rep0}, [{p}, #112]",
                "mov    {rep0:w}, {tab:w}",
                "b      33f",
                "35:",
                branch_bit_one!("{step}", "{r_is_rep_g1}"),
                branch_bit!("{step}", "{r_is_rep_g2}", "36"),
                // The third distance.
                "ldp    {t}, {tab}, [{p}, #112]",
                "stp    {rep0}, {t}, [{p}, #112]",
                "mov    {rep0:w}, {tab:w}",
                "b      33f",
                "36:",
                branch_bit_one!("{step}", "{r_is_rep_g2}"),
                // The fourth.
                "ldp    {t}, {u}, [{p}, #112]",
                "ldr    {tab}, [{p}, #128]",
                "stp    {rep0}, {t}, [{p}, #112]",
                "str    {u}, [{p}, #128]",
                "mov    {rep0:w}, {tab:w}",
                "33:",
                // A repeat match with a length: the state, the repeat length
                // coder.
                "cmp    {state:w}, #7",
                "mov    {t:w}, #8",
                "mov    {u:w}, #11",
                "csel   {state:w}, {t:w}, {u:w}, lo",
                "add    {tab}, {probs}, #{o_rep_len_coder}",
                "40:",
                // ---- The length in bytes: two choice bits, then the low or
                // mid tree at the position state or the high tree. ----
                branch_bit!("{tab}", "0", "41"),
                "add    {n1}, {tab}, {n1}",
                tree3!("{n1}"),
                "sub    {cnt:w}, {sym:w}, #6",
                "b      45f",
                "41:",
                branch_bit_one!("{tab}", "0"),
                branch_bit!("{tab}", "16", "42"),
                "add    {n1}, {tab}, {n1}",
                "add    {n1}, {n1}, #16",
                tree3!("{n1}"),
                "add    {cnt:w}, {sym:w}, #2",
                "b      45f",
                "42:",
                branch_bit_one!("{tab}", "16"),
                "add    {n1}, {tab}, #512",
                tree8!("{n1}"),
                "sub    {cnt:w}, {sym:w}, #238",
                "45:",
                // A repeat match copies now.
                "tbz    {state:w}, #4, 50f",
                // ---- A fresh distance: the slot tree of the length state.
                // The length waits in the state block meanwhile. ----
                "str    {cnt}, [{p}, #152]",
                "sub    {t:w}, {cnt:w}, #2",
                "cmp    {t:w}, #3",
                "mov    {u:w}, #3",
                "csel   {t:w}, {t:w}, {u:w}, lo",
                "add    {n1}, {probs}, #{o_pos_slot}",
                "add    {n1}, {n1}, {t}, lsl #7",
                tree6!("{n1}"),
                "sub    {sym:w}, {sym:w}, #64",
                "cmp    {sym:w}, #4",
                "b.lo   48f",
                // The slot's direct bits: how many, and the two bits on top.
                "lsr    {cnt:w}, {sym:w}, #1",
                "sub    {cnt:w}, {cnt:w}, #1",
                "and    {tab:w}, {sym:w}, #1",
                "orr    {tab:w}, {tab:w}, #2",
                "cmp    {sym:w}, #14",
                "b.hs   46f",
                // The special positions: a reverse tree from the distance
                // plus one, with a step that doubles, walked as many bits as
                // the slot has, the children loaded ahead but for the last.
                "lsl    {tab:w}, {tab:w}, {cnt:w}",
                "add    {sym:w}, {tab:w}, #1",
                "mov    {step:w}, #1",
                "ldrh   {prob:w}, [{probs}, {sym}, lsl #1]",
                "47:",
                normalize!(),
                "subs   {cnt:w}, {cnt:w}, #1",
                "add    {n0}, {sym}, {step}",
                "add    {step}, {step}, {step}",
                "add    {n1}, {sym}, {step}",
                "csel   {t}, xzr, {n0}, eq",
                "ldrh   {v:w}, [{probs}, {t}, lsl #1]",
                "csel   {t}, xzr, {n1}, eq",
                "ldrh   {p1:w}, [{probs}, {t}, lsl #1]",
                decide!(),
                "strh   {u:w}, [{probs}, {sym}, lsl #1]",
                "csel   {sym}, {n1}, {n0}, hs",
                "csel   {prob:w}, {p1:w}, {v:w}, hs",
                "cbnz   {cnt:w}, 47b",
                "sub    {tab:w}, {sym:w}, {step:w}",
                "b      49f",
                "46:",
                // The direct bits, each a halving of the range: the first
                // eight check for a normalisation before each bit, and the
                // rest go eight to a normalisation, as a normalised range
                // stands eight halvings. Then the four align bits, a reverse
                // tree from node 1.
                "sub    {cnt:w}, {cnt:w}, #4",
                direct_checked!(),
                direct_checked!(),
                direct_checked!(),
                direct_checked!(),
                direct_checked!(),
                direct_checked!(),
                direct_checked!(),
                direct_checked!(),
                "44:",
                "lsl    {range:w}, {range:w}, #{shift_bits}",
                "ldrb   {t:w}, [{pos}], #1",
                "orr    {code:w}, {t:w}, {code:w}, lsl #{shift_bits}",
                direct_bit!(),
                direct_bit!(),
                direct_bit!(),
                direct_bit!(),
                direct_bit!(),
                direct_bit!(),
                direct_bit!(),
                direct_bit!(),
                "b      44b",
                "43:",
                "lsl    {tab:w}, {tab:w}, #4",
                "add    {cnt}, {probs}, #{o_align}",
                "mov    {sym:w}, #1",
                "mov    {step:w}, #1",
                "ldrh   {prob:w}, [{cnt}, #2]",
                reverse_step!("{cnt}"),
                reverse_step!("{cnt}"),
                reverse_step!("{cnt}"),
                reverse_last_step!("{cnt}"),
                "sub    {t:w}, {sym:w}, #16",
                "orr    {tab:w}, {tab:w}, {t:w}",
                "b      49f",
                "48:",
                "mov    {tab:w}, {sym:w}",
                "49:",
                // The distances rotate, the state settles, the length comes
                // back, and the distance is checked for the end marker and
                // against the window's fill, which is the position until the
                // window first wraps.
                "ldp    {t}, {u}, [{p}, #112]",
                "stp    {rep0}, {t}, [{p}, #112]",
                "str    {u}, [{p}, #128]",
                "add    {rep0:w}, {tab:w}, #1",
                "cmp    {state:w}, #23",
                "mov    {t:w}, #7",
                "mov    {u:w}, #10",
                "csel   {state:w}, {t:w}, {u:w}, lo",
                "ldr    {cnt}, [{p}, #152]",
                "cbz    {rep0:w}, 93f",
                "ldr    {t}, [{p}, #80]",
                "cmp    {t}, {dpos}",
                "csel   {t}, {t}, {dpos}, hi",
                "cmp    {tab}, {t}",
                "b.hs   92f",
                // ---- The copy of `cnt` bytes from `rep0` back. ----
                "50:",
                // A copy past the limit is Rust's. A source before the
                // window's start wraps to its end, unless it would run within
                // a word of the end, which is Rust's too.
                "add    {t}, {dpos}, {cnt}",
                "cmp    {t}, {dlim}",
                "b.hi   91f",
                "subs   {n0}, {dpos}, {rep0}",
                "mov    {u}, {rep0}",
                "b.hs   56f",
                "ldr    {u}, [{p}, #88]",
                "add    {n0}, {n0}, {u}",
                "add    {sym}, {n0}, {cnt}",
                "add    {sym}, {sym}, #8",
                "cmp    {sym}, {u}",
                "b.hi   91f",
                "sub    {u}, {u}, {rep0}",
                "56:",
                // `u` is how far the source lies from the destination in
                // memory: a word or more, and the copy goes by words.
                "add    {n0}, {dic}, {n0}",
                "add    {n1}, {dic}, {dpos}",
                "mov    {dpos}, {t}",
                "cmp    {u}, #16",
                "b.lo   55f",
                "cmp    {cnt}, #16",
                "b.lo   53f",
                // Sixteen bytes at a time, the last sixteen overlapping the
                // ones before, so that exactly the match is written. Every
                // double word read lies a whole one behind the write position.
                "sub    {t}, {cnt}, #16",
                "add    {step}, {n0}, {t}",
                "add    {sym}, {n1}, {t}",
                "51:",
                "ldr    {q:q}, [{n0}], #16",
                "subs   {t}, {t}, #16",
                "str    {q:q}, [{n1}], #16",
                "b.hs   51b",
                "ldr    {q:q}, [{step}]",
                "str    {q:q}, [{sym}]",
                "b      59f",
                "53:",
                // Fewer than sixteen: a word, then a masked word with the
                // destination's own bytes past the match kept, when the
                // window has that word of room.
                // The match's last byte, the next literal's context, is read
                // from the source here, which the copy does not reach: the
                // source lies a word or more behind and the rest is shorter
                // than a word. The other copies read it back from the window.
                "sub    {sym}, {cnt}, #1",
                "ldrb   {sym:w}, [{n0}, {sym}]",
                "cmp    {cnt}, #8",
                "b.lo   54f",
                "ldr    {t}, [{n0}], #8",
                "str    {t}, [{n1}], #8",
                "subs   {cnt}, {cnt}, #8",
                "b.eq   58f",
                "54:",
                "ldr    {step}, [{p}, #88]",
                "add    {t}, {dpos}, #8",
                "cmp    {t}, {step}",
                "b.hi   55f",
                "ldr    {t}, [{n0}]",
                "ldr    {u}, [{n1}]",
                "lsl    {step}, {cnt}, #3",
                "mov    {v}, #-1",
                "lsl    {v}, {v}, {step}",
                "and    {u}, {u}, {v}",
                "bic    {t}, {t}, {v}",
                "orr    {t}, {t}, {u}",
                "str    {t}, [{n1}]",
                "b      58f",
                "55:",
                // A source within sixteen bytes of the destination, or a copy
                // at the window's end: a byte at a time, as the source may be
                // the pattern that repeats.
                "ldrb   {t:w}, [{n0}], #1",
                "strb   {t:w}, [{n1}], #1",
                "subs   {cnt}, {cnt}, #1",
                "b.ne   55b",
                "mov    {sym}, {t}",
                "b      58f",
                "59:",
                "sub    {t}, {dpos}, #1",
                "ldrb   {sym:w}, [{dic}, {t}]",
                "58:",
                "orr    {sym}, {sym}, #0x100",
                next_symbol!("{dpos}"),
                "ldrh   {prob:w}, [{step}, {n1}]",
                // ---- After a match: the state is 7 or more, and a literal
                // is decoded against the byte at the last distance, which
                // may lie before the window's start and wrap. ----
                "16:",
                "cmp    {dpos}, {dlim}",
                "b.hs   90f",
                "cmp    {pos}, {stop}",
                "b.hi   90f",
                is_match_bit!("20b"),
                literal_coder!(),
                "cmp    {state:w}, #10",
                "sub    {t:w}, {state:w}, #3",
                "sub    {u:w}, {state:w}, #6",
                "csel   {state:w}, {t:w}, {u:w}, lo",
                "sub    {t}, {dpos}, {rep0}",
                "ldr    {u}, [{p}, #88]",
                "add    {u}, {t}, {u}",
                "cmp    {dpos}, {rep0}",
                "csel   {t}, {u}, {t}, lo",
                "ldrb   {cnt:w}, [{dic}, {t}]",
                // The first bit's probability, and the second's two, which
                // the first bit alone decides, loaded outright.
                "lsl    {cnt}, {cnt}, #8",
                "mov    {n0}, #0x100",
                ml_bit!("7"),
                "add    {t}, {n1}, #0x101",
                "ldrh   {prob:w}, [{tab}, {t}, lsl #1]",
                "mov    {sym}, #1",
                "eor    {u}, {n0}, {n1}",
                "and    {t}, {n0}, {cnt}, lsr #6",
                "and    {v}, {u}, {t}",
                "add    {v}, {v}, {u}",
                "add    {v}, {v}, #2",
                "ldrh   {rep1:w}, [{tab}, {v}, lsl #1]",
                "and    {v}, {n1}, {t}",
                "add    {v}, {v}, {n1}",
                "add    {v}, {v}, #3",
                "ldrh   {rep2:w}, [{tab}, {v}, lsl #1]",
                ml_loads!("14", "5", "{p1:w}", "{step:w}"),
                ml_decide!(),
                ml_next!("{rep1:w}", "{rep2:w}"),
                ml_bit!("6"),
                ml_pair!("{p1:w}", "{step:w}"),
                ml_loads!("13", "4", "{rep1:w}", "{rep2:w}"),
                ml_decide!(),
                ml_next!("{p1:w}", "{step:w}"),
                ml_bit!("5"),
                ml_pair!("{rep1:w}", "{rep2:w}"),
                ml_loads!("12", "3", "{p1:w}", "{step:w}"),
                ml_decide!(),
                ml_next!("{rep1:w}", "{rep2:w}"),
                ml_bit!("4"),
                ml_pair!("{p1:w}", "{step:w}"),
                ml_loads!("11", "2", "{rep1:w}", "{rep2:w}"),
                ml_decide!(),
                ml_next!("{p1:w}", "{step:w}"),
                ml_bit!("3"),
                ml_pair!("{rep1:w}", "{rep2:w}"),
                ml_loads!("10", "1", "{p1:w}", "{step:w}"),
                ml_decide!(),
                ml_next!("{rep1:w}", "{rep2:w}"),
                ml_bit!("2"),
                ml_pair!("{p1:w}", "{step:w}"),
                ml_loads!("9", "0", "{rep1:w}", "{rep2:w}"),
                ml_decide!(),
                ml_next!("{p1:w}", "{step:w}"),
                ml_bit!("1"),
                ml_pair!("{rep1:w}", "{rep2:w}"),
                ml_decide!(),
                ml_next!("{rep1:w}", "{rep2:w}"),
                ml_bit!("0"),
                ml_decide!(),
                "adc    {sym:w}, {sym:w}, {sym:w}",
                "add    {n0}, {dpos}, #1",
                next_symbol!("{n0}"),
                "strb   {sym:w}, [{dic}, {dpos}]",
                "add    {dpos}, {dpos}, #1",
                "ldrh   {prob:w}, [{step}, {n1}]",
                "b      15b",
                // ---- Exits: the state back into memory, and why. ----
                "91:",
                "mov    {t:w}, #1",
                "b      95f",
                "92:",
                "mov    {t:w}, #2",
                "b      95f",
                "93:",
                "mov    {t:w}, #3",
                "b      95f",
                "90:",
                "mov    {t:w}, #0",
                "95:",
                "ldr    {p}, [sp], #16",
                "str    {pos}, [{p}, #16]",
                "stp    {range}, {code}, [{p}, #40]",
                "str    {dpos}, [{p}, #64]",
                "str    {state}, [{p}, #96]",
                "str    {rep0}, [{p}, #104]",
                "and    {sym}, {sym}, #0xFF",
                "str    {sym}, [{p}, #136]",
                "stp    {cnt}, {t}, [{p}, #152]",
                p = inout(reg) &mut run as *mut Run => _,
                probs = out(reg) _,
                pos = out(reg) _,
                dlim = out(reg) _,
                range = out(reg) _,
                code = out(reg) _,
                dic = out(reg) _,
                dpos = out(reg) _,
                masks = out(reg) _,
                state = out(reg) _,
                rep0 = out(reg) _,
                rep1 = out(reg) _,
                rep2 = out(reg) _,
                stop = out(reg) _,
                pbm = out(reg) _,
                pim = out(reg) _,
                prob = out(reg) _,
                p1 = out(reg) _,
                sym = out(reg) _,
                cnt = out(reg) _,
                step = out(reg) _,
                n0 = out(reg) _,
                n1 = out(reg) _,
                tab = out(reg) _,
                tab2 = out(reg) _,
                t = out(reg) _,
                u = out(reg) _,
                v = out(reg) _,
                q = out(vreg) _,
                shift_bits = const SHIFT_BITS,
                total_bits = const BIT_MODEL_TOTAL_BITS,
                move_bits = const MOVE_BITS,
                offset = const BIT_MODEL_OFFSET,
                o_is_match = const IS_MATCH * 2,
                r_is_rep = const (IS_REP - IS_MATCH) * 2,
                r_is_rep_g0 = const (IS_REP_G0 - IS_MATCH) * 2,
                r_is_rep_g1 = const (IS_REP_G1 - IS_MATCH) * 2,
                r_is_rep_g2 = const (IS_REP_G2 - IS_MATCH) * 2,
                o_is_rep0_long = const IS_REP0_LONG * 2,
                o_len_coder = const LEN_CODER * 2,
                o_rep_len_coder = const REP_LEN_CODER * 2,
                o_pos_slot = const POS_SLOT * 2,
                o_align = const ALIGN * 2,
            );
        }

        // SAFETY: the kernel moves the position within the buffer and to at
        // most one past its last byte.
        self.pos = unsafe { run.pos.offset_from(run.buf) } as usize;
        self.range = run.range as u32;
        self.code = run.code as u32;
        w.set_pos(run.dic_pos as usize);
        st.state = run.state as u32;
        st.reps = run.reps.map(|r| r as u32);
        st.previous = run.previous as u32;
        st.pending = run.len as u32;
        match run.exit {
            0 => Exit::Limit,
            1 => Exit::Copy,
            2 => Exit::Overflow,
            _ => Exit::EndMarker,
        }
    }
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
        let probs = vec![PROB_INIT; NUM_BASE_PROBS + (LIT_STRIDE << (lc + lp))];
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
        // The state of the run in locals; the decoder and the window take it
        // back at the end.
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
            lp: self.lp,
            lp_mask: (0x100u32 << self.lp) - (0x100u32 >> self.lc),
            pb_mask: (1u32 << self.pb) - 1,
            end_marker: false,
            pending: 0,
        };
        let probs = &mut self.probs[..];
        let mut result = Ok(());
        if rc.inner().is_buffer() {
            // The symbols a whole one of which is sure to be in the buffer,
            // through the kernel; the last few bytes are left to the loop
            // below.
            let state = rc.state();
            let mut coder = Coder {
                range: state.range,
                code: state.code,
                pos: rc.inner().pos(),
                buf: rc.inner().buf(),
            };
            while w.has_space() && coder.has_symbol() {
                match coder.run(probs, &mut w, &mut st) {
                    Exit::Limit => break,
                    Exit::Copy => {
                        if let Err(error) = w.repeat((st.reps[0] - 1) as usize, st.pending as usize)
                        {
                            result = Err(error);
                            break;
                        }
                        st.previous = u32::from(w.get_byte(0));
                    }
                    Exit::Overflow => {
                        result = Err(error_other("dist overflow"));
                        break;
                    }
                    Exit::EndMarker => {
                        st.end_marker = true;
                        result = Err(error_other("dist overflow"));
                        break;
                    }
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

    /// A bit tree of `probs.len()` leaves, a power of two; the leaf. Index 0
    /// is not used.
    #[inline(always)]
    fn tree<R: RangeReader>(rc: &mut RangeDecoder<R>, probs: &mut [u16]) -> u32 {
        let limit = probs.len();
        let mut i = 1usize;
        while i < limit {
            // The mask is a no-op on the index and lets the bounds check go.
            i = (i << 1) | rc.decode_bit(&mut probs[i & (limit - 1)]) as usize;
        }
        (i - limit) as u32
    }

    /// A literal against `match_byte`, over the 0x300 probabilities of one
    /// literal coder, the way `LzmaDec.c` indexes them.
    #[inline(always)]
    fn matched_literal<R: RangeReader>(
        rc: &mut RangeDecoder<R>,
        probs: &mut [u16],
        match_byte: u32,
    ) -> u32 {
        let mut match_byte = match_byte;
        let mut offs = 0x100u32;
        let mut symbol = 1u32;
        while symbol < 0x100 {
            match_byte += match_byte;
            let bit = offs;
            offs &= match_byte;
            let decoded = rc.decode_bit(&mut probs[(offs + bit + symbol) as usize]) as u32;
            symbol = (symbol << 1) | decoded;
            if decoded == 0 {
                offs ^= bit;
            }
        }
        symbol - 0x100
    }

    /// `count` bits of a reverse bit tree from node `start`, walked as 7-Zip's
    /// `REV_BIT` walks it: each bit moves to `node + step` for a 0 or `node +
    /// 2 * step` for a 1, and the step doubles. Returns the node reached; the
    /// value is that less `1 << count`.
    #[inline(always)]
    fn reverse<R: RangeReader>(
        rc: &mut RangeDecoder<R>,
        probs: &mut [u16],
        start: u32,
        count: u32,
    ) -> u32 {
        let mut node = start;
        let mut step = 1u32;
        for _ in 0..count {
            let bit = rc.decode_bit(&mut probs[node as usize]) as u32;
            step += step;
            node += if bit == 0 { step >> 1 } else { step };
        }
        node
    }

    /// One LZMA symbol, as 7-Zip's `LzmaDec_DecodeReal` decodes it, into
    /// the window, over any reader.
    #[inline(always)]
    fn decode_symbol<R: RangeReader>(
        probs: &mut [u16],
        lz: &mut WindowParts<'_>,
        st: &mut Symbols,
        rc: &mut RangeDecoder<R>,
    ) -> crate::Result<()> {
        let pos_state = (lz.get_pos() as u32 & st.pb_mask) as usize;
        let mut state = st.state as usize;
        if rc.decode_bit(&mut probs[IS_MATCH + (pos_state << NUM_POS_BITS_MAX) + state]) == 0 {
            let context = (((lz.get_pos() as u32) << 8) + st.previous) & st.lp_mask;
            let base = LITERAL + (((context << st.lc) as usize) << 2);
            let symbol = if state < NUM_LIT_STATES {
                state -= if state < 4 { state } else { 3 };
                Self::tree(rc, &mut probs[base..base + 0x100])
            } else {
                let match_byte = u32::from(lz.get_byte((st.reps[0] - 1) as usize));
                state -= if state < 10 { 3 } else { 6 };
                Self::matched_literal(rc, &mut probs[base..base + LIT_SIZE], match_byte)
            };
            st.state = state as u32;
            st.previous = symbol;
            lz.put_byte(symbol as u8);
            return Ok(());
        }
        let len_base = if rc.decode_bit(&mut probs[IS_REP + state]) == 0 {
            state += NUM_STATES;
            LEN_CODER
        } else {
            if rc.decode_bit(&mut probs[IS_REP_G0 + state]) == 0 {
                if rc.decode_bit(&mut probs[IS_REP0_LONG + (pos_state << NUM_POS_BITS_MAX) + state])
                    == 0
                {
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
                let distance = if rc.decode_bit(&mut probs[IS_REP_G1 + state]) == 0 {
                    st.reps[1]
                } else {
                    let distance = if rc.decode_bit(&mut probs[IS_REP_G2 + state]) == 0 {
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
            let mut distance = Self::tree(rc, &mut probs[slot..slot + (1 << NUM_POS_SLOT_BITS)]);
            if distance >= START_POS_MODEL_INDEX {
                let pos_slot = distance;
                let mut direct_bits = (distance >> 1) - 1;
                distance = 2 | (distance & 1);
                if pos_slot < END_POS_MODEL_INDEX {
                    distance <<= direct_bits;
                    let node = Self::reverse(
                        rc,
                        &mut probs[SPEC_POS..SPEC_POS + NUM_FULL_DISTANCES],
                        distance + 1,
                        direct_bits,
                    );
                    distance = node - (1 << direct_bits);
                } else {
                    direct_bits -= NUM_ALIGN_BITS;
                    distance =
                        (distance << direct_bits) | rc.decode_direct_bits(direct_bits) as u32;
                    distance <<= NUM_ALIGN_BITS;
                    let node = Self::reverse(
                        rc,
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
    fn decode_len<R: RangeReader>(
        probs: &mut [u16],
        rc: &mut RangeDecoder<R>,
        base: usize,
        pos_state: usize,
    ) -> u32 {
        let len_state = pos_state << (LEN_NUM_LOW_BITS + 1);
        if rc.decode_bit(&mut probs[base + LEN_CHOICE]) == 0 {
            let low = base + LEN_LOW + len_state;
            Self::tree(rc, &mut probs[low..low + LEN_NUM_LOW_SYMBOLS])
        } else if rc.decode_bit(&mut probs[base + LEN_CHOICE2]) == 0 {
            let mid = base + LEN_LOW + len_state + LEN_NUM_LOW_SYMBOLS;
            LEN_NUM_LOW_SYMBOLS as u32 + Self::tree(rc, &mut probs[mid..mid + LEN_NUM_LOW_SYMBOLS])
        } else {
            let high = base + LEN_HIGH;
            2 * LEN_NUM_LOW_SYMBOLS as u32
                + Self::tree(rc, &mut probs[high..high + LEN_NUM_HIGH_SYMBOLS])
        }
    }
}
