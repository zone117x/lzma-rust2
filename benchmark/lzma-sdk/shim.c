/* One-shot decoding through the SDK's public API, for the benchmark. */

#include <stdlib.h>

#include "Lzma2Dec.h"
#include "LzmaDec.h"

static void *SzAlloc(ISzAllocPtr p, size_t size)
{
  (void)p;
  return malloc(size);
}

static void SzFree(ISzAllocPtr p, void *address)
{
  (void)p;
  free(address);
}

static const ISzAlloc g_alloc = { SzAlloc, SzFree };

/* A raw LZMA stream with its five property bytes, into a buffer of exactly
   the uncompressed size. Returns SZ_OK (0) on success. */
int bench_lzma_decode(unsigned char *dest, size_t *dest_len, const unsigned char *src, size_t *src_len,
    const unsigned char *props)
{
  ELzmaStatus status;
  return LzmaDecode(dest, dest_len, src, src_len, props, LZMA_PROPS_SIZE, LZMA_FINISH_END, &status, &g_alloc);
}

/* A raw LZMA2 stream with its dictionary property byte, likewise. */
int bench_lzma2_decode(unsigned char *dest, size_t *dest_len, const unsigned char *src, size_t *src_len,
    unsigned char prop)
{
  ELzmaStatus status;
  return Lzma2Decode(dest, dest_len, src, src_len, prop, LZMA_FINISH_END, &status, &g_alloc);
}
