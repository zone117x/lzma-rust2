/* Every function of the SDK decoder and of the shim, renamed, so that two
 * builds of the decoder (plain C, and with the assembly inner loop) can be
 * linked into one benchmark binary. */
#ifndef BENCH_RENAME_SDKC_
#define BENCH_RENAME_SDKC_
#define Lzma2Dec_Allocate SdkC_Lzma2Dec_Allocate
#define Lzma2Dec_AllocateProbs SdkC_Lzma2Dec_AllocateProbs
#define Lzma2Dec_DecodeToBuf SdkC_Lzma2Dec_DecodeToBuf
#define Lzma2Dec_DecodeToDic SdkC_Lzma2Dec_DecodeToDic
#define Lzma2Dec_Init SdkC_Lzma2Dec_Init
#define Lzma2Dec_Parse SdkC_Lzma2Dec_Parse
#define Lzma2Decode SdkC_Lzma2Decode
#define LzmaDec_Allocate SdkC_LzmaDec_Allocate
#define LzmaDec_AllocateProbs SdkC_LzmaDec_AllocateProbs
#define LzmaDec_DecodeToBuf SdkC_LzmaDec_DecodeToBuf
#define LzmaDec_DecodeToDic SdkC_LzmaDec_DecodeToDic
#define LzmaDec_Free SdkC_LzmaDec_Free
#define LzmaDec_FreeProbs SdkC_LzmaDec_FreeProbs
#define LzmaDec_Init SdkC_LzmaDec_Init
#define LzmaDec_InitDicAndState SdkC_LzmaDec_InitDicAndState
#define LzmaDecode SdkC_LzmaDecode
#define LzmaProps_Decode SdkC_LzmaProps_Decode
#define bench_lzma2_decode SdkC_bench_lzma2_decode
#define bench_lzma_decode SdkC_bench_lzma_decode
#define LzmaDec_DecodeReal_3 SdkC_LzmaDec_DecodeReal_3
#endif
