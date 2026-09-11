/* Every function of the SDK decoder and of the shim, renamed, so that two
 * builds of the decoder (plain C, and with the assembly inner loop) can be
 * linked into one benchmark binary. */
#ifndef BENCH_RENAME_SDKASM_
#define BENCH_RENAME_SDKASM_
#define Lzma2Dec_Allocate SdkAsm_Lzma2Dec_Allocate
#define Lzma2Dec_AllocateProbs SdkAsm_Lzma2Dec_AllocateProbs
#define Lzma2Dec_DecodeToBuf SdkAsm_Lzma2Dec_DecodeToBuf
#define Lzma2Dec_DecodeToDic SdkAsm_Lzma2Dec_DecodeToDic
#define Lzma2Dec_Init SdkAsm_Lzma2Dec_Init
#define Lzma2Dec_Parse SdkAsm_Lzma2Dec_Parse
#define Lzma2Decode SdkAsm_Lzma2Decode
#define LzmaDec_Allocate SdkAsm_LzmaDec_Allocate
#define LzmaDec_AllocateProbs SdkAsm_LzmaDec_AllocateProbs
#define LzmaDec_DecodeToBuf SdkAsm_LzmaDec_DecodeToBuf
#define LzmaDec_DecodeToDic SdkAsm_LzmaDec_DecodeToDic
#define LzmaDec_Free SdkAsm_LzmaDec_Free
#define LzmaDec_FreeProbs SdkAsm_LzmaDec_FreeProbs
#define LzmaDec_Init SdkAsm_LzmaDec_Init
#define LzmaDec_InitDicAndState SdkAsm_LzmaDec_InitDicAndState
#define LzmaDecode SdkAsm_LzmaDecode
#define LzmaProps_Decode SdkAsm_LzmaProps_Decode
#define bench_lzma2_decode SdkAsm_bench_lzma2_decode
#define bench_lzma_decode SdkAsm_bench_lzma_decode
#endif
