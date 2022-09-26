/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

#define numResidual 2

/**
 * Convolutional Layer L0:
 *      IFM  =    32  IFM_CH =     3
 *      OFM  =    30  OFM_CH =    16
 *     SIMD  =     3    PE   =     4
 *     WMEM  =    36   TMEM  =     4
 *     #Ops  = 777600   Est Latency  = 32400
**/

#define L0_K 3
#define L0_IFM_CH 3
#define L0_IFM_DIM 32
#define L0_OFM_CH 16
#define L0_OFM_DIM 30
#define L0_SIMD 3
#define L0_PE 4
#define L0_WMEM 36
#define L0_TMEM 4
/**
 * Convolutional Layer L1:
 *      IFM  =    30  IFM_CH =    16
 *      OFM  =    28  OFM_CH =    16
 *     SIMD  =    16    PE   =     4
 *     WMEM  =    36   TMEM  =     4
 *     #Ops  = 3612672   Est Latency  = 28224
**/

#define L1_K 3
#define L1_IFM_CH 16
#define L1_IFM_DIM 30
#define L1_OFM_CH 16
#define L1_OFM_DIM 28
#define L1_SIMD 16
#define L1_PE 4
#define L1_WMEM 36
#define L1_TMEM 4
/**
 * Convolutional Layer L2:
 *      IFM  =    14  IFM_CH =    16
 *      OFM  =    12  OFM_CH =    32
 *     SIMD  =    16    PE   =     4
 *     WMEM  =    72   TMEM  =     8
 *     #Ops  = 1327104   Est Latency  = 10368
**/

#define L2_K 3
#define L2_IFM_CH 16
#define L2_IFM_DIM 14
#define L2_OFM_CH 32
#define L2_OFM_DIM 12
#define L2_SIMD 16
#define L2_PE 4
#define L2_WMEM 72
#define L2_TMEM 8
/**
 * Convolutional Layer L3:
 *      IFM  =    12  IFM_CH =    32
 *      OFM  =    10  OFM_CH =    32
 *     SIMD  =    32    PE   =     4
 *     WMEM  =    72   TMEM  =     8
 *     #Ops  = 1843200   Est Latency  =  7200
**/

#define L3_K 3
#define L3_IFM_CH 32
#define L3_IFM_DIM 12
#define L3_OFM_CH 32
#define L3_OFM_DIM 10
#define L3_SIMD 32
#define L3_PE 4
#define L3_WMEM 72
#define L3_TMEM 8
/**
 * Convolutional Layer L4:
 *      IFM  =     5  IFM_CH =    32
 *      OFM  =     3  OFM_CH =    64
 *     SIMD  =    32    PE   =     1
 *     WMEM  =   576   TMEM  =    64
 *     #Ops  = 331776   Est Latency  =  5184
**/

#define L4_K 3
#define L4_IFM_CH 32
#define L4_IFM_DIM 5
#define L4_OFM_CH 64
#define L4_OFM_DIM 3
#define L4_SIMD 32
#define L4_PE 1
#define L4_WMEM 576
#define L4_TMEM 64
/**
 * Fully-Connected Layer L5:
 *     MatW =   576 MatH =   128
 *     SIMD =    16  PE  =     1
 *     WMEM =  4608 TMEM =   128
 *     #Ops  = 147456   Est Latency  =  4608
**/

#define L5_SIMD 16
#define L5_PE 1
#define L5_WMEM 4608
#define L5_TMEM 128
#define L5_MW 576
#define L5_MH 128
/**
 * Fully-Connected Layer L6:
 *     MatW =   128 MatH =     4
 *     SIMD =     1  PE  =     1
 *     WMEM =   512 TMEM =     4
 *     #Ops  =  1024   Est Latency  =   512
**/

#define L6_SIMD 1
#define L6_PE 1
#define L6_WMEM 512
#define L6_TMEM 4
#define L6_MW 128
#define L6_MH 4
#endif //__LAYER_CONFIG_H_
