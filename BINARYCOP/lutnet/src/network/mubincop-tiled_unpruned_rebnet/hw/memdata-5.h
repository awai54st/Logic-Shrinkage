/*
Weight and threshold memory initialization for Vivado HLS
PEs = 1, SIMD width = 16, threshold bits = 28
weight mem depth = 4608, thres mem depth = 128, alpha mem depth = 128
layer sizes (neurons, synapses per neuron): 
(128, 576) 
padded neurons for each layer: 
0 
padded synapses for each layer: 
0 
*/

static ap_uint<16> weightMem5[1][4608] = {
{
0x9388,
0xf2c8,
0x48a9,
0x98ae,
0x170e,
0x65d8,
0xcbe8,
0x41ac,
0x760a,
0x13d4,
0xa391,
0x8ba,
0xd229,
0x5b48,
0x1b8,
0x4afc,
0x72a,
0x85c7,
0x81fc,
0x4afc,
0x3702,
0xc25f,
0x8112,
0x37e,
0x762a,
0xab04,
0x21ba,
0xc2fe,
0x16ae,
0x8b57,
0xe99c,
0x877c,
0x371b,
0xcb1e,
0xcc31,
0x83fe,
0xbff,
0xc52b,
0x6268,
0xbe65,
0x339f,
0x7371,
0x50d9,
0x6e71,
0x209e,
0x6b71,
0xc4eb,
0x1573,
0x83e5,
0xe669,
0x1aa5,
0xe58d,
0x7ec,
0x6779,
0xd23e,
0xe365,
0x369c,
0x1b51,
0x94eb,
0x7377,
0x8be5,
0x74fb,
0xdac5,
0xe585,
0xe7fc,
0x7771,
0x1263,
0x69f5,
0x744c,
0x737d,
0xe2eb,
0x6973,
0x36fe,
0x7337,
0xe662,
0xdf19,
0xb6fe,
0x5b33,
0x434f,
0x8718,
0x1a3e,
0x53da,
0xc46f,
0x5d54,
0x5fce,
0xbb3d,
0xe614,
0x8f0f,
0x2f4c,
0xb97d,
0xe9b,
0x1d57,
0x3216,
0xbb1f,
0x944d,
0x151c,
0x54ba,
0x572a,
0x636a,
0x8fde,
0x7780,
0x6193,
0xe5c2,
0x86d9,
0x54ba,
0x6353,
0xc7eb,
0x793,
0xb51,
0xa4e2,
0x764d,
0xa4c5,
0x6b81,
0x2829,
0xbc49,
0x56e7,
0x2353,
0x3cec,
0x2a8d,
0x5f85,
0xa971,
0x3aa5,
0x4c6e,
0xe291,
0xd965,
0x568f,
0xda5e,
0xaaf5,
0xcde0,
0x85,
0x9994,
0x2a23,
0xbd0e,
0x7b8c,
0xa46e,
0xc5fc,
0xdcde,
0x7f6d,
0x1a6a,
0xe1f7,
0x954f,
0x366b,
0xe64d,
0x7f45,
0x7d40,
0x9a95,
0x9d06,
0x22f8,
0xb570,
0x7ddf,
0xa8a2,
0xfa,
0xd640,
0x3897,
0xf8e0,
0x909a,
0xb940,
0xa8d1,
0x1513,
0xa158,
0x9944,
0x888a,
0x2c63,
0x41a6,
0x362,
0xa4ab,
0xe3cc,
0x5192,
0x6c40,
0xa915,
0x398c,
0x27a,
0xa0,
0xd91,
0x63ae,
0x93ea,
0x33b,
0xe7b3,
0xcbbe,
0xda8e,
0x1698,
0x937a,
0x1b8,
0x923e,
0x561e,
0xc300,
0xa179,
0x138,
0x923e,
0x2bb6,
0x1a1,
0x802a,
0x763a,
0xab38,
0x3138,
0x26e,
0x763a,
0x8336,
0xe13b,
0xc62e,
0x772f,
0x833c,
0x63d0,
0xd62e,
0x760a,
0xcb1c,
0xd1bc,
0x80dc,
0x322f,
0x9b56,
0xf50f,
0x80dd,
0xaba7,
0x389e,
0x1c95,
0xa04d,
0xe93b,
0xc7a,
0x66eb,
0xee86,
0xa39,
0xa0f8,
0x91ac,
0x6f84,
0xfa,
0x3bf3,
0xf6ed,
0x4e86,
0xa912,
0x2478,
0x3329,
0x7182,
0xf85a,
0x2d38,
0xb805,
0x71be,
0x8972,
0xbc7b,
0xca3c,
0x6d1a,
0xe400,
0x5315,
0xa5b8,
0x12eb,
0xe002,
0xd4d,
0x229e,
0x126a,
0xe911,
0xe855,
0xac16,
0x27e,
0x7e28,
0x42e4,
0x79d5,
0xf4a9,
0x5b6d,
0x47a2,
0x3295,
0x84e9,
0x5368,
0xe3ae,
0xe0b1,
0xf4ed,
0xabec,
0x8faa,
0x3894,
0xb7a9,
0x36e1,
0x8d69,
0xb5c4,
0x8069,
0x21e9,
0xafec,
0xb0ac,
0xa2c9,
0xcb64,
0x8ac6,
0xf095,
0xb4b9,
0x5f52,
0x8c86,
0xfdd5,
0xa6e8,
0xcba0,
0xb86,
0xbf94,
0xb4e9,
0x108f,
0x857f,
0xc302,
0x4c14,
0x309f,
0xc051,
0x4722,
0xe12,
0xf7,
0x85fb,
0x8e2e,
0x932,
0x729f,
0x56f9,
0xc62c,
0x5d44,
0x69c,
0x57dd,
0x32ba,
0x7d1e,
0x329f,
0x75d1,
0xae3,
0x5936,
0x49e,
0x377b,
0x826a,
0xd94e,
0x14ae,
0x73fd,
0x526f,
0x5d1f,
0x46e,
0x7779,
0xc24b,
0x1b56,
0xa4d6,
0xfc9b,
0x9e22,
0xb16,
0xe4d2,
0x7457,
0xac2a,
0x2356,
0xac94,
0x745d,
0x1d1e,
0x2b14,
0xfd10,
0xb8d5,
0x6f16,
0x176,
0xa95e,
0xf08c,
0x6f33,
0x3912,
0xec00,
0x389f,
0x4913,
0x2936,
0xed5b,
0xbc33,
0x1d2a,
0x37e,
0xe110,
0xd813,
0xed2e,
0xbaba,
0xb916,
0x7c19,
0xbc3a,
0x4233,
0xf402,
0x7cdf,
0x8d02,
0xb56,
0xe452,
0x745f,
0xa93a,
0x356,
0xa492,
0x540b,
0x981e,
0xa212,
0xdd51,
0x18df,
0x4e9e,
0x4056,
0xe81c,
0x399f,
0x4f75,
0x916,
0x2041,
0x51de,
0x4b90,
0x8916,
0x305b,
0x3d39,
0x5f0a,
0xb5c,
0x2014,
0xf555,
0x612a,
0xb5e,
0xb457,
0xfcdd,
0x882a,
0x4376,
0xb9c1,
0x1cca,
0x4e9c,
0xc3ac,
0xa2d0,
0x2c0b,
0x1ed8,
0x73ad,
0xb541,
0x645f,
0xa818,
0x52f8,
0xf962,
0xc244,
0x6f1e,
0xc3ba,
0x49d9,
0x2ab9,
0x4f47,
0xc0bf,
0x7f40,
0x4011,
0xdc03,
0x40d0,
0x7c00,
0x8b05,
0x25be,
0x12f8,
0xd400,
0x8b05,
0x431c,
0x123a,
0xf440,
0xc21c,
0x89b8,
0x527e,
0xeae0,
0xa4c5,
0x3b14,
0x22b9,
0xe4e1,
0xe4cd,
0x3f74,
0xf2a9,
0xc7e2,
0x548c,
0xbd94,
0xa2ed,
0x6fe1,
0xa8c5,
0xfd9e,
0x9cf6,
0x2ccc,
0xfc0d,
0xb872,
0xe6fc,
0x6c10,
0x709d,
0xa7e0,
0x1ee5,
0xf9f3,
0xcceb,
0x1e57,
0xbeb4,
0xadf2,
0x5293,
0xfc0a,
0xaed6,
0xcbd1,
0x30af,
0x7722,
0x6c81,
0xc7f1,
0x3b62,
0xec9d,
0xf4db,
0x6fb7,
0x1f7a,
0x34e9,
0x7d09,
0x5457,
0x576e,
0x25ad,
0x558d,
0xd7c0,
0xd665,
0x3cdd,
0xb3f3,
0x4db2,
0x1876,
0xe5cd,
0x77e1,
0x75bf,
0x5a6a,
0xfd95,
0x4fff,
0x7b60,
0xb284,
0xbc95,
0xf499,
0x7340,
0xc24,
0xf431,
0xe2a9,
0x43e1,
0xaa2,
0x2595,
0xdc89,
0x334e,
0x8335,
0x10aa,
0x266b,
0x729b,
0xab75,
0xc74b,
0xc64a,
0x381a,
0x1bf9,
0x846b,
0x2d0a,
0x130a,
0x73f5,
0xabb9,
0xd15c,
0x30e,
0x622e,
0x43a8,
0xd158,
0x36be,
0x1aa8,
0x82e1,
0xd173,
0x1e2e,
0x7b68,
0x62fb,
0xc54e,
0x162c,
0x7b6c,
0x1079,
0xd94d,
0x167c,
0x7b7c,
0xc2cb,
0xc35f,
0x875b,
0x9ca2,
0x5e5e,
0xa6c5,
0x89c7,
0xac20,
0x5c56,
0xe1db,
0xe9c5,
0x9c0b,
0x3c1f,
0xa0d7,
0xc02b,
0x3165,
0x3512,
0xee36,
0xedc9,
0x32a4,
0x5717,
0xa435,
0xa9fd,
0x541b,
0x4f16,
0x2b15,
0x54bb,
0x4727,
0x7d78,
0x3eb4,
0xc135,
0x6d8b,
0x66dc,
0xb75c,
0x89d5,
0x648b,
0x1e61,
0xbd01,
0x5b60,
0x8aa4,
0xf49d,
0xf589,
0x5b2c,
0x22a2,
0xee9d,
0xdc3b,
0x5668,
0xa396,
0xe2d5,
0xddea,
0xd2f0,
0xa286,
0x7581,
0xb6a9,
0x57e4,
0xc6e2,
0xbc84,
0xf4ea,
0x77e2,
0xa73e,
0xbe9f,
0xb6c9,
0x52e6,
0x82c4,
0x7095,
0xb4f8,
0x5fe2,
0x87ec,
0xbad5,
0x94e0,
0x4b60,
0x83cc,
0xa199,
0xf4a0,
0x1e1,
0xaab,
0x7a0c,
0xfc85,
0xce9,
0x2ca9,
0x17c4,
0xffcf,
0x4cab,
0x1e6a,
0x5c4e,
0xbfc5,
0x3f5,
0x46f3,
0x8e4f,
0xbc94,
0x8aff,
0x44ab,
0xdf04,
0xfa85,
0x8bed,
0xc8e,
0x7606,
0xaed5,
0x83e5,
0x3c76,
0xdcff,
0x7d87,
0x83f5,
0x1cfb,
0x1e45,
0xec07,
0x8bef,
0xc22,
0x7e47,
0xed85,
0x81df,
0xd51d,
0xc62a,
0x2b56,
0xa056,
0x38d9,
0xcf6a,
0x6336,
0xa896,
0x3819,
0x9c6e,
0x2112,
0x9d07,
0x9c19,
0xce3e,
0x4954,
0x995c,
0x1c5d,
0x462b,
0x156,
0xb216,
0x3c1d,
0x4a17,
0x6116,
0xd58a,
0xb523,
0x9e02,
0x94e,
0xa884,
0xf995,
0x8d0e,
0x95e,
0xe09b,
0x3d39,
0x1c6f,
0x4b57,
0x4d3f,
0x442a,
0x7041,
0xf407,
0xebad,
0x71aa,
0x12e5,
0xdd05,
0x7a7,
0x34e6,
0x526c,
0xf401,
0x1ff,
0xc423,
0x5ec7,
0xfca1,
0xfe5,
0xe473,
0x7a0d,
0xf485,
0x9f5,
0xc4f1,
0x9cc7,
0xbea5,
0xcbe7,
0x7a9,
0x5e47,
0xec95,
0x2feb,
0xa9,
0x5a47,
0xacc7,
0xcbe1,
0x4a3,
0x5647,
0x2cc5,
0x7660,
0x2c4,
0x79d5,
0xb4a9,
0x5fec,
0x83a4,
0x32d5,
0x94a9,
0x56e8,
0x83a4,
0xb2b5,
0xd4ad,
0x63f8,
0xe7e2,
0x30c0,
0xb6a9,
0x62a1,
0x87a3,
0xb9cc,
0xa6ed,
0x5ae9,
0xc7e0,
0xb0cd,
0xa6c9,
0x324,
0x2bc6,
0xe0ed,
0xd4b1,
0x9773,
0xabe4,
0x3af1,
0x96a9,
0x4fe1,
0x8386,
0xab94,
0x9489,
0x30fe,
0x9335,
0x4462,
0xf14,
0xb8bf,
0xb9db,
0x4423,
0xa16,
0x16eb,
0xd1f3,
0x864f,
0x1f50,
0x30ff,
0x33fb,
0xd26b,
0x5b44,
0x16be,
0x725e,
0x72ed,
0xf716,
0x128e,
0x5301,
0x82eb,
0x7536,
0x14ae,
0x7f79,
0x627b,
0xdd4e,
0x143e,
0x777d,
0x32ef,
0xd957,
0x143e,
0x3779,
0xc26b,
0x4357,
0x563a,
0x6730,
0x61b1,
0x9c8c,
0xdc1e,
0xd573,
0x839,
0x3c7c,
0xe557,
0xcd07,
0x4a97,
0x6b2e,
0x46ba,
0x453a,
0xfd6c,
0x7aaf,
0xc0be,
0xdd9,
0x704a,
0x6c85,
0xeddb,
0xdc09,
0x5d1c,
0x2e54,
0x69e3,
0x1d77,
0x7c37,
0xbad5,
0xabc5,
0x7c1f,
0xfe07,
0x20dc,
0xe9c5,
0x1ca9,
0x3e16,
0x2841,
0x920c,
0xdbe0,
0xb8d5,
0x9d18,
0xb1c,
0xcdf0,
0xb968,
0xf5b8,
0x6713,
0x5da8,
0xf9c5,
0xa40c,
0x1223,
0x733,
0x559c,
0x40af,
0x713f,
0x99d,
0x21a9,
0x622f,
0x823b,
0xb7fb,
0x456e,
0xd61e,
0xc3b7,
0x6c7b,
0x1fdd,
0x3605,
0x2dc7,
0xf99b,
0xfcce,
0x2758,
0xf5d4,
0xb0df,
0xbc62,
0x30f1,
0x6e20,
0x6a88,
0x799d,
0xf4a9,
0x1be1,
0x8724,
0x1991,
0xeceb,
0xc7ed,
0xe36e,
0x6ab1,
0xd4ef,
0xa670,
0x6bc0,
0x3cd0,
0xf6a1,
0xdbf1,
0xcfa3,
0xb8d4,
0xeee9,
0x5be1,
0xaa64,
0xb19d,
0xf6e9,
0x6370,
0x2c4,
0xf4f5,
0xf6b1,
0x43f1,
0x4ec,
0xbbf0,
0x56a8,
0x4ba1,
0x4186,
0x2b94,
0xb4a9,
0x562a,
0x5b46,
0xa9b8,
0xd0fc,
0x5630,
0x4b46,
0xedb0,
0x12f8,
0x7502,
0x43f6,
0xa288,
0x40fa,
0x1e9a,
0x3b1c,
0x21aa,
0x2c8,
0x7692,
0x7905,
0x95ec,
0x46fe,
0x5012,
0x735e,
0xa46a,
0x46fe,
0x561a,
0x8314,
0x61ba,
0x2d8,
0x161a,
0x5b56,
0xe12e,
0x435a,
0x361a,
0x735e,
0x81ba,
0x43fa,
0xa0df,
0x34b9,
0x7662,
0x4a44,
0xc8d7,
0x3c3b,
0x667,
0xaac4,
0xbf7,
0x78f3,
0xce6e,
0x2d61,
0xbc7,
0x9cdf,
0xda46,
0x4c54,
0x8b25,
0xb47f,
0x5e03,
0x6d37,
0xb0d,
0x7cfb,
0x5e55,
0x6904,
0xad9f,
0x3cab,
0x5e07,
0x6947,
0x289d,
0x7c29,
0x3a67,
0xdd47,
0x89be,
0x7c6b,
0x7e67,
0x2f47,
0x85d1,
0x64f9,
0x8a3f,
0x9ca3,
0xad0e,
0x54b1,
0xbc6f,
0x2dc7,
0x818d,
0x3ca8,
0x3b57,
0x7cc7,
0xe869,
0xa460,
0x18df,
0x9ca7,
0x2c39,
0x2479,
0x96ee,
0x7d8f,
0x8a8d,
0xaca9,
0x3a46,
0x2ae7,
0xa9c5,
0x84aa,
0x1c07,
0xad87,
0xa0c5,
0x34b7,
0xdf45,
0x2887,
0x89cd,
0x3cab,
0x5e16,
0xa845,
0xacd1,
0x8dd5,
0x8e6e,
0x4b52,
0x2c96,
0x17,
0xd0f,
0x4914,
0xec47,
0xcc13,
0x8d16,
0xb34,
0xf503,
0x9917,
0x4766,
0x6378,
0x741a,
0xd335,
0x3772,
0x1218,
0xe641,
0xfb37,
0x914,
0x33a,
0xfd00,
0xa815,
0x8f34,
0x272,
0xf804,
0xaa15,
0xa39c,
0xa7a,
0xfd14,
0x9d75,
0xd1c,
0x17e,
0x70df,
0xdb71,
0xa566,
0x7e10,
0x7497,
0xe1f5,
0xe76e,
0x6b16,
0x1497,
0x2dd3,
0xe64f,
0x5970,
0x42a,
0x64fc,
0xfe1c,
0x4916,
0x8d7c,
0xae5d,
0x12ef,
0x4116,
0x613b,
0x303f,
0x4a16,
0x6116,
0x44be,
0x617b,
0x274a,
0xf6d,
0x2dac,
0x5513,
0x7d0e,
0x3b9a,
0x4b8,
0x74b9,
0xaa6a,
0xb76,
0x92c9,
0xbaf8,
0x80e2,
0x817e,
0xb20d,
0xaa80,
0xa7ea,
0x47e2,
0xbe8b,
0x8830,
0xd97,
0x217a,
0x924b,
0x83e0,
0x45d0,
0x1a76,
0xf689,
0xf3a6,
0xd538,
0xf6a,
0xeb17,
0x6baa,
0x4392,
0x834b,
0x563a,
0x4320,
0x61e8,
0xd23e,
0x56be,
0x834a,
0xc3c8,
0x953d,
0x12ae,
0xc710,
0xc2c9,
0x853b,
0x81bd,
0xa67a,
0xd46f,
0xbde5,
0x68f,
0x7f6d,
0x566d,
0x8c05,
0x4dea,
0xd2e7,
0x765e,
0xbcc5,
0xa0ef,
0x3efb,
0x9c47,
0xed67,
0x83cf,
0xf82b,
0xfa4c,
0xff47,
0x8bff,
0xbc68,
0x524f,
0xad45,
0x89e5,
0x3cfb,
0xe665,
0xedc7,
0xded,
0xdce9,
0x1e61,
0xed85,
0x89fe,
0x5562,
0x76c5,
0x7d85,
0x363f,
0x133e,
0x31b2,
0xfd75,
0x1f1e,
0x3376,
0x14d1,
0x9d19,
0x1736,
0x13f2,
0xc4ef,
0x7d08,
0x66bf,
0x19a4,
0xf1e7,
0xd645,
0x337,
0x3150,
0xd22c,
0xf50c,
0x23f,
0x35e8,
0x5aad,
0xef05,
0x6bf,
0x57ea,
0xf2c9,
0xdd8f,
0x6fe,
0x1428,
0x5261,
0xfd07,
0x3bf,
0x35e3,
0x56cd,
0xff05,
0x8ff,
0x3471,
0x4a6b,
0x4846,
0x7d,
0x9cd7,
0x9b4e,
0xb746,
0xa4eb,
0x3888,
0x3063,
0xab06,
0xc15a,
0x72f2,
0x530f,
0x641e,
0xcb7f,
0x32ae,
0x48e1,
0xf50f,
0xd356,
0x32be,
0x4222,
0xc102,
0xa792,
0xaffd,
0x1929,
0x16a,
0x8206,
0xbf5f,
0x259c,
0x14b,
0xa03f,
0xbffc,
0xa853,
0x8329,
0x54af,
0x557e,
0x432a,
0x6e05,
0x329f,
0xc977,
0x7743,
0x6a10,
0x16fe,
0x52db,
0x866e,
0x1d30,
0x42bb,
0x53fa,
0xc224,
0xdb4e,
0x7da,
0x167a,
0xe8ac,
0x5f4c,
0x22be,
0x7d53,
0xaea5,
0x4d57,
0x54be,
0x773b,
0xe76a,
0xdf0c,
0x52bf,
0x3373,
0x41ef,
0x5d5f,
0x14be,
0x7373,
0xc66f,
0x5f57,
0x8c5f,
0x1c2f,
0x664d,
0xde84,
0x1a7b,
0x7f,
0x26ae,
0xfd46,
0x3851,
0x1b33,
0x664f,
0x9d8f,
0x53b,
0x347a,
0x7e69,
0xee85,
0x1a3b,
0x913e,
0x44d,
0x6265,
0xd4c7,
0x9b2c,
0x2adc,
0x55a9,
0x2c09,
0x8679,
0x2320,
0x5901,
0x80c,
0xe64c,
0xa0f0,
0x1361,
0x7408,
0xeb54,
0xa0b0,
0x927a,
0x1032,
0xdb52,
0xc53a,
0xc37e,
0x90ec,
0xb7c2,
0xcdc2,
0xab0c,
0x1bef,
0xf017,
0x1c70,
0x3218,
0x749a,
0xfb42,
0x6110,
0x1b66,
0x679e,
0xa522,
0xedb1,
0xbd5a,
0x1246,
0xf5da,
0xc64d,
0x5542,
0x761a,
0xa106,
0x55ba,
0x27a,
0xe9a3,
0x8192,
0xe504,
0x9e18,
0x7b91,
0x9ad7,
0x8d32,
0x12ba,
0xb691,
0xddc0,
0x8ebc,
0x83fa,
0xc736,
0xff4e,
0x9eb8,
0x217b,
0xef18,
0xa222,
0x9890,
0x818b,
0xf825,
0xabc5,
0x48d2,
0xe13a,
0xa537,
0xbbaa,
0x85b3,
0xf4b9,
0x892d,
0xaf84,
0x3b7,
0xf1ac,
0x325,
0xe280,
0xa0b1,
0xd2b9,
0xd2b8,
0xc7e8,
0x10f1,
0x54a9,
0x47b1,
0xc3a2,
0x6090,
0xf7a9,
0xc59f,
0xfdfb,
0xc622,
0xb56,
0xe45a,
0x7859,
0x8f62,
0x2356,
0xad92,
0x5c79,
0x854e,
0x3b12,
0x9407,
0x5e59,
0xcf3e,
0x4956,
0x411e,
0x186d,
0x464b,
0x4157,
0xa00e,
0x545d,
0x4a78,
0x6936,
0xfdde,
0xbd3b,
0x60a,
0x894e,
0xa00c,
0xf959,
0x2b,
0x495f,
0xbc5e,
0x9f79,
0x1e6b,
0x4b16,
0xa9f5,
0x3cb9,
0x5e4f,
0xad03,
0xa9e9,
0x14bf,
0x5b0f,
0xaf03,
0x88f5,
0x3c65,
0x156e,
0xae41,
0x8945,
0xec3f,
0xe47,
0xfd01,
0xa9c4,
0x8f3a,
0x7a57,
0xf911,
0xed6d,
0xada3,
0x7b55,
0x2d51,
0xa9c5,
0x74ab,
0x1e43,
0xad25,
0xe9a1,
0xb5a3,
0x14fd,
0x7d85,
0xc9e7,
0x9d63,
0x7f56,
0xad01,
0x95d2,
0x5b46,
0xc79b,
0x4b7a,
0x9dc4,
0xcad6,
0xcd92,
0x220a,
0xf1c5,
0xc296,
0xa913,
0x93a,
0x7c1a,
0x3997,
0x35b8,
0xa62,
0xe352,
0xd92d,
0x85b5,
0x1c5b,
0x7302,
0xdb9b,
0xe546,
0x94e,
0x7612,
0x8117,
0x55ba,
0x32fa,
0x7302,
0xd086,
0xc594,
0x1ea8,
0x7a92,
0xf297,
0x8d32,
0x10fa,
0x8be5,
0xe4b3,
0x9806,
0x2351,
0xc1f0,
0x78db,
0xcbe,
0xe0c2,
0x1f3a,
0x107f,
0xc8eb,
0x40fa,
0xba45,
0x82df,
0x88c5,
0x155b,
0x3c54,
0x5a4e,
0x8ab3,
0x9541,
0x661a,
0x315a,
0x83b8,
0x555e,
0x8261,
0xa4f9,
0x1da5,
0x612b,
0xfa30,
0x5bc2,
0xa196,
0x86a8,
0x7432,
0x7355,
0xa1aa,
0x2fa,
0x8981,
0xace9,
0x7e46,
0x2ec7,
0xc983,
0xbc97,
0x3b46,
0xbae4,
0xcbae,
0xd25,
0x5b96,
0xae85,
0x8df7,
0x7ceb,
0x1e6e,
0x7cd5,
0xea3,
0x8c84,
0x3b86,
0x9c81,
0xcb89,
0x484,
0x3994,
0x9aa8,
0x89c5,
0x3ceb,
0x9e07,
0xe905,
0xc9e3,
0x748d,
0x5e16,
0xee84,
0xebcb,
0xdc86,
0x3914,
0xa865,
0x2fcc,
0xed45,
0xcab9,
0x9575,
0x8b04,
0xae74,
0xdbdf,
0x894f,
0x2eef,
0xa239,
0xd04f,
0x2ee5,
0x266b,
0xcf22,
0xd24f,
0xa554,
0x87ea,
0x9b60,
0x994f,
0xc27,
0x8bf7,
0x1639,
0x544b,
0xa543,
0x7ff,
0x577a,
0xdc61,
0xdd47,
0xffd,
0x7773,
0x3ec5,
0x3d45,
0xfbe,
0x3d63,
0x5e69,
0xbd05,
0x84d3,
0x359b,
0x4e66,
0x2b52,
0xa4da,
0x38d1,
0x8c22,
0x6202,
0xa89e,
0x1c15,
0x5d0a,
0x2252,
0x9b46,
0x9cdf,
0x4b1f,
0x4156,
0xa94c,
0x11cf,
0x4f21,
0x6916,
0x2415,
0x1d5d,
0x4b3b,
0x6d16,
0x744b,
0xb43b,
0xf0a,
0xb4e,
0xe804,
0xdd23,
0xa50e,
0x7b6e,
0xad54,
0x5d73,
0xe2a,
0xb56,
0x8007,
0xbd1b,
0xa76a,
0x956,
0x884e,
0xad5b,
0x2b2a,
0x4912,
0x84b,
0x8d31,
0x8b0e,
0x916,
0x951a,
0x9c1f,
0xb26,
0x6916,
0x5aba,
0xc107,
0xf322,
0x6296,
0xc2f6,
0xf553,
0x4b0a,
0x6072,
0xac0e,
0xdddd,
0xcf22,
0xcb54,
0x991c,
0xbfcd,
0x4aa8,
0x8996,
0x8d4e,
0x9e51,
0x4e3e,
0xb56,
0xebe1,
0xac5,
0xdc57,
0xe6a1,
0xaff1,
0x50cd,
0xa6f4,
0xa2dd,
0x8cb5,
0xe0ef,
0x7e71,
0xb6ec,
0x8e1d,
0x34da,
0x6a19,
0x5c41,
0x1e9d,
0x266c,
0x1c55,
0xd502,
0xe45,
0x76ea,
0xe2f9,
0x5584,
0x7051,
0xa125,
0x5b2,
0x2258,
0x6820,
0xa504,
0x21b4,
0x1708,
0x8a31,
0xad20,
0x932,
0xc63c,
0x5f6e,
0xe326,
0x81c9,
0x9d69,
0x3fae,
0xa3a8,
0x40e1,
0x9460,
0x366a,
0xa3f4,
0xe3c1,
0xd529,
0x32ee,
0xe36c,
0xe0e8,
0xd788,
0x1287,
0x6216,
0xc3d8,
0xc78c,
0x52ad,
0x6a64,
0x90f1,
0xc5cb,
0x162e,
0xd3dc,
0xe2e9,
0xd50c,
0x163c,
0xbaec,
0x12e1,
0xd505,
0x166e,
0xa37a,
0xd3c9,
0xc509,
0xbe1a,
0x831d,
0x89ba,
0x437a,
0xdd02,
0x1357,
0x4d92,
0x17a,
0xf41e,
0xc355,
0xcd3a,
0x17a,
0x69a,
0xf51a,
0xb1a7,
0x54a,
0x69a,
0xaf58,
0x31ae,
0x559a,
0x771b,
0x995e,
0xac7b,
0x4476,
0x8bd0,
0x80eb,
0x5995,
0x2283,
0x2b40,
0xf012,
0xfd94,
0x2e03,
0xfe81,
0x81d7,
0x85b2,
0xa0fe,
0x9950,
0x6cd7,
0x8d3f,
0x4b96,
0xcd12,
0xec55,
0xcba2,
0xb12,
0xf410,
0x904d,
0x8938,
0x4bda,
0xcc10,
0xf05c,
0x23ba,
0x4972,
0xdc18,
0x43dc,
0x21f0,
0x87a,
0x521a,
0xe316,
0x4b9e,
0x11a,
0xf452,
0xa115,
0x8132,
0xa70,
0xf012,
0xa155,
0xa5ba,
0x123a,
0x7d12,
0xed15,
0xa1be,
0x127a,
0x5628,
0xe382,
0x81b1,
0x10fa,
0x562c,
0x43c4,
0xfb90,
0x922a,
0x3724,
0xcacc,
0xc190,
0xc378,
0x56f8,
0xfb01,
0xa9f0,
0x10f8,
0x76cd,
0xea45,
0xccb8,
0xde59,
0xf208,
0xa29e,
0xa4e1,
0x8748,
0x162a,
0x3ba2,
0x4198,
0xc37e,
0x366f,
0x1b1a,
0xd963,
0xcd5f,
0x365e,
0x429c,
0xd06b,
0x617b,
0x8935,
0x4421,
0xca4f,
0xe8e3,
0x8aef,
0x6ca7,
0xf84f,
0x2d17,
0x88da,
0x7c2d,
0x3a67,
0xae05,
0x875f,
0xc472,
0x5a0f,
0x3cc6,
0xa86f,
0x8d79,
0xf881,
0x29e7,
0x896f,
0xc3d,
0x5a4e,
0x6916,
0xa9c5,
0xa4eb,
0x1e47,
0x2d81,
0xcd8d,
0x54fd,
0x3ec4,
0x28a5,
0x8985,
0x9ca3,
0x7f16,
0x2ec5,
0x7eba,
0x5336,
0x21b8,
0xd69c,
0xfeda,
0xeb4c,
0xc11e,
0xa66c,
0xabcf,
0xc81,
0x5c14,
0xe34f,
0xdebe,
0x4776,
0x6231,
0xd2cb,
0xdb74,
0x2a40,
0x4681,
0xa19a,
0xa80d,
0x3ca8,
0x4a94,
0x8d4f,
0x563a,
0x713c,
0xe3a8,
0x35e,
0x5600,
0x26d4,
0x208d,
0x546a,
0xd209,
0xb34,
0x439d,
0x8a3e,
0x8c0f,
0x9d3f,
0xa26e,
0x4b56,
0x882e,
0x8d53,
0x222f,
0x4356,
0x20de,
0x3439,
0xe4e,
0xb16,
0xec8f,
0x8d39,
0x470a,
0x6b1e,
0x79a3,
0xc093,
0xe50e,
0x2092,
0x90af,
0xed31,
0x4c6f,
0x4116,
0x8d0e,
0xf5dd,
0xe66a,
0x8b54,
0xd0ce,
0xb0d3,
0x5a6b,
0x7152,
0xcc,
0xbd59,
0xce4e,
0x1356,
0x26b2,
0x112e,
0xfcf8,
0xfb26,
0x8ef3,
0x31c2,
0x52d0,
0x2912,
0x1e7c,
0x615a,
0x8caf,
0x4974,
0x1717,
0xf55a,
0xb533,
0xaf04,
0x2792,
0xdff8,
0xb8b2,
0xfd0,
0x279c,
0xad7a,
0x84e7,
0x69c0,
0x40b2,
0xc543,
0x4dba,
0xe26,
0x7be1,
0x5083,
0xcfd4,
0x26d5,
0x78f1,
0xc083,
0x8db2,
0x68fa,
0x7e28,
0x436c,
0xb99d,
0xf4a9,
0x5b69,
0x87a4,
0x12d9,
0xd48b,
0x56e8,
0xe7c6,
0x72f1,
0xfeef,
0x9af8,
0x4be2,
0x3499,
0xb6a1,
0xbe0,
0xce83,
0xf894,
0xa2ee,
0x49ea,
0x8ce4,
0xb4ae,
0xe2f9,
0x4bb4,
0x40c4,
0xf8fd,
0xfcb0,
0x4ffb,
0x45a6,
0x78f0,
0x66a8,
0x4be3,
0x4faa,
0xb394,
0xbc89,
0x6437,
0x1f3b,
0x2669,
0x5d8d,
0x3c16,
0x1b38,
0x354d,
0xd5cd,
0xf493,
0x7b2e,
0xa6a1,
0x9f8d,
0x6d33,
0x573a,
0x7462,
0xf4a3,
0x7ff8,
0x33e,
0x7109,
0xb4af,
0x7361,
0xebac,
0xb095,
0xe4a9,
0xeba4,
0x42d8,
0x3091,
0xd4a9,
0x7b6c,
0xeb0c,
0x28b4,
0xd238,
0x3be0,
0x4acc,
0xb8b0,
0xd2a9,
0xa0d2,
0xbddb,
0x8622,
0xb52,
0xe49b,
0x785f,
0xcf6e,
0x6376,
0xad97,
0x5c19,
0x1d6e,
0x2350,
0x9d47,
0xdc5d,
0xcf3e,
0x4956,
0x301e,
0x125f,
0x462b,
0x4916,
0x9c1c,
0x33d9,
0x4a7b,
0x5d36,
0xfccb,
0x7d3b,
0x1f02,
0x294e,
0xa81e,
0x7733,
0x54e,
0xb53,
0xb41e,
0x5d79,
0x446e,
0x4b72,
0x7f2c,
0x8366,
0x3191,
0xf6ad,
0x5761,
0xc3ac,
0xd2d5,
0x848b,
0x5e6c,
0xc3c0,
0x72c9,
0xd4eb,
0x522e,
0x23ca,
0xb881,
0xb6a9,
0xb7e6,
0x8f8e,
0xb0d4,
0xf4a8,
0x4beb,
0xa3d6,
0xf6e1,
0xe4cb,
0xd2ee,
0x3d6,
0xf0f9,
0xf6b0,
0x56ca,
0x2396,
0x70d5,
0x8489,
0x1260,
0x396,
0xc1c1,
0xc4e8,
0xc615,
0x1d0e,
0x8ab8,
0x837a,
0x8e33,
0x3f62,
0x5a9c,
0xa16f,
0xef3d,
0x700e,
0x5db0,
0xa1eb,
0x3d95,
0x6b46,
0xc0db,
0xb3ba,
0xadb7,
0xaf62,
0xdd14,
0xbd6d,
0x44a5,
0xeb82,
0x17fd,
0xe9e9,
0xf261,
0xac84,
0xc0b1,
0xf2ba,
0xd3b0,
0x83e8,
0x3070,
0xd6a1,
0x53e1,
0x83a2,
0xe390,
0xfca9,
0xe051,
0x7cdb,
0xcf6a,
0xb56,
0xecd0,
0x5453,
0xa922,
0x2bd6,
0xa996,
0x7c5b,
0x893e,
0x2a36,
0xad52,
0x3d79,
0x4f3a,
0x874,
0x481b,
0x353f,
0x67b1,
0x5912,
0xf41a,
0x5991,
0x4b44,
0xc916,
0xbc5b,
0xfd73,
0x8f5e,
0x2b6c,
0xa817,
0xf44b,
0xec2e,
0x3f7e,
0xac17,
0x743b,
0xac26,
0xb76,
0xcb2d,
0x9ce5,
0xf79f,
0x8cc7,
0x5b20,
0xeae,
0x662d,
0xcf6,
0xe850,
0x13a1,
0x3348,
0x2c86,
0x84d9,
0x54f9,
0x194b,
0xf599,
0xe937,
0x54bf,
0x44f,
0x202e,
0xc667,
0x7d2d,
0x581f,
0xed93,
0x240c,
0xb65d,
0xa0e9,
0x7949,
0xdc1c,
0xaded,
0x22e9,
0x5123,
0xa51f,
0xef64,
0x22dc,
0xdb5c,
0x4f6c,
0xca02,
0x71d5,
0xb5a9,
0xdb2d,
0xbb8,
0x67d5,
0xd509,
0x4b6e,
0x8728,
0x2381,
0x5daf,
0x2bac,
0xe524,
0xb9c0,
0xb6e9,
0xa381,
0xcff4,
0xa9da,
0xd6e8,
0xfbaa,
0xe224,
0x343a,
0x32f9,
0xc260,
0xc054,
0xa395,
0x36b3,
0xdae1,
0xc3ec,
0xbbfd,
0x22a1,
0xcbae,
0x87a6,
0x3b95,
0x98a9,
0x5f24,
0x2be4,
0x2c95,
0x94f9,
0x1f2d,
0x8388,
0x65c9,
0x5cbd,
0x3f6e,
0x83d4,
0x20f1,
0xd5aa,
0x97a8,
0xe0a4,
0xa1e0,
0xb7e9,
0x33e6,
0xe635,
0xa1de,
0x54d9,
0xf366,
0xe2e4,
0x20ba,
0x96eb,
0x1628,
0xc244,
0xa1f9,
0x56a1,
0x9760,
0xc3ec,
0xb3a1,
0x82f1,
0xf268,
0x838c,
0xb3b1,
0x14e8,
0xfdc0,
0xe4dd,
0x8bbf,
0x10b2,
0xfcc0,
0xf204,
0xe4d6,
0x20f0,
0xb21c,
0xb39c,
0xdd98,
0x7c,
0xcbc1,
0xd4de,
0x3a97,
0x8a1,
0x78c8,
0xf164,
0x10f7,
0xc4e8,
0xb01a,
0x7f7a,
0xc159,
0x546c,
0xfb41,
0xa0d7,
0x8d94,
0x20a1,
0x3801,
0xc204,
0xa196,
0x228,
0xfe1b,
0xbb7c,
0x81b8,
0x7a,
0xdf2c,
0x3be6,
0x78d5,
0x90e9,
0x1b2d,
0xabea,
0xecd5,
0x5c2d,
0xc1fd,
0xa768,
0x76e1,
0x74a5,
0x63ad,
0xa7c2,
0xbdc4,
0xf689,
0x32c3,
0xe472,
0xbadc,
0xd6a9,
0x31e6,
0xe026,
0x34ad,
0x36e9,
0x9be5,
0x80c6,
0xa1c5,
0xb4b1,
0xcfe1,
0x82ac,
0x3b84,
0x2681,
0x4fe1,
0x8486,
0x5f95,
0x74a1,
0x523e,
0x437c,
0x41b8,
0x56b8,
0xd40f,
0x394,
0x4311,
0x8468,
0x985f,
0x911e,
0x5718,
0x605e,
0x563a,
0xe39e,
0xe1b1,
0x8a7c,
0x28e,
0x865a,
0x6394,
0xb8c8,
0xaa9f,
0xc66d,
0xfd90,
0xbded,
0x768a,
0x530c,
0xf1aa,
0xc3fe,
0xd6ae,
0x2afc,
0x2b1,
0xc0e7,
0xab4b,
0x1dbc,
0x5a9d,
0x810f,
0x8153,
0x7d9b,
0x462e,
0xb56,
0xa497,
0x7819,
0xcd2a,
0x6b56,
0xac17,
0x1c59,
0x4d0e,
0x3912,
0xbc17,
0xd03d,
0xc72f,
0x4156,
0x251e,
0x315c,
0x4e2b,
0x5b52,
0xa414,
0x7017,
0x4857,
0x3936,
0xe0c9,
0xd53b,
0x702,
0x14f,
0xe08c,
0xd251,
0x78e,
0xb812,
0xfc1c,
0x9d61,
0x4c0e,
0x4b36,
0x753a,
0x537c,
0x3110,
0x7ec5,
0xf502,
0xe51f,
0x7c31,
0x793f,
0x7571,
0x509f,
0xbcf5,
0x7df0,
0xfd3e,
0x432c,
0x7936,
0x5026,
0x2dda,
0x3c5f,
0x46ec,
0x69e4,
0x2bf7,
0x9cdd,
0xd35,
0x6e04,
0x4db9,
0x6363,
0x6dfa,
0x9eb4,
0x8bf5,
0xbc03,
0x5e51,
0x2746,
0x89c5,
0xbccb,
0x1e24,
0xad41,
0x8305,
0xbcc1,
0x8e9d,
0x3db,
0x3c64,
0xeecc,
0x19d8,
0x97ef,
0xfeb0,
0x8a82,
0x32d4,
0xf4c9,
0xf160,
0x2c81,
0xc789,
0x60fa,
0xf02d,
0xc1b6,
0x3151,
0xf4bb,
0xd168,
0xa3aa,
0xf2a4,
0xdca9,
0xab74,
0xca40,
0x18f5,
0x30a9,
0x277d,
0x8ac2,
0x20e1,
0xf5a8,
0x5b34,
0xa6a8,
0xb0b1,
0xd689,
0xdf24,
0x4e46,
0xf999,
0xd1e9,
0x5f60,
0xc7c2,
0xb4d1,
0x958b,
0x5768,
0x43a0,
0xb2d1,
0xdcef,
0xa7b0,
0x90a,
0x3599,
0x83eb,
0xac73,
0xcd2b,
0xbdd0,
0x8ae9,
0xd9f9,
0x8fe2,
0xb548,
0x42cd,
0x6370,
0x8244,
0x299c,
0xb6b8,
0x5f52,
0x9c2,
0xefd4,
0xa6f0,
0xcbc1,
0xca8e,
0xb9b4,
0xa4e9,
0xfc50,
0xd916,
0x8d3a,
0x23fe,
0xd442,
0x5552,
0xd96a,
0x317e,
0xb40e,
0x714c,
0x2bba,
0x17a,
0x7d20,
0xda06,
0xfd5e,
0x2aba,
0xdf63,
0xa4b3,
0xfd2e,
0xab8,
0xf450,
0xc35c,
0xa93a,
0x40fa,
0xfb42,
0xa887,
0x1d1e,
0xa2d8,
0xbb47,
0x8c4b,
0x1c27,
0xada4,
0xff41,
0xc05c,
0x8812,
0x207a,
0x84df,
0xfd1b,
0x662,
0x2b56,
0xa49b,
0xd857,
0xcd2a,
0x2b56,
0x8897,
0x3819,
0x8d1e,
0x974,
0x3c47,
0x581d,
0xc73f,
0x4156,
0x191f,
0x345d,
0x423b,
0x6d16,
0xb015,
0x1893,
0x4b51,
0x1916,
0x2dcb,
0x7529,
0x1f02,
0x4b4e,
0xa804,
0x793f,
0x890f,
0x4957,
0xb41d,
0xfd79,
0x66a,
0x4956,
0x80d7,
0x7d7f,
0x664e,
0xf16,
0x88df,
0xfcf9,
0x4f6b,
0x7e16,
0xacd7,
0x547f,
0x4d0f,
0x2972,
0x2516,
0x7f3b,
0x8e67,
0x5d57,
0x844c,
0x395b,
0xce2f,
0x1516,
0x251f,
0x3475,
0xe0f,
0x6b76,
0xe499,
0x543b,
0x4662,
0xc947,
0xe43b,
0x7037,
0xd21e,
0x3b57,
0xede7,
0x3571,
0x4f0f,
0x2b72,
0xf4de,
0xbdd5,
0xc66e,
0xb56,
0xe4d2,
0xfcdb,
0xc56a,
0x2356,
0xb89b,
0x781b,
0x8f5e,
0x2b56,
0x4c1b,
0x9c19,
0xcf3e,
0x15e,
0xa85e,
0x380a,
0x472b,
0x291c,
0x341e,
0xf9db,
0x4f20,
0xc036,
0x2c0f,
0x9b79,
0x423a,
0x4b4c,
0x2836,
0x7f7b,
0xb,
0x7916,
0x341f,
0xf059,
0x806a,
0x4b17,
0x9d7,
0xf411,
0xca2b,
0x2956,
0x80df,
0xb853,
0xc93b,
0x2156,
0xa81f,
0x5c93,
0x497a,
0xa156,
0x5b5b,
0x53b5,
0xc79d,
0x342d,
0xe1df,
0x76ef,
0x426b,
0xd557,
0xe81c,
0x7dca,
0x4962,
0x997e,
0xad83,
0xb8a9,
0x1a07,
0x816e,
0xfb36,
0x8e61,
0xbe8f,
0xb97f,
0xa919,
0xbded,
0x1d0b,
0xa17f,
0x7f2c,
0x43ec,
0x7895,
0x90e9,
0x576c,
0x43e6,
0xbe90,
0x90e9,
0x7798,
0xc3ac,
0x3690,
0xd4eb,
0xdba9,
0x4764,
0x3891,
0x96e9,
0xaaf1,
0xc7e1,
0xbed4,
0xc2e9,
0xa381,
0xcf8a,
0xb4cc,
0xc2cd,
0x5b74,
0x83c4,
0xf9b5,
0xd4ba,
0x8391,
0x8bee,
0x4ea1,
0xc4ec,
0x724,
0xc34e,
0xab91,
0x96cc,
0x7e2a,
0x53ac,
0x7595,
0xd6f9,
0x7f40,
0x4384,
0x6295,
0x1eb9,
0x57a4,
0xc753,
0xee85,
0x1ef9,
0xef21,
0xe708,
0x30b0,
0xb2b8,
0x5600,
0xc7c6,
0xa39e,
0x1038,
0x6f65,
0xe770,
0xbf9d,
0xaf0,
0x4ee5,
0xc786,
0xe0e4,
0xde91,
0x5fd9,
0xc7b4,
0x23a,
0x16f0,
0x5f88,
0xe242,
0x6a9c,
0x1088,
0x76e6,
0x4d8d,
0xab9a,
0x42fe,
0xee60,
0xc444,
0xa990,
0x237a,
0xb801,
0xec0d,
0x5f02,
0x22df,
0x57e2,
0x4182,
0x7b9b,
0xecde,
0x7f60,
0x89e0,
0x7d31,
0x3436,
0xfb50,
0xcc9a,
0xcd13,
0x293a,
0x76ba,
0xc1d8,
0xb392,
0x526a,
0xf63e,
0xc7d4,
0xe1b4,
0x126a,
0xf23a,
0xc9dc,
0xa1b0,
0x826a,
0x18c2,
0x6955,
0xcd22,
0x33e,
0x5f1a,
0x5b13,
0xeda2,
0x332,
0xfc02,
0xc853,
0xb13a,
0x230,
0x1499,
0x5b54,
0x33a,
0x407e,
0x72aa,
0x1b14,
0xc5a9,
0x5a,
0xd40a,
0x135f,
0xc19a,
0x412a,
0xf400,
0xc185,
0x8d3a,
0x27a,
0x3412,
0xdb57,
0x619c,
0x33a,
0xf412,
0xf11c,
0x8538,
0x33a,
0xf670,
0x1806,
0x9bbc,
0xe0eb,
0x8361,
0x1702,
0xd994,
0xa56b,
0xe3d9,
0x5606,
0x1eb0,
0xe1eb,
0x7b95,
0xb03,
0x4993,
0x97bb,
0x6da1,
0x8b52,
0xff94,
0xafe0,
0x49dd,
0xb0a,
0x35e9,
0xc9e9,
0x7a71,
0xc684,
0xf9b5,
0xfeb2,
0xd3d2,
0x83e0,
0xdfb0,
0xd6a5,
0xebd1,
0xc106,
0x6194,
0x9ca8,
0x7f3f,
0x3be8,
0x56fd,
0xfced,
0x13ad,
0xcbae,
0x76f1,
0x990d,
0x166e,
0xe3aa,
0x33e1,
0xd4a8,
0x23fa,
0x43aa,
0xa04c,
0xb7a9,
0x12a3,
0xe3a5,
0xa4e0,
0xc6ab,
0x57ef,
0x4b70,
0x70eb,
0xf6eb,
0x1fae,
0xebde,
0xe1f9,
0xd433,
0x177f,
0x36e,
0x6ab1,
0xd6c5,
0xc7ec,
0xa3ce,
0x53d1,
0xd4c7,
0xa951,
0xb489,
0x3e17,
0xecc7,
0xad44,
0xac09,
0x3f56,
0x7973,
0xe8c1,
0xfc09,
0x381f,
0x2a95,
0x927,
0x61e1,
0x5d7c,
0x6ae6,
0xaf78,
0x34ab,
0x7edf,
0x7bf5,
0xe961,
0x1cbd,
0x3b16,
0x2bf1,
0x55ba,
0x4371,
0x6378,
0x5eb4,
0x8db9,
0x4521,
0x7259,
0x3a2c,
0xa9d5,
0xacab,
0x1e17,
0xac65,
0xf08e,
0xd91f,
0x9922,
0x237e,
0xf45a,
0x4c57,
0xe18a,
0x637e,
0xb89a,
0x5801,
0xc83a,
0x37a,
0xdc18,
0x18df,
0x67b8,
0x4916,
0x5c59,
0x3185,
0x4531,
0x2876,
0xf058,
0x119d,
0xcba0,
0xd93e,
0x341e,
0xb95b,
0x33a,
0x4b5a,
0x901e,
0xbc1b,
0xc83e,
0xc11e,
0x3414,
0x7c59,
0x802b,
0x435e,
0x122d,
0x13a2,
0xe0b1,
0x95e9,
0x52ae,
0x33f4,
0xc3d1,
0x95ef,
0x3278,
0xd324,
0xe2c9,
0xf7cf,
0x36ae,
0x136a,
0xc8e9,
0x9729,
0x36ae,
0x4322,
0xe2d8,
0x935d,
0x32ac,
0x62e4,
0x91f9,
0xd5e9,
0x1aae,
0x73b8,
0xe2e1,
0xd5cb,
0x1eac,
0xa3fc,
0x12c1,
0xd465,
0x7ae,
0x23ca,
0xcbc1,
0xd549,
0xdb,
0xbd3b,
0xc76e,
0x6b56,
0xc05b,
0x9839,
0x636f,
0x2b16,
0xa8d7,
0x5c53,
0xdd0e,
0x2b76,
0x2417,
0x3d1b,
0x876f,
0x7156,
0xdc6c,
0x6e46,
0x622b,
0x7197,
0x6c15,
0x6473,
0x4c0b,
0x7976,
0x44ce,
0x1cb3,
0x9f46,
0x2b47,
0xf4ac,
0x7191,
0xe86e,
0x3d57,
0x40cb,
0x75f1,
0xaf6e,
0x3876,
0x372e,
0x3b36,
0x5140,
0x8d17,
0x23af,
0x65a6,
0xb6e4,
0xc50d,
0xae7,
0x3aa1,
0x52ed,
0xfd4d,
0x769e,
0x7328,
0xe162,
0x9f2e,
0x2bb4,
0x319e,
0xae88,
0xdd0c,
0x23e5,
0xc4f7,
0x9e05,
0xff85,
0x54be,
0x7f64,
0x626a,
0x9f56,
0xead,
0x7cf3,
0xcf4d,
0x970d,
0x3f5,
0x54a3,
0x5e45,
0xff85,
0x322e,
0x93b2,
0xa2f9,
0x550d,
0xf62e,
0x8b90,
0x43e5,
0x452b,
0x1a2f,
0x83b8,
0x87e9,
0x573a,
0xd7ac,
0xd33e,
0xa9e0,
0x9728,
0x17ae,
0xe2be,
0xc27a,
0xd32f,
0x16a2,
0x24c,
0xf5eb,
0xd46b,
0x16ae,
0xf3c0,
0xf0e1,
0xc54a,
0x9e8c,
0x9bde,
0x2b1,
0xc545,
0x363c,
0x235e,
0xc2e9,
0xc54b,
0x89ad,
0x8c6b,
0x524f,
0xfee7,
0x8baf,
0xbce8,
0x146f,
0xad67,
0x89fb,
0xbee7,
0x9e67,
0xfec7,
0x835d,
0xacfa,
0x1e0f,
0xbdc7,
0xc3b7,
0x647a,
0xb847,
0x2537,
0xcfe7,
0x44eb,
0x5a07,
0x2d97,
0x82e5,
0x34fb,
0x9647,
0x7ca1,
0xc38d,
0x55ed,
0x3acd,
0x5b25,
0x89af,
0x3ca0,
0x5ed7,
0xbe05,
0x6eb3,
0x554e,
0x79e4,
0xf28d,
0xe771,
0x64ce,
0x7890,
0xb0ed,
0xe915,
0xd444,
0x5a90,
0xe2e9,
0x6633,
0x755e,
0x71a9,
0x5a87,
0xc4f8,
0x36c1,
0xd890,
0x6a60,
0xe999,
0x408a,
0x7d90,
0x2acc,
0xea51,
0x2cdf,
0x1cb7,
0x62a0,
0x6961,
0xf8c6,
0xfe94,
0x2ea8,
0xe9c1,
0x98ac,
0xbd90,
0x28e8,
0x52c8,
0xdbe3,
0x8b96,
0x20be,
0xf786,
0xe990,
0xe982,
0x87a,
0x7496,
0xbbd8,
0x8198,
0x2a7a,
0xd2b8,
0xb2dc,
0x21b2,
0x129a,
0x529a,
0x7c5d,
0xcdae,
0x823e,
0xf21b,
0xb35c,
0xc38a,
0x19e,
0xd21a,
0x9144,
0xa1ba,
0x10ca,
0x361e,
0xdb54,
0xe19e,
0x5076,
0x361a,
0x4b1c,
0xa9ba,
0x837a,
0x800f,
0xbd17,
0xc70a,
0xb16,
0x8cf7,
0x1c13,
0x456f,
0x2316,
0xa9f7,
0x355b,
0x8d2f,
0x912,
0x249f,
0x5c96,
0x8767,
0xb05,
0x3eb5,
0x4693,
0xb606,
0xbf8c,
0x66bc,
0x5571,
0x786b,
0x7f07,
0xf40b,
0xbd27,
0x4f0a,
0xab56,
0x69c8,
0x10fb,
0xdf46,
0xad47,
0x54cc,
0x7573,
0xd74b,
0x6857,
0x889e,
0xbd93,
0xcc2a,
0x2b52,
0xe416,
0x7c53,
0x8f6a,
0x2332,
0xb813,
0x789b,
0x8d5e,
0xb52,
0x5d4e,
0x509d,
0x470f,
0x4956,
0xad56,
0x18df,
0x4737,
0x4916,
0xb81e,
0x37df,
0xf21,
0x8d16,
0x1c9e,
0xbf29,
0x1b1a,
0x434c,
0x103e,
0x7b6b,
0x442f,
0xd95e,
0xb45e,
0xba39,
0x846b,
0x4357,
0x88d7,
0xbd9b,
0x72a,
0x952,
0xec9a,
0xfc15,
0x252a,
0x2b56,
0xa917,
0x1d57,
0xd1e,
0x2134,
0xa516,
0xd8fd,
0xc61e,
0x4956,
0x251e,
0x398f,
0x476f,
0x7914,
0xb417,
0x5cdb,
0x4a45,
0x5936,
0xe4ca,
0x5d2b,
0x8f2a,
0x494e,
0xa89c,
0x787d,
0x30e,
0x695f,
0xb018,
0x3df1,
0x846f,
0x4b52,
0xad3f,
0x14fb,
0x5e65,
0xed85,
0x873f,
0x54fb,
0x9e67,
0x3e20,
0x1f2,
0x367f,
0xaa6f,
0x5d20,
0x23ff,
0x57b9,
0xd6c7,
0xf585,
0x8ba1,
0x4479,
0x12df,
0xdfa7,
0x5b3,
0x70dd,
0xacc3,
0x5785,
0x89ed,
0x75b3,
0x7645,
0xfd85,
0xcefc,
0x5de3,
0x16c7,
0xaf05,
0x24fe,
0x15e3,
0x5643,
0xffc5,
0x7d06,
0x4597,
0xfd45,
0x78b1,
0x6546,
0x539f,
0xe8cb,
0x7da9,
0x7225,
0x7ba4,
0x68ef,
0x14a2,
0xafd3,
0x9c8f,
0x1446,
0xf494,
0x697b,
0xb4dd,
0xcc6f,
0x7962,
0x9662,
0xa4d9,
0xa1ef,
0x4712,
0xa9c5,
0xbcdb,
0x1e55,
0x2061,
0xb40,
0xdc83,
0xdfcf,
0xa3d0,
0x56f0,
0xc243,
0xb06a,
0x38d3,
0xa4e3,
0x510f,
0x8e22,
0xb32,
0xc473,
0x34d7,
0xcd56,
0x7b92,
0x9bb2,
0x485f,
0xdd7a,
0x43d4,
0x7f07,
0xe35b,
0xe297,
0x182e,
0x909d,
0x8b7c,
0x3031,
0x5d10,
0x1a9c,
0x759b,
0xcde7,
0x7bd4,
0xdbd1,
0xecf3,
0x4f97,
0x12a3,
0xe301,
0xe092,
0xe796,
0xbe0a,
0xfae1,
0xc487,
0x1d22,
0xa0b1,
0x9b2d,
0x98f4,
0x528f,
0xec4f,
0x832d,
0x8af2,
0x428f,
0x8df7,
0xeaec,
0xf7b3,
0xa24f,
0xad06,
0xa06c,
0x1e7c,
0x486f,
0xf599,
0x41b6,
0x707b,
0x2007,
0x780f,
0xc7a7,
0x3f2b,
0x7a4f,
0x619b,
0xac9d,
0x311d,
0xa068,
0x6949,
0x9c1c,
0x3d7d,
0x22e9,
0x5941,
0xa11f,
0x2f74,
0x4e5c,
0x9f5c,
0xce41,
0x8447,
0xa936,
0x22b2,
0xecd3,
0x4c21,
0xfd1e,
0x7a86,
0x89bd,
0x1c3b,
0x7f1e,
0x2a94,
0xdf60,
0xf082,
0x5994,
0x22f0,
0x42ed,
0xf4a0,
0x5b97,
0xe4b8,
0x8df3,
0x5c21,
0x5b14,
0x3ea5,
0xa945,
0x8c85,
0x1d97,
0xa0f1,
0xad67,
0xec21,
0xca16,
0xecfc,
0xa9c7,
0x9489,
0x3c12,
0x2ca5,
0x7f24,
0x6a80,
0x1191,
0xd0a9,
0x1f2d,
0xa3a0,
0x3391,
0x9cad,
0x7a0d,
0xe380,
0xa1e1,
0xd5af,
0x8238,
0xe5a0,
0x31c4,
0xb6a9,
0xf688,
0xeb65,
0xb5d4,
0x92e9,
0xdffb,
0xc364,
0xa0fe,
0x6cb,
0x5a60,
0x8ac4,
0xe1d9,
0x76b1,
0x5fe1,
0xc246,
0xbd94,
0xe6b1,
0xff20,
0x824e,
0xa195,
0x94a9,
0x3b69,
0xe4b7,
0xaa17,
0x5ce7,
0x7646,
0xd3f1,
0x24ab,
0x87c,
0x75d2,
0x7b3d,
0x8762,
0x1d72,
0xc3e9,
0xc4fb,
0xb8c7,
0x74a5,
0x67cc,
0xf573,
0xfe0f,
0x18e1,
0x769e,
0x4b55,
0x90cb,
0x5777,
0x8be5,
0x34ea,
0xdac5,
0xfda5,
0x87dd,
0x54ed,
0x9af9,
0xe9e7,
0x36df,
0x577d,
0xf2eb,
0x5163,
0x49b,
0xfd7b,
0x866a,
0xb56,
0xa49a,
0x785f,
0x8f6a,
0x6352,
0xb816,
0x5c5b,
0x9d6a,
0x2352,
0x1d1e,
0x1a5f,
0xc73b,
0x4956,
0xd5e,
0x305c,
0x4f39,
0x7516,
0x201c,
0x1295,
0x4f01,
0x1516,
0x40a,
0xf92b,
0xe2a,
0x494e,
0xa13c,
0xf37b,
0xc50b,
0xe95e,
0x345f,
0xff79,
0x816b,
0x4b56,
0x273e,
0x532e,
0x3467,
0xbd61,
0x6f15,
0x51af,
0xb6d1,
0xfccd,
0x3f5,
0x1576,
0x56ed,
0xfd0f,
0xffe,
0x1570,
0xf6e3,
0xd555,
0x4d76,
0x1268,
0x5229,
0xf505,
0x3b6,
0x546a,
0x7ae1,
0xfc05,
0xabf,
0x76fa,
0xf261,
0xdd0f,
0xebd,
0x796b,
0x726b,
0xfd05,
0x1fd,
0x15a3,
0x7e65,
0xff01,
0x7942,
0x1834,
0xd4bf,
0xcf14,
0x1b1a,
0x79f4,
0xc48f,
0x8e04,
0x52df,
0x1171,
0x3747,
0x1e14,
0x2552,
0x553c,
0x3723,
0x611e,
0xc33b,
0x7bfb,
0x20c7,
0x6824,
0xf77,
0x353b,
0x4a86,
0x6100,
0x6418,
0x6305,
0x21b8,
0x127a,
0x4030,
0x7347,
0xa7b4,
0x1328,
0x4431,
0x5f5f,
0xa5f1,
0x537a,
0x732e,
0x83a6,
0x30d1,
0xb4c9,
0x13bc,
0xabff,
0xb6e9,
0x9549,
0x3361,
0xdba0,
0xa0c5,
0x9508,
0x5a2f,
0xe67d,
0xc294,
0x950f,
0x7719,
0xaa9d,
0x12eb,
0x815f,
0x12e2,
0xaaca,
0x52e9,
0xd589,
0x47be,
0xc37a,
0x71d9,
0x9608,
0x2b06,
0xcf82,
0xdcc5,
0x8148,
0x16b4,
0x1298,
0x90e9,
0xc5a9,
0x151e,
0x733e,
0x277a,
0xb72,
0x3f1e,
0x13f3,
0x456b,
0x1978,
0x3636,
0x2b14,
0xc6e3,
0x1d0c,
0x769e,
0xbb3d,
0xb463,
0xb06e,
0x6627,
0x8030,
0x202b,
0x5b7f,
0x32ab,
0x47b0,
0x35b6,
0xb36f,
0x7ff,
0x56fa,
0x5245,
0xdd2d,
0xc7cf,
0xcba,
0x56c1,
0xed85,
0x7ee,
0x5e3,
0x56c5,
0xad05,
0x249f,
0x1ddb,
0xa766,
0x914,
0xe45f,
0xfc95,
0xc66,
0xb14,
0x34a3,
0x211b,
0x8647,
0xb70,
0x251b,
0x1df8,
0xdbb8,
0x6176,
0xe35d,
0x6c96,
0x31b5,
0x603b,
0x3273,
0xd9,
0xea82,
0x1137,
0x44ae,
0x576b,
0x6746,
0x4b2d,
0x64e8,
0x7191,
0xe04e,
0x5a9f,
0x54ba,
0x767b,
0xe86b,
0x5b36,
0x7ba0,
0xeaa0,
0xb89d,
0xf6eb,
0x7b88,
0xebac,
0xb0f8,
0xd4ed,
0x5729,
0xaba4,
0xf0d0,
0xd6c9,
0x38e0,
0x2c65,
0xa8ac,
0xa4d3,
0x7f3b,
0xa86a,
0x60d9,
0xd8f2,
0xda20,
0xa28a,
0xf0a4,
0xf680,
0x9a15,
0xaadc,
0x1091,
0x709a,
0xae16,
0xae0c,
0x95d1,
0xe338,
0x5330,
0x8a88,
0x30b4,
0xf689,
0x89b1,
0x144d,
0xde47,
0x6740,
0x61d3,
0x14ad,
0xae7e,
0xeee4,
0x8d97,
0x940d,
0x7c76,
0x6a44,
0xa964,
0x7d09,
0x1e6f,
0x6855,
0xd940,
0xc66c,
0x3d55,
0x3c90,
0x8d61,
0x846f,
0x6e57,
0x7c74,
0xe9c5,
0x14a3,
0x1c03,
0x7883,
0x68b1,
0x5421,
0x75d2,
0x7eb2,
0xe991,
0xfc63,
0x1e16,
0x2eb5,
0x5442,
0xd91f,
0xb12a,
0x217a,
0xd50e,
0xb17,
0x39b3,
0x413a,
0xfc40,
0xeb15,
0xa11a,
0x178,
0xb2d2,
0xc888,
0xdd06,
0xa33b,
0xf7f3,
0x489,
0xd387,
0xa8ed,
0xfa81,
0x39d9,
0x70ca,
0x83fe,
0x950a,
0xbb8d,
0x6b1e,
0x425e,
0x9b5d,
0xfecb,
0x4a7e,
0xe156,
0xb252,
0xc21d,
0x880a,
0x17e,
0x8843,
0xec17,
0x8e0e,
0x4b52,
0x8817,
0x8553,
0x8b2b,
0x912,
0xec0f,
0xe553,
0xb1f,
0x2b16,
0xb112,
0xd5b7,
0x4b26,
0x6b16,
0x563a,
0x5527,
0x5b0e,
0x6bdc,
0xc7a3,
0x4137,
0x4914,
0x413e,
0xac06,
0xfc5d,
0xcf3a,
0x2b56,
0x941e,
0x955d,
0xcb2e,
0x2076,
0xfd4e,
0xc455,
0xb1e,
0x2976,
0x6bb6,
0xa6a9,
0x7645,
0x7cc1,
0x66e3,
0x70af,
0x76e4,
0xbec7,
0x82b5,
0xbcea,
0x566d,
0xfcc5,
0x83e7,
0x65eb,
0x7245,
0x5d41,
0x9ec5,
0xcefc,
0xe5d,
0xdd43,
0x46c,
0x266b,
0x5add,
0x7f05,
0x89bd,
0x443a,
0x4643,
0xfda1,
0xe028,
0x65b6,
0xa0d1,
0x536a,
0x1ad,
0xda2,
0x57f5,
0x8f05,
0x7f34,
0x2aec,
0x77dd,
0xb4e9,
0x4bad,
0x27ac,
0x32c5,
0x9c8d,
0x63e9,
0x87aa,
0x2285,
0xdcad,
0x22f9,
0xefea,
0x10c4,
0xbea9,
0x46a1,
0xe7f3,
0xa8c4,
0x96e9,
0x43e9,
0xae64,
0xb0df,
0xcee9,
0xcba5,
0x86de,
0xe4f5,
0xf4b3,
0x565b,
0xe624,
0x1bf1,
0x56a1,
0xcba2,
0xa206,
0x7f95,
0xbc89,
0x48fb,
0x3dbd,
0x434a,
0xaf15,
0x3af3,
0x48fd,
0xa16f,
0xba14,
0x57e7,
0x97dd,
0x656f,
0x7e14,
0x249a,
0x1c33,
0xb128,
0x3e47,
0x4e7a,
0xccd5,
0xb8ab,
0xf602,
0x4a33,
0xf357,
0xadca,
0xe616,
0xac4b,
0x3c3f,
0x970e,
0x947,
0x6c5e,
0x3d9f,
0xdf8e,
0x2b46,
0x7cba,
0x3c57,
0xd64e,
0xed7,
0x661f,
0x1f2e,
0x66ed,
0xdd85,
0x3df7,
0x1e3a,
0x14ed,
0x55c5,
0x74d7,
0x1737,
0xa5cd,
0x1dc5,
0x6fb9,
0xde2e,
0xb462,
0xb7a9,
0x75b1,
0x903a,
0xb50c,
0xaa2b,
0x57fb,
0x7a3f,
0xf46f,
0x63bd,
0x5b64,
0xd284,
0xa0b1,
0xf488,
0x3fcc,
0xa26c,
0x3cf1,
0xf5a4,
0x1be2,
0xe2c4,
0xbd95,
0xf589,
0x8c4f,
0x8c99,
0x860e,
0x4b56,
0x8c0e,
0xad51,
0xea2e,
0x6b52,
0xa45f,
0xf533,
0x8c4f,
0x916,
0x99ca,
0xa695,
0x4f06,
0x295e,
0x7932,
0x8133,
0xc72e,
0x22d6,
0xc0b2,
0xbb71,
0x4866,
0x4112,
0xfd0e,
0xf71d,
0xc622,
0x4356,
0xd404,
0xbbdf,
0x866a,
0x3956,
0x890c,
0xbf51,
0xcf6e,
0x1b56,
0x36af,
0x5f3e,
0x6247,
0x5d41,
0xb6ee,
0xc8df,
0x6627,
0x4b09,
0x14e2,
0x133b,
0xa66f,
0x5d19,
0x639b,
0x67eb,
0x2694,
0xf74c,
0xf350,
0x50b1,
0x3d81,
0x6e5e,
0x32f7,
0x529b,
0x86e5,
0xff12,
0x54ba,
0x537b,
0x6368,
0xbe2c,
0x2fc,
0x71e3,
0xfac4,
0x161f,
0x54bc,
0x53d3,
0xc6e9,
0x5e96,
0x7e3a,
0x433c,
0x65b0,
0x9698,
0xfa5b,
0x13be,
0x5495,
0x840a,
0xb85f,
0xfc30,
0x5c19,
0x623a,
0x769e,
0x634a,
0xe0b1,
0x836c,
0xbad7,
0x3639,
0xc290,
0xbd2c,
0xf9d5,
0x8a08,
0xc91,
0x8978,
0x563a,
0x633c,
0xe1aa,
0xd6fe,
0xd6ba,
0x42e8,
0x639d,
0x9425,
0xcbc1,
0x1db8,
0x5b95,
0xab0c,
0x53ed,
0x4246,
0x2ab9,
0x91ba,
0x926d,
0xcb80,
0xbf5,
0x9db8,
0x786a,
0xa394,
0xb941,
0xfabc,
0x92f0,
0xc6a6,
0xb1e8,
0x9ea9,
0x16e5,
0xe728,
0xa0dc,
0x16e9,
0x7faa,
0xa32c,
0xa85e,
0xb6db,
0x5204,
0x8254,
0xe1e8,
0x1692,
0x5be1,
0x82e4,
0x33d5,
0x60a1,
0xde60,
0xc3d4,
0xe195,
0xd4ae,
0x772c,
0x43c4,
0x3991,
0xb0ad,
0x5f64,
0xa3c6,
0x38b1,
0xd1a9,
0x7200,
0xe3a4,
0xa2a1,
0xd4ab,
0x3728,
0xa108,
0x51d0,
0xa3a8,
0x278e,
0xc280,
0xd584,
0xc7e9,
0x97e8,
0xcf9a,
0xb0c1,
0x86a9,
0x5f70,
0xabc4,
0x7990,
0x96f8,
0x9be1,
0x8a8a,
0x5f81,
0x87e9,
0x3351,
0x835e,
0x81b1,
0xc0ab,
0x702f,
0x933b,
0x826a,
0xcd5d,
0x322e,
0x2b33,
0xa36b,
0x1f4e,
0x1a76,
0x333f,
0xb36f,
0x5538,
0x102f,
0x733f,
0xc362,
0xe554,
0xb7af,
0x323e,
0x329,
0x917a,
0x16ef,
0x525c,
0x6eb,
0xd56b,
0x1ebe,
0x7b79,
0xe2f9,
0xd54e,
0x971c,
0xfb5e,
0x40e3,
0xd155,
0x167c,
0x1b58,
0xc2e9,
0xd558,
0x8bec,
0x226c,
0xead5,
0xf5a9,
0x1b2d,
0x7fae,
0xf6c5,
0xfc8d,
0x437d,
0xc7a6,
0x76e5,
0xfccd,
0x237c,
0x8d22,
0xb8c4,
0xbe89,
0xafb3,
0xcf3a,
0xb854,
0xb6ed,
0x5fe5,
0xc628,
0xbcef,
0xf6e9,
0xcbe5,
0x2ae,
0x64c5,
0xbc91,
0x4feb,
0xc2b8,
0xb3f1,
0xa6a5,
0xc3e1,
0x83c2,
0x7f95,
0xbca1,
0x88e1,
0xbcd3,
0x4eef,
0x2926,
0x8af9,
0x8441,
0xcfbe,
0x2244,
0x8dae,
0xe4d9,
0x4c9d,
0xa27d,
0x84cc,
0xaf62,
0xab1a,
0xb41,
0x548c,
0xc3a0,
0xf950,
0xfba5,
0x8894,
0xce20,
0xda98,
0xaa6f,
0x520a,
0x8384,
0xa1aa,
0xcb5a,
0x740a,
0x334,
0xa130,
0xdefe,
0x500f,
0xba4,
0x6399,
0x8a0e,
0x896f,
0x1c27,
0x564f,
0xff8e,
0x11e3,
0x2430,
0x644f,
0xed84,
0xb20f,
0x5d31,
0x5ee,
0x188e,
0x94db,
0x147a,
0x2b,
0xddcb,
0xf8f2,
0x183b,
0x3641,
0xb80d,
0x345f,
0xfb74,
0xe0d9,
0x516b,
0x2c10,
0x6479,
0xe70a,
0x7901,
0x4005,
0xa1ca,
0xa6f4,
0x1329,
0xe008,
0x6bd4,
0x89f8,
0x93ea,
0x89d7,
0x14fb,
0x4e6f,
0xef05,
0x8af3,
0x14b3,
0x1a46,
0x2f85,
0x8cff,
0x9c33,
0x1cd6,
0x6e45,
0xc0ef,
0x7522,
0x4e7e,
0x9f05,
0xce84,
0x56f0,
0x5810,
0xfe05,
0x9bc,
0x5c63,
0x5edd,
0xaf05,
0x6c0d,
0x7f27,
0x6662,
0x8e57,
0x649e,
0x1727,
0xf714,
0x9a15,
0xc9ef,
0x1d23,
0x5f57,
0x2f05,
0xa36d,
0x1e6a,
0x76dd,
0xfd89,
0xbcd,
0x1ee9,
0x7ec5,
0xfc6d,
0x43ef,
0xf62a,
0xdced,
0xfdc1,
0xbfd,
0x8228,
0x1e45,
0xff85,
0x8fc5,
0x9cfa,
0xfa01,
0xf78d,
0xbf7,
0x8eb2,
0x1c45,
0xbec1,
0xbf7,
0x56ab,
0xdeed,
0xfca5,
0xfe9,
0x54a9,
0xde4d,
0xbec5,
0x8fef,
0x14ea,
0x5e45,
0xbc85,
0x85d3,
0xbc9b,
0x472e,
0xb56,
0xa417,
0x1819,
0x8d6a,
0x2b56,
0xad16,
0x3919,
0x8d36,
0xb54,
0x3c13,
0xf61d,
0xc76f,
0x4156,
0x341f,
0x327d,
0x256b,
0x5713,
0xf40c,
0x737d,
0x4873,
0x5b1c,
0xf449,
0x143b,
0xf0a,
0xb4f,
0xb000,
0x5b7f,
0x251f,
0x3953,
0xb41c,
0xd555,
0x446b,
0x2b76,
0xed50,
0x7cc5,
0xbd36,
0x42b2,
0xcd50,
0x7c40,
0xb83a,
0x6094,
0xc145,
0x5d04,
0x393a,
0xea96,
0xec50,
0x9880,
0x179d,
0x68e2,
0xe808,
0xf5a3,
0x3cd5,
0x28e8,
0xcd5a,
0xed97,
0xeb14,
0x2a1a,
0xe851,
0xec07,
0x8d1e,
0x3ab8,
0x6951,
0x2407,
0xed0c,
0x3aba,
0xf903,
0x8c0d,
0xad36,
0x2afa
}
};
static ap_fixed<28, 16> alphaMem5[2] = {
1.0319891,
0.48674005
};
static ap_fixed<28, 16> thresMem5[1][128] = {
{
-37.384735,
-33.13035,
-121.34204,
-33.719852,
-55.177254,
3.5858154,
-58.427086,
315.71814,
-197.73555,
-75.91904,
-349.13422,
20.775066,
36.036922,
52.55759,
-71.47148,
10.615274,
-108.33123,
29.39852,
-130.61143,
-104.81645,
750.92474,
-217.90814,
-50.376053,
-60.234634,
105.43776,
26.618603,
-62.89436,
40.82168,
-20.94579,
-150.62025,
-48.00701,
50.04911,
-23.488068,
-2.1032047,
-198.88745,
-70.00041,
55.7334,
116.959625,
-252.33536,
46.154655,
32.301575,
87.990746,
49.301437,
61.70131,
-102.3601,
-41.47506,
146.14609,
116.79856,
43.77386,
134.77423,
-36.36959,
52.087837,
73.78531,
-115.53391,
64.34177,
-11.240562,
25.950823,
371.7422,
-268.80817,
92.5237,
118.91872,
-60.2638,
187.55765,
505.4453,
64.41948,
-130.36047,
35.04862,
-207.79678,
-3.7878838,
93.33244,
157.2536,
61.373135,
858.04474,
-111.78645,
-586.1021,
-20.673145,
98.60485,
-17.366566,
-55.727295,
52.112663,
112.587006,
89.02084,
-32.11862,
-2801.0903,
69.10386,
-127.61586,
-43.875782,
39.505116,
22.659042,
139.09987,
-9.763992,
-55.851032,
-212.48132,
-75.15227,
-27.32803,
-54.798122,
11.707773,
-45.20666,
47.42283,
617.2414,
-80.92815,
-542.4944,
39.323196,
-31.065811,
63.736458,
-77.86859,
-108.38168,
58.530468,
17.432602,
-4.689251,
-29.946793,
-47.96189,
110.05628,
-124.74482,
4.064904,
24.332893,
-122.1432,
55.965324,
-184.127,
129.71536,
-56.709007,
3790.4038,
20.172,
-37.912865,
17.10596,
-75.93751,
-24.548452,
9.121326
}
};
static ap_fixed<28, 16> means_out5[1][128] = {
{
15.318404,
22.675802,
15.397558,
14.479708,
21.7286,
22.66875,
20.62365,
208.45454,
16.01228,
32.02834,
200.76605,
17.984966,
20.512417,
14.665645,
30.785347,
22.950449,
47.737198,
18.474302,
34.953327,
23.719126,
677.9859,
16.62871,
33.14374,
13.760938,
636.5919,
21.31029,
162.73076,
16.257755,
25.860518,
14.756193,
18.188663,
25.360521,
17.169254,
13.121051,
15.530482,
15.601016,
22.458538,
12.352066,
253.51556,
25.602413,
21.110592,
12.445216,
18.008137,
20.031887,
130.96577,
17.332205,
17.524841,
30.598618,
17.30366,
30.645348,
17.287228,
21.305847,
32.70314,
12.794639,
13.115796,
59.192043,
16.04093,
730.2001,
5986.8745,
15.049909,
38.180088,
16.480139,
276.94357,
501.07407,
18.464785,
58.961327,
23.451035,
159.66484,
18.089973,
13.400513,
129.9952,
26.289948,
416.422,
34.02914,
478.08255,
21.230993,
27.227827,
19.44874,
19.926685,
19.442768,
11.887077,
34.599575,
24.177502,
5687.4307,
32.997833,
19.03414,
27.519411,
34.04661,
18.861345,
15.488174,
25.81496,
13.875115,
648.48145,
28.661419,
22.908869,
16.257248,
17.048359,
20.54282,
20.253895,
597.46857,
25.106142,
492.98666,
16.127684,
14.924062,
16.137573,
31.253143,
13.384138,
10.970763,
19.341743,
19.529743,
16.628832,
33.226486,
42.040363,
13.281579,
16.087875,
93.97173,
12.557351,
17.3659,
116.553925,
48.538532,
24.414814,
3814.9905,
21.021624,
16.839985,
18.698166,
23.340097,
22.767935,
54.874207
}
};
