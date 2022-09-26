#ifndef __TOP_H_
#define __TOP_H_

// INCLUDES
#include "bnn-library.h"	// bnn libraries needed for the ReBNet layers
#include "config.h"			// config file of neural network model
#include "memdata-0.h"		// parameters of layer 0
#include "memdata-1.h"		// parameters of layer 1
#include "memdata-2.h"		// parameters of layer 2
#include "memdata-3.h"		// parameters of layer 3
//#include "memdata-4.h"		// parameters of layer 4
#include "weights.h"
#include "memdata-5.h"		// parameters of layer 5
#include "memdata-6.h"		// parameters of layer 6

#include <iostream>
#include <string.h>

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0')

// PROTOTYPES
void DoCompute(ap_uint<64> * in, ap_uint<64> * out, const unsigned int numReps);
void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, ap_uint<64> val);
void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, bool doInit, unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, ap_uint<64> val, unsigned int numReps);

#endif //__TOP_H_
