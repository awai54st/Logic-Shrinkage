/******************************************************************************
 *  Copyright (c) 2018, ACES Lab, University of California San Diego, CA, US.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 *  IMPORTANT NOTE:
 *  This work builds upon the binary CNN library (BNN-PYNQ) provided by the following:
 *	Copyright (c) 2016, Xilinx, Inc.
 *	link to the original library (BNN-PYNQ) : https://github.com/Xilinx/BNN-PYNQ
 *
 *****************************************************************************/
/******************************************************************************
 *
 *
 * @file bnn-library.h
 *
 * Library of templated HLS functions for BNN deployment. 
 * Include this file in the network top function.
 * 
 *
 *****************************************************************************/


#include <hls_stream.h>
#include "ap_int.h"
#include <iostream>
#include <string>

using namespace hls;
using namespace std;

#define CASSERT_DATAFLOW(x) ;

#include "streamtools.h"
#include "dma.h"
#include "matrixvector.h"
#include "slidingwindow.h"
#include "maxpool.h"
#include "fclayer.h"
#include "convlayer.h"
#include "LUTNET_StreamingNumResConverter.h"
#include "LUTNET_sliding_window.h"
//#include "LUTNET_lut5_matrixvector.h"
#include "LUTNET_lut4_matrixvector.h"
//#include "LUTNET_lut2_matrixvector.h"
#include "LUTNET_reb_matrixvector.h"