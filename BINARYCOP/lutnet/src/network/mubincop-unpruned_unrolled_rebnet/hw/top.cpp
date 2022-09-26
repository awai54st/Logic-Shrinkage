/******************************************************************************
 *  Copyright (c) 2018, ACES Lab, Univesity of California San Diego, CA, US.
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
 *  This work builds upon the binary CNN libary (BNN-PYNQ) provided by the following:
 *	Copyright (c) 2016, Xilinx, Inc.
 *	link to the original library (BNN-PYNQ) : https://github.com/Xilinx/BNN-PYNQ
 *
 *****************************************************************************/
/******************************************************************************
 *
 *
 * @file top.cpp
 *
 * HLS Description of the BNN, with axi-lite based parameter loading (DoMemInit)
 * and  dataflow architecture of the image inference (DoCompute)
 *
 *
 *****************************************************************************/

#include "top.h"

void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, ap_uint<64> val) {
	// do nothing since in this ReBNet implementation the parameters are baked into the bitfile
}

void DoCompute(ap_uint<64> * in, ap_uint<64> * out, const unsigned int numReps) {
#pragma HLS DATAFLOW

	// --- Input conversion streams ---
	// mem2stream to StreamingDataWidthConverter_Batch
	stream<ap_uint<64> > inter0("DoCompute.inter0");
	// StreamingDataWidthConverter_Batch to StreamingDataWidthConverter_Batch
	stream<ap_uint<192> > inter0_1("DoCompute.inter0_1");
	// StreamingDataWidthConverter_Batch to Conv1_1 layer
	stream<ap_uint<24> > inter0_2("DoCompute.inter0_2");
	#pragma HLS STREAM variable=inter0_2 depth=256

	// --- µ-CNV model streams ---
	// Conv1_1 layer to Conv1_2 layer
	stream<ap_uint<16> > inter1("DoCompute.inter1");
	// Conv1_2 layer to MaxPool after Conv1 group
	#pragma HLS STREAM variable=inter1 depth=256
	stream<ap_uint<16> > inter2("DoCompute.inter2");
	//  MaxPool after Conv1 group to Conv2_1 layer
	stream<ap_uint<16> > inter3("DoCompute.inter3");
	#pragma HLS STREAM variable=inter3 depth=256

	//  Conv2_1 layer to Conv2_2 layer
	stream<ap_uint<32> > inter4("DoCompute.inter4");
	#pragma HLS STREAM variable=inter4 depth=256
	// Conv2_2 layer to MaxPool after Conv2 group
	stream<ap_uint<32> > inter5("DoCompute.inter5");
	//  MaxPool after Conv2 group to Conv3_1 layer
	stream<ap_uint<32> > inter6("DoCompute.inter6");
	#pragma HLS STREAM variable=inter6 depth=256
// This is where we implement LUTNet
    stream<ap_uint<32*2> > inter6_1("DoCompute.inter6_1");
    #pragma HLS STREAM variable=inter6_1 depth=256
    stream<ap_uint<32*2> > inter6_2("DoCompute.inter6_2");
    #pragma HLS STREAM variable=inter6_2 depth=256
    stream<ap_uint<64*2> > inter6_3("DoCompute.inter6_3");
    #pragma HLS STREAM variable=inter6_3 depth=256
// Back to REBNet 

	//  Conv3_1 layer to FC1 layer
	stream<ap_uint<64> > inter7("DoCompute.inter7");
	#pragma HLS STREAM variable=inter7 depth=256
	//  FC1 layer to FC2 layer
	stream<ap_uint<64> > inter8("DoCompute.inter8");
	#pragma HLS STREAM variable=inter8 depth=256

	// --- Output conversion streams ---
	//  FC2 layer to stream2mem
	stream<ap_uint<64> > memOutStrm("DoCompute.memOutStrm");

	const unsigned int inBits = 32*32*3*8;	// image shape dependent = 24576
	const unsigned int outBits = L6_MH*16;	// tiling factor dependent = 64

	/*std::cout << "================================================================" << std::endl;
	std::cout << "Do compute is called" << std::endl;

	std::cout << "L0_PE=" << L0_PE << "\tL0_SIMD=" << L0_SIMD << std::endl;
	std::cout << "L1_PE=" << L1_PE << "\tL1_SIMD=" << L1_SIMD << std::endl;
	std::cout << "L2_PE=" << L2_PE << "\tL2_SIMD=" << L2_SIMD << std::endl;
	std::cout << "L3_PE=" << L3_PE << "\tL3_SIMD=" << L3_SIMD << std::endl;
	std::cout << "L4_PE=" << L4_PE << "\tL4_SIMD=" << L4_SIMD << std::endl;
	std::cout << "L5_PE=" << L5_PE << "\tL5_SIMD=" << L5_SIMD << std::endl;
	std::cout << "L6_PE=" << L6_PE << "\tL6_SIMD=" << L6_SIMD << std::endl;*/



	//std::cout << "Stream2Mem_Batch is called" << std::endl;
	Mem2Stream_Batch<64, inBits/8>(in, inter0, numReps);
	StreamingDataWidthConverter_Batch<64, 192, (32*32*3*8) / 64>(inter0, inter0_1, numReps);
	StreamingDataWidthConverter_Batch<192, 24, (32*32*3*8) / 192>(inter0_1, inter0_2, numReps);

	// --- CONV LAYERS ---
	// Conv1_1 layer
	//std::cout << "\n Conv1_1 layer is called" << std::endl;
	StreamingFxdConvLayer_Batch<L0_K, L0_IFM_CH, L0_IFM_DIM, L0_OFM_CH, L0_OFM_DIM, 8, 1, L0_SIMD, L0_PE, 28, 16, L0_WMEM,	L0_TMEM, numResidual>(inter0_2, inter1, weightMem0, thresMem0, alphaMem0, means_out0, numReps);
	// Conv1_2 layer
	//std::cout << "\n Conv1_2 layer is called" << std::endl;
	StreamingConvLayer_Batch<L1_K, L1_IFM_CH, L1_IFM_DIM, L1_OFM_CH, L1_OFM_DIM, L1_SIMD, L1_PE, 16, 28, 16, L1_WMEM, L1_TMEM, numResidual>(inter1, inter2, weightMem1, thresMem1, alphaMem1, means_out1, numReps);
	// MaxPool after Conv1 group
	//std::cout << "\n MaxPool after Conv1 group layer is called" << std::endl;
	StreamingMaxPool_Batch<L1_OFM_DIM, 2, L1_OFM_CH, numResidual>(inter2, inter3, numReps);

	// Conv2_1 layer
	//std::cout << "\n Conv2_1 layer is called" << std::endl;
	StreamingConvLayer_Batch<L2_K, L2_IFM_CH, L2_IFM_DIM, L2_OFM_CH, L2_OFM_DIM, L2_SIMD, L2_PE, 16, 28, 16, L2_WMEM, L2_TMEM, numResidual>(inter3, inter4, weightMem2, thresMem2, alphaMem2, means_out2, numReps);
	// Conv2_2 layer
	//std::cout << "\n Conv2_2 layer is called" << std::endl;
	StreamingConvLayer_Batch<L3_K, L3_IFM_CH, L3_IFM_DIM, L3_OFM_CH, L3_OFM_DIM, L3_SIMD, L3_PE, 16, 28, 16, L3_WMEM, L3_TMEM, numResidual>(inter4, inter5, weightMem3, thresMem3, alphaMem3, means_out3, numReps);
	// MaxPool after Conv2 group
	//std::cout << "\n MaxPool after Conv2 group layer is called" << std::endl;
	StreamingMaxPool_Batch<L3_OFM_DIM, 2, L3_OFM_CH, numResidual>(inter5, inter6, numReps);

	// Conv3_1 layer (LUTNet layers)
// This is where we implement LUTNet
	//std::cout << "\n LUTNET_StreamingNumResConverter<32, 32*2, 25, 2>(inter6, inter6_1) is called" << std::endl;
    LUTNET_StreamingNumResConverter<32, 32*2, L4_IFM_DIM*L4_IFM_DIM, 2>(inter6, inter6_1, numReps);
	//std::cout << "\n LUTNET_SlidingWindow is called" << std::endl;
    LUTNET_SlidingWindow<L4_K, L4_IFM_CH, L4_IFM_DIM, L4_OFM_DIM, 2, 1>(inter6_1, inter6_2, L4_K, L4_IFM_DIM, L4_OFM_DIM, 0, numReps);
	//std::cout << "\n LUTNET_REBNETMV is called" << std::endl;
    LUTNET_REBNETMV<L4_IFM_DIM, L4_IFM_DIM, L4_IFM_CH, L4_OFM_DIM, L4_OFM_DIM, L4_OFM_CH, L4_K, 1, numResidual, 28, 16, hls::stream<ap_uint<32*2>>, hls::stream<ap_uint<64*2>>>(inter6_2, inter6_3, thresh_conv4, alpha_conv4, next_layer_means_conv4, fanin_conv4, numReps);
	//std::cout << "\n LUTNET_StreamingNumResConverter<64*2, 64, 1*1, 2>(inter6_3, inter7) is called" << std::endl;
    LUTNET_StreamingNumResConverter<64*2, 64, 9, 2>(inter6_3, inter7, numReps);
// Back to REBNet 
	
	// --- FC LAYERS ---
	// FC1
	//std::cout << "\n FC1 layer is called" << std::endl;
	StreamingFCLayer_Batch<64, 64, L5_SIMD, L5_PE, 16, 28, 16 , L5_MW, L5_MH, L5_WMEM, L5_TMEM, numResidual>(inter7, inter8, weightMem5, thresMem5, alphaMem5, means_out5, numReps);
	// FC2 (no activation)
	//std::cout << "\n FC2 layer is called" << std::endl;
	StreamingFCLayer_NoActivation_Batch<64, 64, L6_SIMD, L6_PE, 16, 28, 16, L6_MW, L6_MH, L6_WMEM, numResidual>(inter8, memOutStrm, weightMem6, alphaMem6, numReps);

	//std::cout << "\n Stream2Mem_Batch is called" << std::endl;
	Stream2Mem_Batch<64, outBits/8>(memOutStrm, out, numReps);

	//*out = 1;

	//std::cout << "\nDo compute done" << std::endl;
	//std::cout << "================================================================" << std::endl;
}

void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, ap_uint<64> val, unsigned int numReps) {

	// pragmas for MLBP jam interface
	// signals to be mapped to the AXI Lite slave port
	#pragma HLS INTERFACE s_axilite port=return bundle=control
	#pragma HLS INTERFACE s_axilite port=doInit bundle=control
	#pragma HLS INTERFACE s_axilite port=targetLayer bundle=control
	#pragma HLS INTERFACE s_axilite port=targetMem bundle=control
	#pragma HLS INTERFACE s_axilite port=targetInd bundle=control
	#pragma HLS INTERFACE s_axilite port=val bundle=control
	#pragma HLS INTERFACE s_axilite port=numReps bundle=control
	// signals to be mapped to the AXI master port (hostmem)
	#pragma HLS INTERFACE m_axi offset=slave port=in bundle=hostmem depth=384
	#pragma HLS INTERFACE s_axilite port=in bundle=control
	#pragma HLS INTERFACE m_axi offset=slave port=out bundle=hostmem depth=384
	#pragma HLS INTERFACE s_axilite port=out bundle=control

	// partition PE arrays
	#pragma HLS ARRAY_PARTITION variable=weightMem0 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=thresMem0 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=weightMem1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=thresMem1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=weightMem2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=thresMem2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=weightMem3 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=thresMem3 complete dim=1
	// #pragma HLS ARRAY_PARTITION variable=weightMem4 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=thresh_conv4 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=weightMem5 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=thresMem5 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=weightMem6 complete dim=1

	#pragma HLS ARRAY_PARTITION variable=alphaMem0 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=alphaMem1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=alphaMem2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=alphaMem3 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=alpha_conv4 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=alphaMem5 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=alphaMem6 complete dim=1

	#pragma HLS ARRAY_PARTITION variable=means_out0 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=means_out1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=means_out2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=means_out3 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=next_layer_means_conv4 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=means_out5 complete dim=1

	#pragma HLS ARRAY_PARTITION variable=fanin_conv4 complete dim=1

	// resource directives
	//#pragma HLS RESOURCE variable=thresMem4 core=RAM_S2P_LUTRAM
	//#pragma HLS RESOURCE variable=thresMem5 core=RAM_S2P_LUTRAM
	//#pragma HLS RESOURCE variable=means_out4 core=RAM_S2P_LUTRAM
	//#pragma HLS RESOURCE variable=means_out5 core=RAM_S2P_LUTRAM
	
	//DoCompute(in, out, numReps);

	if (doInit) {
		DoMemInit(targetLayer, targetMem, targetInd, val);
	} else {
		DoCompute(in, out, numReps);
	}
}