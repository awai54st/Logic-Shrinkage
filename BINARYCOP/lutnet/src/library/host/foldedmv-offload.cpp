/******************************************************************************
 *  Copyright (c) 2016, Xilinx, Inc.
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
 *****************************************************************************/
/******************************************************************************
 *
 *
 * @file foldedmv-offload.cpp
 *
 * Library of functions for host code and managing HW offload
 * 
 *
 *****************************************************************************/
#include "foldedmv-offload.h"
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>

using namespace tiny_cnn;
using namespace std;

//#define INCLUDE_TRANSFER_TIMES_IN_BENCHMARK

#if defined(INCLUDE_TRANSFER_TIMES_IN_BENCHMARK) || defined(RAWHLS)
#define TRANSFER_EXCL(x) ;
#define TRANSFER_INCL(x) x;
#else
#define TRANSFER_EXCL(x) x;
#define TRANSFER_INCL(x) ;
#endif


string getBNNRoot() {
  char * bnnRoot = getenv ("XILINX_BNN_ROOT");
  if(!bnnRoot)
    throw "XILINX_BNN_ROOT must be set";
  return string(bnnRoot);
}

// return in padded to a multiple of padTo
unsigned int paddedSize(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0)
    return in;
  else
    return in + padTo - (in % padTo);
}

// binarize an array of floating point values according to their sign and
// pack into a stream of bits
void binarizeAndPack(const vec_t & in, ExtMemWord * out, unsigned int inBufSize) {
  if(in.size() / bitsPerExtMemWord > inBufSize)
    throw "Not enough space in input buffer";
  // first, fill the target buffer with padding data
  memset(out, FOLDEDMV_INPUT_PADCHAR, inBufSize * sizeof(ExtMemWord));
  // now clear/set each bit position as needed
  for(unsigned int i=0; i < in.size(); i++) {
      if(in[i] >= 0) {
          // set bit
          out[i / bitsPerExtMemWord] |= ((ExtMemWord)1 << (i % bitsPerExtMemWord));
      } else {
          // clear bit
          out[i / bitsPerExtMemWord] &= ~((ExtMemWord)1 << (i % bitsPerExtMemWord));
      }
  }
}

// unpack a stream of bit and debinarize them into -1 and +1 floating point
// values (where a 0 bit is -1 and 1 bit is +1)
void unpackAndDebinarize(const ExtMemWord * in, vec_t &out) {
  for(unsigned int i=0; i < out.size(); i++) {
      if((in[i / bitsPerExtMemWord] >> (i % bitsPerExtMemWord)) & 0x1) {
          out[i] = 1;
      } else {
          out[i] = -1;
        }
    }
}

#ifdef OFFLOAD

std::vector<unsigned int> testPrebinarized_nolabel(std::vector<vec_t> & imgs, const unsigned int labelBits, float &usecPerImage) {
  // TODO support labelBits > bitsPerExtMemWord
  if(labelBits > bitsPerExtMemWord)
    throw "labelBits > bitsPerExtMemWord not yet supported";
  const unsigned int count = 1;
  cout << "Running prebinarized test for " << count << " images..." << endl;
  // compute the number of words needed to store the each img and label in binarized form
  const unsigned int psi = paddedSize(imgs[0].size(), bitsPerExtMemWord) / bitsPerExtMemWord;
  const unsigned int psl = paddedSize(labelBits, bitsPerExtMemWord) / bitsPerExtMemWord;
  // allocate buffers for binarized input and output data
  ExtMemWord * binImages = new ExtMemWord[(count * psi)];

  // binarize each image and label
  for(unsigned int i = 0; i < count; i++) {
      binarizeAndPack(imgs[i], &binImages[i * psi], psi);
  }
  unsigned int r = 0;

  // recognize
  unsigned int ok = 0, failed = 0;
  ExtMemWord * outLabel = new ExtMemWord[(count * psl)];
  TRANSFER_EXCL(thePlatform->copyBufferHostToAccel((void *)binImages, accelBufIn, sizeof(ExtMemWord)*count*psi));
  auto t1 = chrono::high_resolution_clock::now();
    FoldedMVOffloadBinarized(binImages, outLabel, count*psi, count*psl, count);
  auto t2 = chrono::high_resolution_clock::now();
  TRANSFER_EXCL(thePlatform->copyBufferAccelToHost(accelBufOut, (void *)outLabel, sizeof(ExtMemWord)*count*psl));
  // compare against labels
  uint64_t mask_output=0xFFFFFFFFFFFFFFFF;
  mask_output = mask_output>> (64-labelBits);
  for(unsigned int i = 0; i < count; i++) {
	  outLabel[i*psl]=outLabel[i*psl]&mask_output;
  }
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  std::vector<unsigned int> result;
  double internal_result;
  if (outLabel[0] == 0)
  {
		internal_result=0;
  }
  else
  {
		internal_result = log2 ((double) outLabel[0]);
  }
  for (unsigned int i = 0 ; i < 64; i++)
  {
	  if (i!=(unsigned int) internal_result)
		  result.push_back(0);
	  else
		  result.push_back(1);
  }
  delete [] outLabel;
  delete [] binImages;
  return(result);
}


std::vector<unsigned int> testPrebinarized_nolabel_multiple_images(std::vector<vec_t> & imgs, const unsigned int labelBits, float &usecPerImage) {
  // TODO support labelBits > bitsPerExtMemWord
  if(labelBits > bitsPerExtMemWord)
    throw "labelBits > bitsPerExtMemWord not yet supported";
  const unsigned int count = imgs.size();
  cout << "Running prebinarized test for " << count << " images..." << endl;
  // compute the number of words needed to store the each img and label in binarized form
  const unsigned int psi = paddedSize(imgs[0].size(), bitsPerExtMemWord) / bitsPerExtMemWord;
  const unsigned int psl = paddedSize(labelBits, bitsPerExtMemWord) / bitsPerExtMemWord;
  // allocate buffers for binarized input and output data
  ExtMemWord * binImages = new ExtMemWord[(count * psi)];

  // binarize each image and label
  for(unsigned int i = 0; i < count; i++) {
      binarizeAndPack(imgs[i], &binImages[i * psi], psi);
  }
  unsigned int r = 0;

  // recognize
  unsigned int ok = 0, failed = 0;
  ExtMemWord * outLabel = new ExtMemWord[(count * psl)];
  TRANSFER_EXCL(thePlatform->copyBufferHostToAccel((void *)binImages, accelBufIn, sizeof(ExtMemWord)*count*psi));
  auto t1 = chrono::high_resolution_clock::now();
    FoldedMVOffloadBinarized(binImages, outLabel, count*psi, count*psl, count);
  auto t2 = chrono::high_resolution_clock::now();
  TRANSFER_EXCL(thePlatform->copyBufferAccelToHost(accelBufOut, (void *)outLabel, sizeof(ExtMemWord)*count*psl));
  // compare against labels
  uint64_t mask_output=0xFFFFFFFFFFFFFFFF; // = uint64_t(uint64_t(1<<labelBits)-1);
  mask_output = mask_output>> (64-labelBits);
  for(unsigned int i = 0; i < count; i++) {
    outLabel[i*psl]=outLabel[i*psl]&mask_output;
  }
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  std::vector<unsigned int> result;
  double internal_result;
  for(unsigned int j = 0; j < count; j++)
  {
      if (outLabel[j] == 0)
      {
            internal_result=0;
      }
      else
      {
            internal_result = log2 ((double) outLabel[j]);
      }
    result.push_back((unsigned int) internal_result);
  }
  delete [] outLabel;
  delete [] binImages;
  return(result);
}


void testPrebinarized(std::vector<vec_t> & imgs, std::vector<label_t> & labels, const unsigned int labelBits) {
  // TODO support labelBits > bitsPerExtMemWord
  if(labelBits > bitsPerExtMemWord)
    throw "labelBits > bitsPerExtMemWord not yet supported";
  const unsigned int count = imgs.size();
  cout << "Running prebinarized test for " << count << " images..." << endl;
  // compute the number of words needed to store the each img and label in binarized form
  const unsigned int psi = paddedSize(imgs[0].size(), bitsPerExtMemWord) / bitsPerExtMemWord;
  const unsigned int psl = paddedSize(labelBits, bitsPerExtMemWord) / bitsPerExtMemWord;
  // allocate buffers for binarized input and output data
  ExtMemWord * binImages = new ExtMemWord[(count * psi)];
  ExtMemWord * binLabels = new ExtMemWord[(count * psl)];
  // binarize each image and label
  for(unsigned int i = 0; i < count; i++) {
      binarizeAndPack(imgs[i], &binImages[i * psi], psi);
      memset(&binLabels[i * psl], 0, sizeof(ExtMemWord)*psl);
      binLabels[i * psl] = 1 << labels[i];
  }
  unsigned int r = 0;
  cout << "Enter number of times to repeat test: " << endl;
  cin >> r;
  // recognize
  unsigned int ok = 0, failed = 0;
  ExtMemWord * outLabel = new ExtMemWord[(count * psl)];
  TRANSFER_EXCL(thePlatform->copyBufferHostToAccel((void *)binImages, accelBufIn, sizeof(ExtMemWord)*count*psi));
  auto t1 = chrono::high_resolution_clock::now();
  for(unsigned int y = 0; y < r; y++)
    FoldedMVOffloadBinarized(binImages, outLabel, count*psi, count*psl, count);
  auto t2 = chrono::high_resolution_clock::now();
  TRANSFER_EXCL(thePlatform->copyBufferAccelToHost(accelBufOut, (void *)outLabel, sizeof(ExtMemWord)*count*psl));
  // compare against labels
  uint64_t mask_output=0xFFFFFFFFFFFFFFFF; // = uint64_t(uint64_t(1<<labelBits)-1);
  mask_output = mask_output>> (64-labelBits);
  for(unsigned int i = 0; i < count; i++) {
      outLabel[i*psl]=outLabel[i*psl]&mask_output;
      if(memcmp(&outLabel[i*psl], &binLabels[i*psl], psl * sizeof(ExtMemWord)) == 0)
        ok++;
      else
        failed++;
  }
  cout << "Succeeded " << ok << " failed " << failed << " accuracy " << 100.0*(float)ok/count << "%" << endl;
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  float usecPerImage = (float)duration / (count*r);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete [] outLabel;
  delete [] binImages;
  delete [] binLabels;
}

void FoldedMVLoadLayerMem(std::string dir, unsigned int layerNo, unsigned int peCount, unsigned int linesWMem, unsigned int linesTMem)
{
  for(unsigned int pe = 0; pe < peCount; pe++) {
    // load weights
    ifstream wf(dir + "/" + to_string(layerNo) + "-" + to_string(pe) + "-weights.bin", ios::binary | ios::in);
    if(!wf.is_open())
      throw "Could not open file";
    for(unsigned int line = 0 ; line < linesWMem; line++) {
      ExtMemWord e = 0;
      wf.read((char *)&e, sizeof(ExtMemWord));
      FoldedMVMemSet(layerNo*2, pe, line, e);
    }
    wf.close();
    // load thresholds
    ifstream tf(dir + "/" + to_string(layerNo) + "-" + to_string(pe) + "-thres.bin", ios::binary | ios::in);
    if(!tf.is_open())
      throw "Could not open file";
    for(unsigned int line = 0 ; line < linesTMem; line++) {
      ExtMemWord e = 0;
      tf.read((char *)&e, sizeof(ExtMemWord));
      FoldedMVMemSet(layerNo*2 + 1, pe, line, e);
    }
    tf.close();
  }
}
#endif

#if defined(OFFLOAD) && !defined(RAWHLS)
#include "platform.hpp"
#include <vector>

DonutDriver * thePlatform = 0;
void * accelBufIn, * accelBufOut;
ExtMemWord * bufIn, * bufOut;

// register map for FoldedMV:
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x10 : Data signal of in_V
//        bit 31~0 - in_V[31:0] (Read/Write)
// 0x14 : Data signal of in_V
//        bit 31~0 - in_V[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of out_V
//        bit 31~0 - out_V[31:0] (Read/Write)
// 0x20 : Data signal of out_V
//        bit 31~0 - out_V[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of doInit
//        bit 0  - doInit[0] (Read/Write)
//        others - reserved
// 0x2c : reserved
// 0x30 : Data signal of targetLayer
//        bit 31~0 - targetLayer[31:0] (Read/Write)
// 0x34 : reserved
// 0x38 : Data signal of targetMem
//        bit 31~0 - targetMem[31:0] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of targetInd
//        bit 31~0 - targetInd[31:0] (Read/Write)
// 0x44 : reserved
// 0x48 : Data signal of val_V
//        bit 31~0 - val_V[31:0] (Read/Write)
// 0x4c : Data signal of val_V
//        bit 31~0 - val_V[63:32] (Read/Write)
// 0x50 : reserved
// 0x54 : Data signal of numImages
//        bit 31~0 - numImages[31:0] (Read/Write)
// 0x58 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

void ExecAccel() {
  // invoke accelerator and wait for result
  thePlatform->writeJamRegAddr(0x00, 1);
  while((thePlatform->readJamRegAddr(0x00) & 0x2) == 0) usleep(1);
}

// TODO this variant always assumes an 8 byte val port on the accelerator
void FoldedMVMemSet(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, ExtMemWord val) {
  // enable weight loading mode
  thePlatform->writeJamRegAddr(0x28, 1);
  // set up init data
  thePlatform->writeJamRegAddr(0x30, targetLayer);
  thePlatform->writeJamRegAddr(0x38, targetMem);
  thePlatform->writeJamRegAddr(0x40, targetInd);
  thePlatform->write64BitJamRegAddr(0x48, (AccelDblReg) val);
  // do write
  ExecAccel();
  // disable weight loading mode
  thePlatform->writeJamRegAddr(0x28, 0);
}

void FoldedMVInit(const char * attachName) {
    thePlatform = initPlatform();
    thePlatform->attach(attachName);

    // allocate input/output buffers
    // TODO should be dynamically sized based on the largest I/O
    if (!bufIn) {
        bufIn = new ExtMemWord[INPUT_BUF_ENTRIES];
        if (!bufIn) throw "Failed to allocated host buffer";
    }
    if (!bufOut) {
        bufOut = new ExtMemWord[OUTPUT_BUF_ENTRIES];
        if (!bufOut) throw "Failed to allocated host buffer";
    }
    if (!accelBufIn) {
        accelBufIn = thePlatform->allocAccelBuffer(INPUT_BUF_ENTRIES * sizeof(ExtMemWord));
        if (!accelBufIn) throw "Failed to allocate accel buffer";
        accelBufOut = thePlatform->allocAccelBuffer(OUTPUT_BUF_ENTRIES * sizeof(ExtMemWord));
        if (!accelBufOut) throw "Failed to allocate accel buffer";
    }
    // set up I/O buffer addresses for the accelerator
    thePlatform->write64BitJamRegAddr(0x10, (AccelDblReg) accelBufIn);
    thePlatform->write64BitJamRegAddr(0x1c, (AccelDblReg) accelBufOut);
    thePlatform->writeJamRegAddr(0x28, 0);
}

void FoldedMVDeinit() {
    delete bufIn;
    delete bufOut;
    bufIn = 0;
    bufOut = 0;
    if (thePlatform && accelBufIn) thePlatform->deallocAccelBuffer(accelBufIn);
    accelBufIn = 0;
    if (thePlatform && accelBufOut) thePlatform->deallocAccelBuffer(accelBufOut);
    accelBufOut = 0;
    deinitPlatform(thePlatform);
    thePlatform = 0;
}

void FoldedMVOffload(const vec_t &in,
                     vec_t &out,
                     unsigned int offloadID,
                     OffloadConvParams * convParams)
{
  // always operates on a single image per call for now -- set numImages to 1
  thePlatform->writeJamRegAddr(0x54, 1);
  // binarize input and pack into bit stream
  binarizeAndPack(in, bufIn);

  // TODO size to pad input to is max(64, PE_SYNGROUP_BITS)
  unsigned int paddedInDim = paddedSize(in.size(), bitsPerExtMemWord);
  // copy into accelerator input
  const unsigned int numInpWords = (paddedInDim / bitsPerExtMemWord);
  thePlatform->copyBufferHostToAccel((void *)bufIn, accelBufIn, sizeof(ExtMemWord)*numInpWords);

  // launch
  ExecAccel();

  // TODO add parameters to function call to control how output copy will be done
  if(offloadID == 0xdeadbeef) {
      // TODO make this controllable -- hacked in for cifar10 for 2-byte (nonbinarized activations) now
      unsigned int paddedOutDim = paddedSize(out.size() * 16, bitsPerExtMemWord);
      const unsigned int numOutWords = ( paddedOutDim / bitsPerExtMemWord);
      thePlatform->copyBufferAccelToHost(accelBufOut, (void *)bufOut, sizeof(ExtMemWord)*numOutWords);
      copyFromLowPrecBuffer<unsigned short>((void *)bufOut, out);
  } else {
      // TODO size to pad input to is max(64, NUM_PE_ELEMENTS)
      unsigned int paddedOutDim = paddedSize(out.size(), bitsPerExtMemWord);

      // copy from accelerator output
      const unsigned int numOutWords = ( paddedOutDim / bitsPerExtMemWord);
      thePlatform->copyBufferAccelToHost(accelBufOut, (void *)bufOut, sizeof(ExtMemWord)*numOutWords);

      // unpack output bits and convert output back to float
      unpackAndDebinarize(bufOut, out);
  }
}

// TODO implement batch execution version
void FoldedMVOffloadBinarized(const ExtMemWord * in, ExtMemWord * out,
                              const unsigned int inBufWords, const unsigned int outBufWords, const unsigned int numImages) {
  TRANSFER_INCL(thePlatform->copyBufferHostToAccel((void *)in, accelBufIn, sizeof(ExtMemWord)*inBufWords));
  // set number of images to stream through
  thePlatform->writeJamRegAddr(0x54, numImages);

  // launch
  ExecAccel();

  // copy from accelerator output
  TRANSFER_INCL(thePlatform->copyBufferAccelToHost(accelBufOut, (void *)out, sizeof(ExtMemWord)*outBufWords));
}

#endif
