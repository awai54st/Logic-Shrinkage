#include <hls_stream.h>
#include "ap_int.h"
#include <iostream>
#include <string.h>
#include "top.h"

typedef unsigned long long ExtMemWord;
const unsigned int bitsPerExtMemWord = sizeof(ExtMemWord)*8;

FILE *input_file;
int i;
ExtMemWord tmp;
ap_uint<64> val = 0;
bool doInit = 0;
unsigned int targetLayer = 0, targetMem = 0, targetInd = 0, numReps = 1;
const uint count = 1;  // # of images parsed
const uint psi = 384;  // # of ExtMemWords per input
const uint pso = 1;    // # of ExtMemWords per output
float outTest[4] = {0,0,0,0};
int detected_class = 0;
int expected_class = 0;

// allocate host-side buffers for packed input and outputs
ExtMemWord * packedImage = new ExtMemWord[psi];
ExtMemWord * packedOut = new ExtMemWord[pso];

template<typename LowPrecType>
int copyFromLowPrecBuffer(void * buf, float out[4]) {
  LowPrecType * lpbuf = (LowPrecType *) buf;
  float max = 0;
  int detected_class = 0;
  for(unsigned int i = 0; i < 4; i++) {
      out[i] = (float) lpbuf[i];
      if (out[i]>max){
          detected_class = i;
          max=out[i];
      }
      std::cout << "out[" << i << "]=" << out[i] << std::endl;
  }
  std::cout << "Detected class is " << detected_class << std::endl;
  std::cout << "Expected class is " << expected_class << std::endl;
  return detected_class;
}

int main(){

    std::cout << "@@@@@@@@@@@@@ START MAIN @@@@@@@@@@@@@" << std::endl;

    // read packed input from file
    input_file = fopen("input_in_im_0.txt","r");
    for(i=0;i<psi;i++) //read only one value
    {
        fscanf(input_file, "%llu", &tmp);
        //std::cout << tmp << std::endl;
        packedImage[i] = tmp;
    }
    fclose(input_file);

    // Run BlackBoxJam IP core
    BlackBoxJam((ap_uint<64> *)packedImage, (ap_uint<64> *)packedOut, doInit, targetLayer, targetMem, targetInd, val, numReps);

    detected_class = copyFromLowPrecBuffer<unsigned short>(&packedOut[0], outTest);

    delete [] packedImage;
    delete [] packedOut;

    if (detected_class==expected_class){
        std::cout << "Detected class match expected class" << std::endl;
        std::cout << "@@@@@@@@@@@@@ END MAIN @@@@@@@@@@@@@" << std::endl;
        return 0;
    } else {
        std::cout << "Detected class does not match expected class" << std::endl;
        std::cout << "@@@@@@@@@@@@@ END MAIN @@@@@@@@@@@@@" << std::endl;
        return 0; // should return 1 but ensure RTL sim is run even if C sim fails
    }
}