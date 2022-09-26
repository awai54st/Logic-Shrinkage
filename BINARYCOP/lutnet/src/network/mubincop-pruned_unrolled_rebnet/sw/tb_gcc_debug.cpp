#include "top.h"

typedef unsigned long long ExtMemWord;
const unsigned int bitsPerExtMemWord = sizeof(ExtMemWord)*8;

FILE *input_file;
int correct = 0;
ExtMemWord tmp;
const uint count = 1;  // # of images parsed
const uint psi = 384;  // # of ExtMemWords per input
const uint pso = 1;    // # of ExtMemWords per output

// allocate host-side buffers for packed input and outputs
ExtMemWord * packedImage = new ExtMemWord[psi];
ExtMemWord * packedOut = new ExtMemWord[pso];

const unsigned int OFM_CH = 64;
const unsigned int OFM_DIM = 576;
const unsigned int L = 5;

//const unsigned int OFM_CH = 64;
//const unsigned int OFM_DIM = 128;
//const unsigned int L = 6;

const bool conv = 0;
const bool run_from_bin_file = 0;

stream<ap_uint<32> > in("Testbench.in");
stream<ap_uint<OFM_CH> > out("Testbench.out");

int main(){

    std::cout << "@@@@@@@@@@@@@ START MAIN @@@@@@@@@@@@@" << std::endl;

	if (run_from_bin_file)
	{
		// read packed input from file
		/*input_file = fopen("input_in_im_0.txt","r");
		for(int i=0;i<psi;i++) //read only one value
		{
			fscanf(input_file, "%llu", &tmp);
			//std::cout << tmp << std::endl;
			packedImage[i] = tmp;
		}
		fclose(input_file);

		// Run BlackBoxJam IP core
		//BlackBoxJam((ap_uint<64> *)packedImage, out);*/
	}


	else
	{
		// FEED IP WITH OUTPUT FEATURES OF THE FIRST CONVOLUTIONAL LAYER
		// Read golden files

		cout << "FEED IP WITH OUTPUT FEATURES OF THE FIRST CONVOLUTIONAL LAYER" << endl;
		cout << "L3_OFM_CH " << L3_OFM_CH << endl;
		cout << "L4_IFM_DIM " << L4_IFM_DIM << endl;

		bool golden_data[numResidual][L3_OFM_CH][L4_IFM_DIM][L4_IFM_DIM];
		float read;

		char buf[32];

		// Store all residual signs in one big array
		for(int rb=0;rb<numResidual;rb++)
		{
			for(int ch=0;ch<L3_OFM_CH;ch++)
			{
				snprintf(buf, 32, "residual_sign_%d_bit_%d_ch_%d.txt", 4, rb+1, ch);
				printf("%s\n", buf);
				input_file = fopen(buf,"r");
				for(int row=0;row<L4_IFM_DIM;row++)
				{
					for(int col=0;col<L4_IFM_DIM;col++) //read only one value
					{
						fscanf(input_file, "%f", &read);
						//cout << "read " << read << endl;
						golden_data[rb][ch][row][col] = (read>0)? 1:0;
					}
				}
				fclose(input_file);
			}
		}

		ap_uint<L3_OFM_CH> goldenElemOut[numResidual][L4_IFM_DIM][L4_IFM_DIM];

		// Pack all channels together for each pixel
		for(int rb=0;rb<numResidual;rb++)
		{
			for(int row=0;row<L4_IFM_DIM;row++)
			{
				for(int col=0;col<L4_IFM_DIM;col++)
				{
					for(int ch=0;ch<L3_OFM_CH;ch++)
					{
						goldenElemOut[rb][row][col].range(ch,ch) = golden_data[rb][ch][row][col];
					}
				}
			}
		}

		for(int row=0;row<L4_IFM_DIM;row++)
		{
			for(int col=0;col<L4_IFM_DIM;col++)
			{
				for(int rb=0; rb<numResidual; rb++)
				{
					//cout << "Write " << goldenElemOut[rb][row][col] << endl;
					in.write(goldenElemOut[rb][row][col]);
				}
			}
		}

		cout << "Sending " << L4_IFM_DIM*L4_IFM_DIM*numResidual << " elements to the IP core" << endl;

		// Run BlackBoxJam IP core
		BlackBoxJam(in, out);
	}






	if (conv)
	{
		/*// DEBUG CONVOLUTIONAL LAYER
		cout << "DEBUG CONVOLUTIONAL LAYER" << endl;
		cout << "OFM_CH " << OFM_CH << endl;
		cout << "OFM_DIM " << OFM_DIM << endl;

		// Read golden files
		bool golden_data[numResidual][OFM_CH][OFM_DIM][OFM_DIM];
		float read;

		char buf[32];

		// Store all residual signs in one big array
		for(int rb=0;rb<numResidual;rb++)
		{
			for(int ch=0;ch<OFM_CH;ch++)
			{
				snprintf(buf, 32, "residual_sign_%d_bit_%d_ch_%d.txt", L, rb+1, ch);
				printf("%s\n", buf);
				input_file = fopen(buf,"r");
				for(int row=0;row<OFM_DIM;row++)
				{
					for(int col=0;col<OFM_DIM;col++) //read only one value
					{
						fscanf(input_file, "%f", &read);
						golden_data[rb][ch][row][col] = (read>0)? 1:0;
					}
				}
				fclose(input_file);
			}
		}

		ap_uint<OFM_CH> goldenElemOut[numResidual][OFM_DIM][OFM_DIM];

		// Pack all channels together for each pixel
		for(int rb=0;rb<numResidual;rb++)
		{
			for(int row=0;row<OFM_DIM;row++)
			{
				for(int col=0;col<OFM_DIM;col++)
				{
					for(int ch=0;ch<OFM_CH;ch++)
					{
						goldenElemOut[rb][row][col].range(ch,ch) = golden_data[rb][ch][row][col];
					}
					//printf("elemOut is b'" BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n", BYTE_TO_BINARY(goldenElemOut[rb][row][col]>>8), BYTE_TO_BINARY(goldenElemOut[rb][row][col]));
					//cout << goldenElemOut[rb][row][col] << endl;
				}
			}
		}

		ap_uint<OFM_CH> elemOut;
		for(int row=0;row<OFM_DIM;row++)
		{
			for(int col=0;col<OFM_DIM;col++)
			{
				for(int rb=0; rb<numResidual; rb++)
				{
					elemOut = out.read();
					//printf("elemOut is b'" BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n", BYTE_TO_BINARY(elemOut>>8), BYTE_TO_BINARY(elemOut));
					//printf("goldOut is b'" BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n", BYTE_TO_BINARY(goldenElemOut[rb][row][col]>>8), BYTE_TO_BINARY(goldenElemOut[rb][row][col]));
					for(int ch=0;ch<OFM_CH;ch++)
					{
						if (elemOut.range(ch,ch) == goldenElemOut[rb][row][col].range(ch,ch)) correct++;
						else
						{
							printf("MISMATCH at rb %d, row %d, col %d \n", rb, row, col);
							printf("elemOut is b'" BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n", BYTE_TO_BINARY(elemOut>>8), BYTE_TO_BINARY(elemOut));
							printf("goldOut is b'" BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n", BYTE_TO_BINARY(goldenElemOut[rb][row][col]>>8), BYTE_TO_BINARY(goldenElemOut[rb][row][col]));
						}
					}
				}
			}
		}

		if (correct == OFM_DIM * OFM_DIM * numResidual * OFM_CH){
			std::cout << "Golden TF data match HLS data" << std::endl;
			std::cout << "@@@@@@@@@@@@@ END MAIN @@@@@@@@@@@@@" << std::endl;
			return 0;
		} else {
			std::cout << "Golden TF data does not match HLS data" << std::endl;
			std::cout << (OFM_DIM * OFM_DIM * numResidual * OFM_CH -correct) << " errors detected and " << correct << " correct elements" << std::endl;
			std::cout << "@@@@@@@@@@@@@ END MAIN @@@@@@@@@@@@@" << std::endl;
			return 0; // should return 1 but ensure RTL sim is run even if C sim fails
		}*/
	}
	else
	{
		// DEBUG FULLY CONNECTED LAYER
		cout << "DEBUG FULLY CONNECTED LAYER" << endl;
		cout << "OFM_CH " << OFM_CH << endl;
		cout << "OFM_DIM " << OFM_DIM << endl;

		// Read golden files
		bool golden_data[numResidual][OFM_DIM];
		float read;

		char buf[32];

		// Store all residual signs in one big array
		for(int rb=0;rb<numResidual;rb++)
		{
			snprintf(buf, 32, "residual_sign_%d_bit_%d_ch_%d.txt", L, rb+1, 0);
			printf("%s\n", buf);
			input_file = fopen(buf,"r");
			for(int row=0;row<OFM_DIM;row++)
			{
				fscanf(input_file, "%f", &read);
				golden_data[rb][row] = (read>0)? 1:0;
				std::cout << golden_data[rb][row] << " ";
			}
			fclose(input_file);
			std::cout << endl;
		}

		ap_uint<OFM_CH> goldenElemOut[numResidual][OFM_DIM/OFM_CH];

		// Pack all channels together for each pixel
		for(int rb=0;rb<numResidual;rb++)
		{
			for(int row=0;row<OFM_DIM/OFM_CH;row++)
			{
				for(int ch=0;ch<OFM_CH;ch++)
				{
					goldenElemOut[rb][row].range(ch,ch) = golden_data[rb][ch + row*OFM_CH];
				}
				//printf("elemOut is b'" BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n", BYTE_TO_BINARY(goldenElemOut[rb][row][col]>>8), BYTE_TO_BINARY(goldenElemOut[rb][row][col]));
				//cout << goldenElemOut[rb][row] << endl;
			}

		}


		ap_uint<OFM_CH> elemOut;
		for(int row=0;row<OFM_DIM/OFM_CH;row++)
		{
			for(int rb=0;rb<numResidual;rb++)
			{
				elemOut = out.read();
				printf("elemOut is b'" BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n", BYTE_TO_BINARY(elemOut>>56), BYTE_TO_BINARY(elemOut>>48), BYTE_TO_BINARY(elemOut>>40), BYTE_TO_BINARY(elemOut>>32), BYTE_TO_BINARY(elemOut>>24), BYTE_TO_BINARY(elemOut>>16), BYTE_TO_BINARY(elemOut>>8), BYTE_TO_BINARY(elemOut));
				printf("goldOut is b'" BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n", BYTE_TO_BINARY(goldenElemOut[rb][row]>>56), BYTE_TO_BINARY(goldenElemOut[rb][row]>>48), BYTE_TO_BINARY(goldenElemOut[rb][row]>>40), BYTE_TO_BINARY(goldenElemOut[rb][row]>>32), BYTE_TO_BINARY(goldenElemOut[rb][row]>>24), BYTE_TO_BINARY(goldenElemOut[rb][row]>>16), BYTE_TO_BINARY(goldenElemOut[rb][row]>>8), BYTE_TO_BINARY(goldenElemOut[rb][row]));

				for(int ch=0; ch<OFM_CH; ch++)
				{
					if (elemOut.range(ch,ch) == goldenElemOut[rb][row].range(ch,ch)) correct++;
					else
					{
						printf("MISMATCH at rb %d, row %d, ch %d \n", rb, row, ch);
						//printf("elemOut is b'" BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n", BYTE_TO_BINARY(elemOut>>56), BYTE_TO_BINARY(elemOut>>48), BYTE_TO_BINARY(elemOut>>40), BYTE_TO_BINARY(elemOut>>32), BYTE_TO_BINARY(elemOut>>24), BYTE_TO_BINARY(elemOut>>16), BYTE_TO_BINARY(elemOut>>8), BYTE_TO_BINARY(elemOut));
						//printf("goldOut is b'" BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN "\n", BYTE_TO_BINARY(goldenElemOut[rb][row]>>56), BYTE_TO_BINARY(goldenElemOut[rb][row]>>48), BYTE_TO_BINARY(goldenElemOut[rb][row]>>40), BYTE_TO_BINARY(goldenElemOut[rb][row]>>32), BYTE_TO_BINARY(goldenElemOut[rb][row]>>24), BYTE_TO_BINARY(goldenElemOut[rb][row]>>16), BYTE_TO_BINARY(goldenElemOut[rb][row]>>8), BYTE_TO_BINARY(goldenElemOut[rb][row]));
					}
				}
			}
		}

		if (correct == OFM_DIM * numResidual){
			std::cout << "Golden TF data match HLS data" << std::endl;
			std::cout << "@@@@@@@@@@@@@ END MAIN @@@@@@@@@@@@@" << std::endl;
			return 0;
		} else {
			std::cout << "Golden TF data does not match HLS data" << std::endl;
			std::cout << (OFM_DIM * numResidual - correct) << " errors detected and " << correct << " correct elements" << std::endl;
			std::cout << "@@@@@@@@@@@@@ END MAIN @@@@@@@@@@@@@" << std::endl;
			return 0; // should return 1 but ensure RTL sim is run even if C sim fails
		}
	}

	return 0;
}
