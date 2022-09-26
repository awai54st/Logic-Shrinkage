#include <ap_int.h>

template<unsigned int InWidth,		// width of input stream
		unsigned int OutWidth,		// width of output stream
		unsigned int NumInWords,	// number of input words to process
		unsigned int NumRes         // number of residual levels used
>
void LUTNET_StreamingNumResConverter(stream<ap_uint<InWidth> > & in, stream<ap_uint<OutWidth> > & out, const unsigned int numReps) {

	// cout << "LUTNET_StreamingNumResConverter is called" << endl;

	// std::cout << "InWidth: " << InWidth << "\t OutWidth: " << OutWidth << std::endl;
	// std::cout << "NumInWords: " << NumInWords << "\t NumRes: " << NumRes << std::endl;
	int count_read_in = 0, count_write_out = 0;

	const unsigned int totalIters = NumInWords * numReps;


	if (InWidth > OutWidth) 
	{
		CASSERT_DATAFLOW(InWidth % OutWidth == 0);
		ap_uint<InWidth> ei;
		ap_uint<OutWidth> eo[NumRes];
		for (unsigned int t = 0; t < totalIters; t++) 
		{
		#pragma HLS PIPELINE II=1
			ei = in.read();
			count_read_in++;
			for(int res_ind=0;res_ind<NumRes;res_ind++)
			{
			#pragma HLS UNROLL
				for(int i=0; i<OutWidth; i++)
				{
					eo[res_ind].range(i,i) = ei.range(NumRes*i+res_ind, NumRes*i+res_ind);
					//cout << "eo[" << res_ind << "].range("<< i << ")=ei.range(" << NumRes*i+res_ind << ")=" << eo[res_ind].range(i,i) << endl;
				}
			}
			// eo[0] is LSB
			// eo[NumRes-1] is MSB
			for(int res_ind=NumRes-1; res_ind>=0; res_ind--)
			{
				out.write(eo[res_ind]);
				count_write_out++;
			}
		}
	}
	else if (InWidth == OutWidth)
	{
		for (unsigned int t = 0; t < totalIters * NumRes; t++)
		{
			// should not be used
			#pragma HLS PIPELINE II=1
			ap_uint<InWidth> e = in.read();
			count_read_in++;
			out.write(e);
			count_write_out++;
		}
	}
	else
	{
		CASSERT_DATAFLOW(OutWidth % InWidth == 0);
		ap_uint<OutWidth> eo;
		for (unsigned int t = 0; t < totalIters; t++)
		{
		#pragma HLS PIPELINE II=1
			for(int res_ind=NumRes-1;res_ind>=0;res_ind--)
			{
				ap_uint<InWidth> ei = in.read();
				count_read_in++;
				for(int i=0; i<InWidth; i++)
				{
					eo.range(NumRes*i+res_ind, NumRes*i+res_ind) = ei.range(i,i);
					//cout << "eo.range("<< NumRes*i+res_ind << ")=ei.range(" << i << ")=" << ei.range(i,i) << endl;
				}
				if (res_ind==0) 
				{
					out.write(eo);
					count_write_out++;
				}
			}
		}
	}
	// std::cout << "reads: " << count_read_in << "\t writes: " << count_write_out << std::endl;
}
