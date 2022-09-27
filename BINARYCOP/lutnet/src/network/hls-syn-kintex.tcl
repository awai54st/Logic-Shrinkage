###############################################################################
 #  Copyright (c) 2018, ACES Lab, Univesity of California San Diego, CA, US.
 #  All rights reserved.
 #
 #  Redistribution and use in source and binary forms, with or without
 #  modification, are permitted provided that the following conditions are met:
 #
 #  1.  Redistributions of source code must retain the above copyright notice,
 #     this list of conditions and the following disclaimer.
 #
 #  2.  Redistributions in binary form must reproduce the above copyright
 #      notice, this list of conditions and the following disclaimer in the
 #      documentation and/or other materials provided with the distribution.
 #
 #  3.  Neither the name of the copyright holder nor the names of its
 #      contributors may be used to endorse or promote products derived from
 #      this software without specific prior written permission.
 #
 #  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 #  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 #  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 #  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 #  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 #  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 #  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 #  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 #  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 #  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 #  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
 #
 #  IMPORTANT NOTE:
 #  This work builds upon the binary CNN libary (BNN-PYNQ) provided by the following:
 #	Copyright (c) 2016, Xilinx, Inc.
 #	link to the original library (BNN-PYNQ) : https://github.com/Xilinx/BNN-PYNQ
 #
###############################################################################
###############################################################################
 #
 #
 # @file hls-syn.tcl
 #
 # Tcl script for HLS synthesis of the target network (this script is
 # automatically launched when executing make-hw.sh script)
 #
 #
###############################################################################
# ignore the first 2 args, since Vivado HLS also passes -f tclname as args
set config_proj_name [lindex $argv 2]
puts "HLS project: $config_proj_name"
set config_hwsrcdir [lindex $argv 3]
puts "HW source dir: $config_hwsrcdir"
set directory_params [lindex $argv 4] 
set test_image [lindex $argv 5] 
set expected_result [lindex $argv 6] 

set config_bnnlibdir "$::env(XILINX_BNN_ROOT)/library/hls"
set config_bnnhostlibdir "$::env(XILINX_BNN_ROOT)/library/host"
set config_tinycnn "$::env(XILINX_BNN_ROOT)/xilinx-tiny-cnn"
puts "BNN HLS library: $config_bnnlibdir"

set config_toplevelfxn "BlackBoxJam"
# set PYNQ-Z1
#set config_proj_part "xc7z020clg400-1"
# set Kintex
set config_proj_part "xcku040-ffva1156-2-e"
set config_clkperiod 20

# set up project
open_project $config_proj_name

# TOP FILES
add_files $config_hwsrcdir/top.cpp -cflags "-std=c++0x -I$config_bnnlibdir"
add_files $config_hwsrcdir/top.h -cflags "-std=c++0x -I$config_bnnlibdir"

# TESTBENCH C++ FILES
# add_files -tb $config_hwsrcdir/../sw/main_python.cpp -cflags "-DOFFLOAD -DRAWHLS -std=c++0x -I$config_bnnhostlibdir -I$config_bnnlibdir -I$config_tinycnn -I$config_hwsrcdir"
# add_files -tb $config_bnnhostlibdir/foldedmv-offload.cpp -cflags "-DOFFLOAD -DRAWHLS -std=c++0x -I$config_bnnhostlibdir -I$config_bnnlibdir -I$config_tinycnn"
# add_files -tb $config_bnnhostlibdir/rawhls-offload.cpp -cflags "-DOFFLOAD -DRAWHLS -std=c++0x -I$config_bnnhostlibdir -I$config_bnnlibdir -I$config_tinycnn"

# add_files -tb $config_hwsrcdir/../sw/tb_gcc_debug.cpp -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
add_files -tb $config_hwsrcdir/../sw/tb_gcc.cpp -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
add_files -tb $config_hwsrcdir/../sw/input_in_im_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"

# # RESIDUAL SIGN 1
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_1.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_2.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_3.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_4.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_5.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_6.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_7.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_8.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_9.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_10.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_11.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_12.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_13.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_14.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_1_ch_15.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"

# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_1.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_2.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_3.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_4.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_5.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_6.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_7.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_8.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_9.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_10.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_11.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_12.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_13.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_14.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_1_bit_2_ch_15.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"

# # RESIDUAL SIGN 2
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_1.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_2.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_3.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_4.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_5.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_6.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_7.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_8.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_9.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_10.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_11.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_12.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_13.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_14.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_1_ch_15.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"

# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_1.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_2.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_3.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_4.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_5.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_6.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_7.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_8.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_9.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_10.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_11.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_12.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_13.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_14.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_2_bit_2_ch_15.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"

# # RESIDUAL SIGN 3
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_1.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_2.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_3.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_4.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_5.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_6.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_7.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_8.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_9.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_10.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_11.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_12.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_13.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_14.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_15.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_16.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_17.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_18.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_19.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_20.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_21.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_22.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_23.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_24.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_25.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_26.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_27.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_28.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_29.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_30.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/bin_conv_3_ch_31.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"

# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_1.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_2.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_3.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_4.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_5.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_6.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_7.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_8.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_9.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_10.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_11.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_12.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_13.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_14.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_15.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_16.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_17.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_18.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_19.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_20.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_21.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_22.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_23.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_24.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_25.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_26.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_27.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_28.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_29.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_30.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_1_ch_31.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"

# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_1.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_2.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_3.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_4.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_5.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_6.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_7.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_8.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_9.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_10.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_11.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_12.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_13.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_14.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_15.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_16.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_17.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_18.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_19.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_20.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_21.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_22.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_23.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_24.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_25.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_26.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_27.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_28.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_29.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_30.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_3_bit_2_ch_31.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"

# # RESIDUAL SIGN 4
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_1.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_2.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_3.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_4.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_5.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_6.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_7.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_8.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_9.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_10.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_11.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_12.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_13.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_14.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_15.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_16.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_17.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_18.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_19.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_20.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_21.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_22.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_23.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_24.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_25.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_26.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_27.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_28.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_29.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_30.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_1_ch_31.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"

# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_1.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_2.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_3.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_4.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_5.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_6.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_7.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_8.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_9.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_10.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_11.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_12.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_13.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_14.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_15.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_16.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_17.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_18.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_19.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_20.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_21.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_22.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_23.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_24.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_25.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_26.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_27.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_28.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_29.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_30.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_4_bit_2_ch_31.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"

# # RESIDUAL SIGN 5
# add_files -tb $config_hwsrcdir/../sw/residual_sign_5_bit_1_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_5_bit_2_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"

# # RESIDUAL SIGN 6
# add_files -tb $config_hwsrcdir/../sw/residual_sign_6_bit_1_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"
# add_files -tb $config_hwsrcdir/../sw/residual_sign_6_bit_2_ch_0.txt -cflags "-I$config_bnnlibdir -I$config_hwsrcdir"

set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

# use 64-bit AXI MM addresses
config_interface -m_axi_addr64

# syntesize and export
create_clock -period $config_clkperiod -name default
# csim_design 
#-argv "$directory_params $test_image 4 $expected_result" -compiler clang
#-compiler clang
csynth_design
# cosim_design 
#-argv "$directory_params $test_image 4 $expected_result"
# export_design -format ip_catalog
exit 0