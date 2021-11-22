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

set task_name "IMAGENET"

if {$task_name == "CNV" } {
    #If CIFAR-10 or SVHN
    set config_proj_name "LUTNET_c6"
    puts "HLS project: $config_proj_name"
    set config_hwsrcdir "CIFAR10/hw"
    puts "HW source dir: $config_hwsrcdir"
    set config_proj_part "xcku115-flva1517-3-e"
} elseif {$task_name == "MNIST" } {
    #if MNIST
    set config_proj_name "LUTNET_MNIST"
    puts "HLS project: $config_proj_name"
    set config_hwsrcdir "MNIST/hw"
    puts "HW source dir: $config_hwsrcdir"
    set config_proj_part "xcku115-flva1517-3-e"
} elseif {$task_name == "IMAGENET" } {
    #if IMAGENET
    set config_proj_name "LUTNET_IMAGENET"
    puts "HLS project: $config_proj_name"
    set config_hwsrcdir "IMAGENET/hw"
    puts "HW source dir: $config_hwsrcdir"
    #set config_proj_part "xcku115-flva1517-3-e"
    set config_proj_part "xcvu9p-flga2104-3-e-EVAL"
} else {error "Unrecognised task_name."}

set config_bnnlibdir "../library/hls"
puts "BNN HLS library: $config_bnnlibdir"

set config_toplevelfxn "BlackBoxJam"
#set config_proj_part "xcku115-flva1517-3-e"
set config_clkperiod 10

# set up project
open_project $config_proj_name

add_files $config_hwsrcdir/top.cpp -cflags "-std=c++0x -I$config_bnnlibdir"

set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

# use 64-bit AXI MM addresses
config_interface -m_axi_addr64

# syntesize and export
create_clock -period $config_clkperiod -name default
# csim_design -argv "$directory_params $test_image 10 $expected_result" -compiler clang
#csynth_design
export_design -rtl verilog -format ip_catalog
exit 0
