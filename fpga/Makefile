#
# Copyright 2019-2020 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# makefile-generator v1.0.3
#

############################## Help Section ##############################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO)  "  make test DEVICE=<FPGA platform>"
	$(ECHO)  "     Command to run the application. This is same as 'run' target but does not have any makefile dependency."
	$(ECHO)  ""
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make host HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build host application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""

############################## Setting up Project Variables ##############################
# Points to top directory of Git repository
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
COMMON_REPO ?= ../
PWD = $(shell readlink -f .)
XF_PROJ_ROOT = $(shell readlink -f $(COMMON_REPO))

DEVICE := xilinx_u200_xdma_201830_2
TARGET := hw
HOST_ARCH := x86
SYSROOT := 

VERSION := 0
PART := fpga2
JOBS := 16

include ./utils.mk

XSA := 
ifneq ($(DEVICE), )
XSA := $(call device2xsa, $(DEVICE))
endif
TEMP_DIR := ./_x.$(TARGET).$(PART).v$(VERSION)
BUILD_DIR := ./build_dir.$(TARGET).$(PART).v$(VERSION)

# SoC variables
RUN_APP_SCRIPT = ./run_app.sh
PACKAGE_OUT = ./package.$(TARGET)

LAUNCH_EMULATOR = $(PACKAGE_OUT)/launch_$(TARGET).sh
RESULT_STRING = TEST PASSED

VPP := v++
CMD_ARGS = $(BUILD_DIR)/$(PART).xclbin
SDCARD := sd_card

include $(XF_PROJ_ROOT)/common/includes/opencl/opencl.mk
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++11
LDFLAGS += $(opencl_LDFLAGS)

ifeq ($(findstring nodma, $(DEVICE)), nodma)
$(error [ERROR]: This example is not supported for $(DEVICE).)
endif

############################## Setting up Host Variables ##############################
#Include Required Host Source Files
CXXFLAGS += -I$(XF_PROJ_ROOT)/common/includes/xcl2 -I$(XF_PROJ_ROOT)/common/includes/cmdparser -I$(XF_PROJ_ROOT)/common/includes/logger
HOST_SRCS += $(XF_PROJ_ROOT)/common/includes/xcl2/xcl2.cpp $(XF_PROJ_ROOT)/common/includes/cmdparser/cmdlineparser.cpp $(XF_PROJ_ROOT)/common/includes/logger/logger.cpp	../src/v$(VERSION)/host_$(PART).cpp
# Host compiler global settings
CXXFLAGS += -fmessage-length=0 
LDFLAGS += -lrt -lstdc++ 

ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

############################## Setting up Kernel Variables ##############################
# Kernel compiler global settings
VPP_FLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps --hls.jobs $(JOBS) --vivado.impl.jobs $(JOBS) --vivado.synth.jobs $(JOBS)
ifneq ($(TARGET), hw)
	VPP_FLAGS += -g
endif

STAGE_SRC := '../src/v$(VERSION)/stages/pipeline.cpp'
STAGE_SRC += '../src/v$(VERSION)/stages/stage1/stage1.cpp' 
STAGE_SRC += '../src/v$(VERSION)/stages/stage2/stage2.cpp' 
STAGE_SRC += '../src/v$(VERSION)/stages/stage3/stage3.cpp' 
STAGE_SRC += '../src/v$(VERSION)/stages/stage4/stage4.cpp' 
STAGE_HDRS := ../src/v$(VERSION)/config.hpp 
STAGE_HDRS += ../src/v$(VERSION)/stages/pipeline.hpp 
STAGE_HDRS += ../src/v$(VERSION)/stages/stage1/stage1.hpp
STAGE_HDRS += ../src/v$(VERSION)/stages/stage2/stage2.hpp
STAGE_HDRS += ../src/v$(VERSION)/stages/stage3/stage3.hpp
STAGE_HDRS += ../src/v$(VERSION)/stages/stage4/stage4.hpp

STAGE_INCL := -I'../src/v$(VERSION)/'
STAGE_INCL += -I'../src/v$(VERSION)/stages/' 
STAGE_INCL += -I'../src/v$(VERSION)/stages/stage1/' 
STAGE_INCL += -I'../src/v$(VERSION)/stages/stage2/' 
STAGE_INCL += -I'../src/v$(VERSION)/stages/stage3/' 
STAGE_INCL += -I'../src/v$(VERSION)/stages/stage4/' 


EXECUTABLE = ./host_$(PART)
EMCONFIG_DIR = $(TEMP_DIR)
EMU_DIR = $(SDCARD)/data/emulation

RES_DIR := ../builds/v$(VERSION).$(PART)

############################## Declaring Binary Containers ##############################
BINARY_CONTAINERS += $(BUILD_DIR)/$(PART).xclbin
BINARY_CONTAINER_vadd_OBJS += $(TEMP_DIR)/$(PART).xo

############################## Setting Targets ##############################
CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig sd_card
ifeq ($(TARGET),hw)
	mkdir -p $(RES_DIR)
	cp $(EXECUTABLE) $(RES_DIR)
	cp $(BUILD_DIR)/$(PART).xclbin $(RES_DIR)
	cp $(TEMP_DIR)/reports/link/imp/impl_1_kernel_util_routed.rpt $(RES_DIR) 
	cp $(TEMP_DIR)/reports/link/imp/impl_1_xilinx_u200_xdma_201830_2_bb_locked_timing_summary_routed.rpt $(RES_DIR) 
	cp $(TEMP_DIR)/reports/$(PART)/hls_reports/$(PART)_csynth.rpt $(RES_DIR)
	cp $(TEMP_DIR)/logs/$(PART)/$(PART)_vitis_hls.log $(RES_DIR)
endif


.PHONY: host
host: $(EXECUTABLE)

.PHONY: build
build: check-vitis $(BINARY_CONTAINERS)

.PHONY: xclbin
xclbin: build

############################## Setting Rules for Binary Containers (Building Kernels) ##############################
$(TEMP_DIR)/$(PART).xo: ../src/v$(VERSION)/stages/pipeline.cpp $(STAGE_HDRS)
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) -c -k $(PART) --temp_dir $(TEMP_DIR) $(STAGE_INCL)  -o'$@' '$<' $(STAGE_SRC)
$(BUILD_DIR)/$(PART).xclbin: $(BINARY_CONTAINER_vadd_OBJS)
	mkdir -p $(BUILD_DIR)
ifeq ($(HOST_ARCH), x86)
	$(VPP) $(VPP_FLAGS) -l $(VPP_LDFLAGS) --temp_dir $(TEMP_DIR) $(STAGE_INCL) -o'$(BUILD_DIR)/$(PART).link.xclbin' $(+)
	$(VPP) -p $(BUILD_DIR)/$(PART).link.xclbin -t $(TARGET) --platform $(DEVICE) --package.out_dir $(PACKAGE_OUT) -o $(BUILD_DIR)/$(PART).xclbin
else
	$(VPP) $(VPP_FLAGS) -l $(VPP_LDFLAGS) --temp_dir $(TEMP_DIR) -o'$(BUILD_DIR)/$(PART).xclbin' $(+)
endif

############################## Setting Rules for Host (Building Host Executable) ##############################
$(EXECUTABLE): $(HOST_SRCS) | check-xrt
		$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(STAGE_INCL)  $(STAGE_SRC)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

############################## Setting Essential Checks and Running Rules ##############################
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)
else
	$(LAUNCH_EMULATOR_CMD)
endif
else
ifeq ($(HOST_ARCH), x86)
	$(EXECUTABLE) $(CMD_ARGS)
endif
endif


.PHONY: test
test: $(EXECUTABLE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)
else
	$(LAUNCH_EMULATOR_CMD)
endif
else
ifeq ($(HOST_ARCH), x86)
ifeq ($(PART), all)
	$(EXECUTABLE) -x1 ../builds/v$(VERSION).fpga1/fpga1.xclbin -x2 ../builds/v$(VERSION).fpga2/fpga2.xclbin
else
	$(EXECUTABLE) $(RES_DIR)/$(PART).xclbin
endif
else
	$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"
endif
endif

############################## Preparing sdcard ##############################
sd_card: $(BINARY_CONTAINERS) $(EXECUTABLE) gen_run_app
ifneq ($(HOST_ARCH), x86)
		$(VPP) -p $(BUILD_DIR)/$(part).xclbin -t $(TARGET) --platform $(DEVICE) --package.out_dir $(PACKAGE_OUT) --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.sd_file $(SD_IMAGE_FILE) --package.sd_file xrt.ini --package.sd_file $(RUN_APP_SCRIPT) --package.sd_file $(EXECUTABLE) -o $(PART).xclbin
endif


############################## Cleaning Rules ##############################
# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) build_dir* sd_card*
	-$(RMDIR) package.*
	-$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin

