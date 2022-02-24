# Daniel Stumpp
# Colman Glagovich
# Owen Lucas
#
# PCA22' Transformer Acceleration
#
# Makefile adapted from Xilinx examples
#

MK_PATH := $(abspath $lastword $(MAKEFILE_LIST))
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
COMMON_REPO := ./$(shell bash -c 'export MK_PATH=$(MK_PATH)')
PWD = $(shell readlink -f .)


###################### HELP Output #########################
.PHONY: help
help:
	@echo $(MK_PATH)
	@echo 	""
	@echo 	"Makefile Usage:" 
	@echo	"	make all"
	@echo 	"		- Makes everything..."
	@echo	""
	@echo 	""

##### Aditional Includes ####
include $(COMMON_REPO)/vitis_includes/opencl/opencl.mk 

#### Setting Project Variables #### TODO
TEMP_DIR := 
BUILD_DIR := 



#### Setting Host Variables ### 
CXXFLAGS := 

#################### Build Hardare ########################
.PHONY: build
build: 
	@echo 	"TODO"
	



