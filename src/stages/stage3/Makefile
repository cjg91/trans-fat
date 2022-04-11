src_files = stage3_gelu_test.cpp stage3.cpp
header_files = stage3.hpp ../pipeline.hpp

.PHONY: all

all: stage3_gelu_test

stage3_gelu_test: $(src_files) $(header_files)
	g++ --std=c++14 -o stage3_gelu_test $(src_files)

.PHONY: clean

clean:
	rm stage3_gelu_test