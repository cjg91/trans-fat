src_files = stage1_test.cpp stage1.cpp
header_files = stage1.hpp ../pipeline.hpp

.PHONY: all

all: stage1_test

stage1_test: $(src_files) $(header_files)
	g++ --std=c++14 -o stage1_test $(src_files)

.PHONY: clean

clean:
	rm stage1_test