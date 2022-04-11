src_files = stage4_test.cpp stage4.cpp
header_files = stage4.hpp ../pipeline.hpp

.PHONY: all

all: stage1_test

stage1_test: $(src_files) $(header_files)
	g++ --std=c++14 -o stage1_test $(src_files)

.PHONY: clean

clean:
	rm stage4_test