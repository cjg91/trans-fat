src_files = stage2_test.cpp stage2.cpp
header_files = stage2.hpp ../pipeline.hpp

.PHONY: all

all: stage2_test

stage2_test: $(src_files) $(header_files)
	g++ --std=c++14 -o stage2_test $(src_files)

.PHONY: clean

clean:
	rm stage2_test