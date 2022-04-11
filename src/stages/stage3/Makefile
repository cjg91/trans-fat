src_files = stage3_test.cpp stage3.cpp
header_files = stage3.hpp ../pipeline.hpp

.PHONY: all

all: stage3_test

stage3_test: $(src_files) $(header_files)
	g++ -Wall --std=c++14 -o stage3_test $(src_files)

.PHONY: clean

clean:
	rm stage3_test