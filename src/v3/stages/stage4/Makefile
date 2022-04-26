src_files = stage4_test.cpp stage4.cpp
header_files = stage4.hpp ../pipeline.hpp ../../config.hpp

.PHONY: all

all: stage4_test

stage4_test: $(src_files) $(header_files)
	g++ -Wall --std=c++11 -o stage4_test $(src_files) -I../../

.PHONY: clean

clean:
	rm stage4_test
