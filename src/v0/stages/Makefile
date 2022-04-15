src_files = pipeline_test.cpp pipeline.cpp ./stage1/stage1.cpp ./stage2/stage2.cpp ./stage3/stage3.cpp ./stage4/stage4.cpp
header_files = pipeline.hpp ../config.hpp

.PHONY: all

all: pipeline_test

pipeline_test: $(src_files) $(header_files)
	g++ -Wall --std=c++11 -o pipeline_test $(src_files) -I../ -I./stage1/ -I./stage2/ -I./stage3/ -I./stage4/

.PHONY: clean

clean:
	rm pipeline_test
