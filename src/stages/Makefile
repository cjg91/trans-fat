src_files = pipeline_test.cpp pipeline.cpp
header_files = pipeline.hpp stage1/stage1.hpp stage2/stage2.hpp stage3/stage3.hpp stage4/stage4.hpp

.PHONY: all

all: pipeline_test

pipeline_test: $(src_files) $(header_files)
	g++ --std=c++14 -o pipeline_test $(src_files)

.PHONY: clean

clean:
	rm pipeline_test