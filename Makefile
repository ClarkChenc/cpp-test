
clean:
	rm -fr build && rm -fr bin && rm -f output.bin

build: clean
	mkdir -p bin && mkdir build && cd build && cmake .. && make -j32

run:
	cd bin && ./main 

rerun: build run