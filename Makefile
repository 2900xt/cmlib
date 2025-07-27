plotTest:
	g++ tests/plotTest.cpp -Iinclude
	./a.out
	rm a.out
	rm data/tmp/*

linReg:
	g++ tests/linearReg.cpp -Iinclude
	./a.out
	rm a.out
	rm data/tmp/*