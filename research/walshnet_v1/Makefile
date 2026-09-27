CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -ffp-contract=fast -fno-math-errno -Wall -Wextra -Wno-unused-parameter
LDLIBS  = -lm
CORE    = gemm.o bitlinear.o walsh.o mlp.o logic.o clopen.o model.o

all: train infer test

%.o: %.c wn.h
	$(CC) $(CFLAGS) -c $< -o $@

train: train.o $(CORE)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)
infer: infer.o $(CORE)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)
test: test.o $(CORE)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

check: test
	./test

clean:
	rm -f *.o train infer test

.PHONY: all check clean
