CC      ?= gcc
CFLAGS  ?= -O3 -march=native -mamx-tile -mamx-int8 -mamx-bf16 -fopenmp -ffp-contract=fast -fno-math-errno -Wall -Wextra -Wno-unused-parameter
LDLIBS  = -lm
CORE    = gemm.o amx.o bitlinear.o block.o model.o stream.o

all: train infer test dilated_walshnet

dilated_walshnet: dilated_walshnet.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

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
	rm -f *.o train infer test dilated_walshnet

.PHONY: all check clean
