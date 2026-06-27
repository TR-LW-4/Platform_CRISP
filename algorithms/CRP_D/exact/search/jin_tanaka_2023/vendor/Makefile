CC      = gcc
CFLAGS  = -O2 -Wall -Wno-unused-result
SRCS    = solve.c instance.c state.c lower_bound.c upper_bound.c move.c algorithm.c report.c timer.c
TARGET  = main-solve

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) -lm

clean:
	rm -f $(TARGET)

.PHONY: clean
