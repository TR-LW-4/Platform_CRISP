#
#  $Id: Makefile,v 1.6 2018/02/28 11:23:22 tanaka Exp tanaka $
#  $Revision: 1.6 $
#  $Date: 2018/02/28 11:23:22 $
#  $Author: tanaka $
#
.PHONY: all strip clean depend

ARCH      := $(shell uname -m)
OBJS       = heuristics.o main.o print.o problem.o solution.o solve.o timer.o
SRCS      := $(OBJS:.o=.c)

TARGET     = brp_bb

CC         = gcc
MAKEDEP    = gcc -MM

CFLAGS     = -Wall -O2
ifeq ($(ARCH),x86_64)
CFLAGS    += -march=corei7
else
CFLAGS    += -march=i686
endif

LIBS       = 

override DEFS += -I. # -DDEBUG -DUSE_CLOCK

all:: $(TARGET)

.c.o:
	$(CC) $(CFLAGS) $(DEFS) -c $<

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(DEFS) -o $@ $(OBJS) $(LIBS)

strip:: $(TARGET)
	@strip $(TARGET)

clean:
	rm -f $(TARGET) $(OBJS) *~ *.bak #*

depend:
	@sed -i -e "/^# START/,/# END/d" Makefile
	@echo "# START" >> Makefile
	@$(MAKEDEP) $(DEFS) $(SRCS) >> Makefile
	@echo "# END" >> Makefile

# START
heuristics.o: heuristics.c define.h heuristics.h problem.h solution.h \
 print.h timer.h
main.o: main.c define.h description.h print.h problem.h solution.h \
 timer.h solve.h
print.o: print.c define.h print.h problem.h solution.h timer.h
problem.o: problem.c define.h problem.h
solution.o: solution.c define.h solution.h problem.h
solve.o: solve.c define.h heuristics.h problem.h solution.h print.h \
 timer.h solve.h
timer.o: timer.c define.h timer.h problem.h
# END
