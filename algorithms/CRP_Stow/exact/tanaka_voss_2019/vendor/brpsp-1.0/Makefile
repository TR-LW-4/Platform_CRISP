#
#  $Id: Makefile,v 1.4 2019/06/11 12:15:48 tanaka Exp tanaka $
#  $Revision: 1.4 $
#  $Date: 2019/06/11 12:15:48 $
#  $Author: tanaka $
#
.PHONY: all strip clean depend

ARCH      := $(shell uname -m)
OBJS       = heuristics.o main.o print.o problem.o solution.o solve.o timer.o
SRCS      := $(OBJS:.o=.c)

# Override TARGET and EXTRA_DEFS from the parent Makefile:
#   make TARGET=brpsp_bb_u
#   make TARGET=brpsp_bb_r EXTRA_DEFS=-DRESTRICTED
TARGET    ?= brpsp_bb

CC         = gcc
MAKEDEP    = gcc -MM

CFLAGS     = -Wall -g -O3 -fomit-frame-pointer
LIBS       = 

override DEFS += -I. $(EXTRA_DEFS)

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
main.o: main.c define.h print.h problem.h solution.h timer.h solve.h \
 heuristics.h
print.o: print.c define.h print.h problem.h solution.h timer.h
problem.o: problem.c define.h problem.h
solution.o: solution.c define.h solution.h problem.h
solve.o: solve.c define.h heuristics.h problem.h solution.h print.h \
 timer.h solve.h
timer.o: timer.c define.h timer.h problem.h
# END
