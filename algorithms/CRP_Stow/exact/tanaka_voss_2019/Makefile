#
# Tanaka & Voß (2019) BRPSP solver build wrapper.
#
# Targets:
#   all      – build unrestricted variant (brpsp_bb_u)   [default]
#   restrict – build restricted   variant (brpsp_bb_r)
#   both     – build both variants
#   clean    – remove compiled objects and binaries
#
# The two variants differ only in whether -DRESTRICTED is defined at compile
# time.  Both are built from the same C source in vendor/brpsp-1.0/.

VENDOR   = vendor/brpsp-1.0
BINARY_U = $(VENDOR)/brpsp_bb_u
BINARY_R = $(VENDOR)/brpsp_bb_r

.PHONY: all restrict both clean

all: $(BINARY_U)

restrict: $(BINARY_R)

both: $(BINARY_U) $(BINARY_R)

$(BINARY_U):
	$(MAKE) -C $(VENDOR) TARGET=brpsp_bb_u

$(BINARY_R):
	$(MAKE) -C $(VENDOR) TARGET=brpsp_bb_r EXTRA_DEFS=-DRESTRICTED

clean:
	$(MAKE) -C $(VENDOR) clean
	rm -f $(BINARY_U) $(BINARY_R)
