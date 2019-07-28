all : test.exe odesolve.exe linsolve.exe odesolve_slepc.exe

clean ::
	rm -f *.o *.exe

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

# FC_FLAGS := $(filter-out -Wall,${FC_FLAGS}) -ffree-line-length-none

%.exe: %.o
	-${FLINKER} -o $@ $< ${SLEPC_LIB}
