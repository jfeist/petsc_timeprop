all : test.exe odesolve.exe linsolve.exe odesolve_slepc.exe

clean ::
	rm -f *.o *.exe

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

# FC_FLAGS := $(filter-out -Wall,${FC_FLAGS}) -ffree-line-length-none

LASERFIELDS_DIRECTORY=/home/feist/laser/laserfields
SLEPC_INCLUDE := ${SLEPC_INCLUDE} -I${LASERFIELDS_DIRECTORY}/lib
SLEPC_LIB := ${SLEPC_LIB} -L${LASERFIELDS_DIRECTORY}/lib -llaserfields

%.exe: %.o
	-${FLINKER} -o $@ $< ${SLEPC_LIB}
