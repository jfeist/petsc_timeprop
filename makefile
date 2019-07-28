all : test.exe odesolve.exe linsolve.exe

clean ::
	rm -f *.o *.exe

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# FC_FLAGS := $(filter-out -Wall,${FC_FLAGS}) -ffree-line-length-none

%.exe: %.o
	-${FLINKER} -o $@ $< ${PETSC_LIB}
