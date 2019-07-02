all : test.exe

clean ::
	rm -f *.o *.exe

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# FC_FLAGS := $(filter-out -Wall,${FC_FLAGS}) -ffree-line-length-none

test.exe : test.o
	-${FLINKER} -o test.exe test.o ${PETSC_LIB}
