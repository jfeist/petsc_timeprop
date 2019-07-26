#include "petsc/finclude/petsc.h"

PROGRAM main
  use petsc
  IMPLICIT NONE

  ! solve d_t u = A u

  Mat            :: A
  PetscViewer    :: binv
  Vec            :: u0, utmp
  TS             :: ts
  PetscScalar    :: A0
  PetscErrorCode :: ierr
  integer        :: my_id
  double precision :: tt, tstart

  tstart = mpi_wtime()

  call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
  call MPI_COMM_RANK(PETSC_COMM_WORLD, my_id, ierr)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  load the matrix from disk
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call MatCreate(PETSC_COMM_WORLD,A,ierr);CHKERRA(ierr)
  call MatSetFromOptions(A,ierr);CHKERRA(ierr)

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,"A.petsc",FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call MatLoad(A,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  call MatCreateVecs(A,PETSC_NULL_VEC,utmp,ierr);CHKERRA(ierr)
  call VecSetFromOptions(utmp,ierr);CHKERRA(ierr)

  call MatCreateVecs(A,PETSC_NULL_VEC,u0,ierr);CHKERRA(ierr)
  call VecSetFromOptions(u0,ierr);CHKERRA(ierr)

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'u0.petsc',FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call VecLoad(u0,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  ! warmup CUDA (compile CUDA kernels?)
  call MatMult(A,u0,utmp,ierr);CHKERRA(ierr)
  ! WARNING: PETSc conjugates the SECOND vector argument, i.e., to get <x|y>, we have to call VecDot(y,x)
  ! https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecDot.html
  call VecDot(utmp,u0,A0,ierr);CHKERRA(ierr)

  tt = mpi_wtime()
  call MatMult(A,u0,utmp,ierr);CHKERRA(ierr)
  call VecDot(utmp,u0,A0,ierr);CHKERRA(ierr)
  
  if (my_id==0) then
    write(6,*) 'time for MatMult:', mpi_wtime() - tt
    write(6,*) '<u0|A|u0> =', A0
  endif

  tt = mpi_wtime()

  call TSCreate(PETSC_COMM_WORLD,ts,ierr);CHKERRA(ierr)
  call TSSetSolution(ts,u0,ierr);CHKERRA(ierr)
  call TSSetProblemType(ts,TS_LINEAR,ierr);CHKERRA(ierr)
  call TSSetRHSFunction(ts,PETSC_NULL_VEC,TSComputeRHSFunctionLinear,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  call TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  !call TSSetIFunction(ts,PETSC_NULL_VEC,TSComputeIFunctionLinear,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  !call TSSetIJacobian(ts,A,A,TSComputeIJacobianConstant,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  call TSSetFromOptions(ts,ierr);CHKERRA(ierr)

  call TSSolve(ts,u0,ierr);CHKERRA(ierr)

  if (my_id==0) then
    write(6,*) 'time for time propagation:', mpi_wtime() - tt
    write(6,*) 'time for total program run:', mpi_wtime() - tstart
  endif

  call PetscFinalize(ierr);CHKERRA(ierr)
END PROGRAM main
