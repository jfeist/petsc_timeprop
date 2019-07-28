#include "petsc/finclude/petsc.h"

PROGRAM main
  use petsc
  IMPLICIT NONE

  ! solve A x = u0

  Mat            :: A
  PetscViewer    :: binv
  Vec            :: u0
  KSP            :: ksp
  PetscErrorCode :: ierr
  PetscInt       :: steps
  integer        :: my_id, ii
  double precision :: tt, tstart
  double precision, allocatable :: times(:)

  tstart = mpi_wtime()

  call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
  call MPI_COMM_RANK(PETSC_COMM_WORLD, my_id, ierr)

  if (my_id==0) write(6,*) 'time for initialization:', mpi_wtime() - tstart

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  load the matrix from disk
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call MatCreate(PETSC_COMM_WORLD,A,ierr);CHKERRA(ierr)
  call MatSetFromOptions(A,ierr);CHKERRA(ierr)

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,"A.petsc",FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call MatLoad(A,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  call MatCreateVecs(A,PETSC_NULL_VEC,u0,ierr);CHKERRA(ierr)
  call VecSetFromOptions(u0,ierr);CHKERRA(ierr)

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'u0.petsc',FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call VecLoad(u0,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  call KSPCreate(PETSC_COMM_WORLD,ksp,ierr);CHKERRA(ierr)
  call KSPSetFromOptions(ksp,ierr);CHKERRA(ierr)
  call KSPSetOperators(ksp,A,A,ierr);CHKERRA(ierr)
  call KSPSetUp(ksp,ierr);CHKERRA(ierr)

  ! warmup
  tt = mpi_wtime()
  call KSPSolve(ksp,u0,u0,ierr);CHKERRA(ierr)
  if (my_id==0) write(6,*) 'time for first linear solve:', mpi_wtime() - tt
  tt = mpi_wtime()
  do ii = 1, 10
     call KSPSolve(ksp,u0,u0,ierr);CHKERRA(ierr)
  end do
  if (my_id==0) write(6,*) 'time for 10 linear solves:', mpi_wtime() - tt

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'xlinsolv.petsc',FILE_MODE_WRITE,binv,ierr);CHKERRA(ierr)
  call VecView(u0,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  if (my_id==0) write(6,*) 'time for total program run:', mpi_wtime() - tstart

  call PetscFinalize(ierr);CHKERRA(ierr)
END PROGRAM main
