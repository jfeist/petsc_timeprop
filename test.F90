#include "petsc/finclude/petsc.h"

PROGRAM main
  use petsc
  IMPLICIT NONE

  Mat            :: HH, HZ
  PetscViewer    :: binv
  PetscScalar    :: E0, z0
  Vec            :: vec0, zvec0, Hvec0
  PetscErrorCode :: ierr
  integer        :: my_id
  double precision :: tt

  call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
  call MPI_COMM_RANK(PETSC_COMM_WORLD, my_id, ierr)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  load the matrix from disk
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call MatCreate(PETSC_COMM_WORLD,HH,ierr);CHKERRA(ierr)
  call MatCreate(PETSC_COMM_WORLD,HZ,ierr);CHKERRA(ierr)

  call MatSetFromOptions(HH,ierr);CHKERRA(ierr)
  call MatSetFromOptions(HZ,ierr);CHKERRA(ierr)

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,"HH.petsc",FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call MatLoad(HH,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,"HZ.petsc",FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call MatLoad(HZ,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  call MatCreateVecs(HH,PETSC_NULL_VEC, vec0,ierr);CHKERRA(ierr)
  call MatCreateVecs(HH,PETSC_NULL_VEC,zvec0,ierr);CHKERRA(ierr)
  call MatCreateVecs(HH,PETSC_NULL_VEC,Hvec0,ierr);CHKERRA(ierr)

  call VecSetFromOptions( vec0,ierr);CHKERRA(ierr)
  call VecSetFromOptions(zvec0,ierr);CHKERRA(ierr)
  call VecSetFromOptions(Hvec0,ierr);CHKERRA(ierr)

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'vec0.petsc',FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call VecLoad(vec0,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  ! warmup CUDA (compile CUDA kernels?)
  call MatMult(HH,vec0,Hvec0,ierr);CHKERRA(ierr)
  call MatMult(HZ,vec0,zvec0,ierr);CHKERRA(ierr)
  call VecDot(vec0,Hvec0,E0,ierr);CHKERRA(ierr)
  call VecDot(vec0,zvec0,z0,ierr);CHKERRA(ierr)

  tt = mpi_wtime()
  call MatMult(HH,vec0,Hvec0,ierr);CHKERRA(ierr)
  call MatMult(HZ,vec0,zvec0,ierr);CHKERRA(ierr)
  call VecDot(vec0,Hvec0,E0,ierr);CHKERRA(ierr)
  call VecDot(vec0,zvec0,z0,ierr);CHKERRA(ierr)
  if (my_id==0) write(6,*) 'time taken:', mpi_wtime() - tt
  if (my_id==0) write(6,*) 'E0 =', E0
  if (my_id==0) write(6,*) '<z> =', z0

  call PetscFinalize(ierr);CHKERRA(ierr)
END PROGRAM main
