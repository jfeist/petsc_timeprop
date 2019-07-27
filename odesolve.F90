#include "petsc/finclude/petsc.h"

PROGRAM main
  use petsc
  IMPLICIT NONE

  ! solve d_t u = A u

  Mat            :: A
  PetscViewer    :: binv
  Vec            :: u0, v_times, v_times_seq
  VecScatter     :: ctx
  TS             :: ts
  PetscScalar, pointer :: times_scalar(:)
  PetscErrorCode :: ierr
  PetscInt       :: steps
  integer        :: my_id, ii
  double precision :: tt, tstart
  double precision, allocatable :: times(:)

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

  call MatCreateVecs(A,PETSC_NULL_VEC,u0,ierr);CHKERRA(ierr)
  call VecSetFromOptions(u0,ierr);CHKERRA(ierr)

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'u0.petsc',FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call VecLoad(u0,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  call VecCreate(PETSC_COMM_WORLD,v_times,ierr);CHKERRA(ierr)
  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'times.petsc',FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call VecLoad(v_times,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  call VecScatterCreateToAll(v_times,ctx,v_times_seq,ierr);CHKERRA(ierr)
  call VecScatterBegin(ctx,v_times,v_times_seq,INSERT_VALUES,SCATTER_FORWARD,ierr);CHKERRA(ierr)
  call VecScatterEnd(ctx,v_times,v_times_seq,INSERT_VALUES,SCATTER_FORWARD,ierr);CHKERRA(ierr)
  call VecScatterDestroy(ctx,ierr);CHKERRA(ierr)
  call VecGetArrayReadF90(v_times_seq,times_scalar,ierr);CHKERRA(ierr)
  allocate(times(size(times_scalar)))
  ! copy from complex to real
  times = times_scalar
  call VecRestoreArrayReadF90(v_times_seq,times_scalar,ierr);CHKERRA(ierr)
  call VecDestroy(v_times_seq,ierr);CHKERRA(ierr)
  call VecDestroy(v_times,ierr);CHKERRA(ierr)

  tt = mpi_wtime()

  call TSCreate(PETSC_COMM_WORLD,ts,ierr);CHKERRA(ierr)
  call TSSetSolution(ts,u0,ierr);CHKERRA(ierr)
  call TSSetProblemType(ts,TS_LINEAR,ierr);CHKERRA(ierr)
  call TSSetRHSFunction(ts,PETSC_NULL_VEC,TSComputeRHSFunctionLinear,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  call TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  !call TSSetIFunction(ts,PETSC_NULL_VEC,TSComputeIFunctionLinear,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  !call TSSetIJacobian(ts,A,A,TSComputeIJacobianConstant,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  call TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP,ierr);CHKERRA(ierr)
  call TSSetFromOptions(ts,ierr);CHKERRA(ierr)

  if (times(1)/=0.d0) then
     SETERRA(PETSC_COMM_SELF,8,"first time in times.petsc must be 0!")
  end if
  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'usol.petsc',FILE_MODE_WRITE,binv,ierr);CHKERRA(ierr)
  call VecView(u0,binv,ierr);CHKERRA(ierr)

  do ii = 2, size(times)
     call TSSetMaxTime(ts,times(ii),ierr);CHKERRA(ierr)
     call TSSolve(ts,u0,ierr);CHKERRA(ierr)
     call VecView(u0,binv,ierr);CHKERRA(ierr)
     call TSGetStepNumber(ts,steps,ierr);CHKERRA(ierr)
     if (my_id==0) write(6,'(A,G12.4,A,G12.4,A,I8,A)') 'propagated until t = ', times(ii), ' in ', mpi_wtime()-tt, ' seconds with ', steps, ' timesteps'
  end do

  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  if (my_id==0) then
    write(6,*) 'time for time propagation:', mpi_wtime() - tt
    write(6,*) 'time for total program run:', mpi_wtime() - tstart
  endif

  call PetscFinalize(ierr);CHKERRA(ierr)
END PROGRAM main
