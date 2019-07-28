#include "petsc/finclude/petsc.h"

PROGRAM main
  use petsc
  IMPLICIT NONE

  ! solve d_t u = A u

  Mat            :: A, J
  PetscViewer    :: binv
  Vec            :: u0
  TS             :: ts
  PetscErrorCode :: ierr
  PetscInt       :: steps
  integer        :: my_id, ii
  double precision :: tt, tstart
  double precision, allocatable :: times(:)

  external :: LinearIFunction, LinearIJacobian

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
  call MatDuplicate(A,MAT_COPY_VALUES,J,ierr);CHKERRA(ierr)

  call MatCreateVecs(A,PETSC_NULL_VEC,u0,ierr);CHKERRA(ierr)
  call VecSetFromOptions(u0,ierr);CHKERRA(ierr)

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'u0.petsc',FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call VecLoad(u0,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)
  
  call read_realarray('times.petsc',times)
  if (times(1)/=0.d0) then
     SETERRA(PETSC_COMM_SELF,8,"first time in times.petsc must be 0!")
  end if

  call TSCreate(PETSC_COMM_WORLD,ts,ierr);CHKERRA(ierr)
  call TSSetSolution(ts,u0,ierr);CHKERRA(ierr)
  call TSSetProblemType(ts,TS_LINEAR,ierr);CHKERRA(ierr)
  call TSSetRHSFunction(ts,PETSC_NULL_VEC,TSComputeRHSFunctionLinear,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  call TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  !call TSSetIFunction(ts,PETSC_NULL_VEC,LinearIFunction,A,ierr);CHKERRA(ierr)
  !call TSSetIJacobian(ts,J,J,LinearIJacobian,A,ierr);CHKERRA(ierr)
  call TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP,ierr);CHKERRA(ierr)
  call TSSetFromOptions(ts,ierr);CHKERRA(ierr)

  if (my_id==0) write(6,*) 'time for loading problem etc:', mpi_wtime() - tstart

  tt = mpi_wtime()
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

contains
  subroutine read_realarray(filename,x)
    character(len=*) :: filename
    Vec          :: vx, vx_seq
    PetscViewer  :: viewer
    VecScatter   :: ctx
    PetscScalar, pointer :: x_scalar(:)
    double precision, allocatable :: x(:)

    ! read from filename into a (distributed) petsc vector
    call VecCreate(PETSC_COMM_WORLD,vx,ierr);CHKERRA(ierr)
    call PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,viewer,ierr);CHKERRA(ierr)
    call VecLoad(vx,viewer,ierr);CHKERRA(ierr)
    call PetscViewerDestroy(viewer,ierr);CHKERRA(ierr)

    call VecScatterCreateToAll(vx,ctx,vx_seq,ierr);CHKERRA(ierr)
    call VecScatterBegin(ctx,vx,vx_seq,INSERT_VALUES,SCATTER_FORWARD,ierr);CHKERRA(ierr)
    call VecScatterEnd(ctx,vx,vx_seq,INSERT_VALUES,SCATTER_FORWARD,ierr);CHKERRA(ierr)
    call VecScatterDestroy(ctx,ierr);CHKERRA(ierr)

    call VecGetArrayReadF90(vx_seq,x_scalar,ierr);CHKERRA(ierr)
    allocate(x(size(x_scalar)))
    ! copy from complex to real
    x = x_scalar
    call VecRestoreArrayReadF90(vx_seq,x_scalar,ierr);CHKERRA(ierr)
    call VecDestroy(vx_seq,ierr);CHKERRA(ierr)
    call VecDestroy(vx,ierr);CHKERRA(ierr)    
  end subroutine read_realarray
END PROGRAM main

subroutine LinearIFunction(ts,t,X,Xdot,F,user,ierr)
  ! we use the (arbitrary) user context to simply pass the matrix A
  use petsc
  implicit none

  TS ts
  PetscReal t
  Vec X,Xdot,F
  Mat user
  PetscErrorCode ierr

  ! we want to represent Xdot = A X in its implicit formulation, F = Xdot - A X
  ! F = A X
  call MatMult(user,X,F,ierr);CHKERRA(ierr)
  ! F = Xdot - F = Xdot - A X
  call VecAYPX(F,(-1.d0,0.d0),Xdot,ierr);CHKERRA(ierr)
end subroutine LinearIFunction

subroutine LinearIJacobian(ts,t,X,Xdot,shift,J,Jpre,user,ierr)
  ! we use the (arbitrary) user context to simply pass the matrix A
  use petsc
  implicit none

  TS ts
  PetscReal t,shift
  PetscScalar shift_scalar
  Vec X,Xdot
  Mat J,Jpre
  Mat user
  PetscErrorCode ierr

  ! should calculate Jacobian of F(t,X,W+a*X), equivalent to dF/dX + a*dF/dXdot
  ! F = Xdot - A X -> dF/dX = -A, dF/dXdot = I
  ! J = -A + a I

  shift_scalar = shift
  call MatCopy(user,J,SAME_NONZERO_PATTERN,ierr);CHKERRA(ierr)
  call MatScale(J,(-1.d0,0.d0),ierr);CHKERRA(ierr)
  call MatShift(J,shift_scalar,ierr);CHKERRA(ierr)
  if (J /= Jpre) then
     call MatCopy(J,Jpre,SAME_NONZERO_PATTERN,ierr);CHKERRA(ierr)
  end if
end subroutine LinearIJacobian
