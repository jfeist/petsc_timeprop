#include "petsc/finclude/petsc.h"

module td_type
  use petsc
  type td_data
    Mat A0, At
    Vec Atu
    PetscScalar last_At_jac
  end type
end module

PROGRAM main
  use petsc
  use laserfields, only: laserfields_read_parameters
  use td_type
  IMPLICIT NONE

  ! solve d_t u = A u

  Mat            :: J
  type(td_data)  :: user
  PetscViewer    :: binv
  Vec            :: u0
  TS             :: ts
  PetscErrorCode :: ierr
  PetscInt       :: steps
  PetscScalar    :: alpha
  integer        :: my_id, ii, ntimes
  double precision :: tt, tstart
  double precision, allocatable :: times(:)

  external :: TDLinearIFunction, TDLinearIJacobian, TDHamiltRHSFunction, TDHamiltJacFunction

  tstart = mpi_wtime()

  call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
  call MPI_COMM_RANK(PETSC_COMM_WORLD, my_id, ierr)

  if (my_id==0) write(6,*) 'time for initialization:', mpi_wtime() - tstart

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  load the input from disk
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call laserfields_read_parameters('laserfields.in')
  call load_mat(user%A0,"A0.petsc")
  call load_mat(user%At,"At.petsc")
  call load_vec(u0,"u0.petsc")
  call load_array(times,"times.in")

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  set up some storage
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ! temp vector used in Hamiltonian application
  call MatCreateVecs(user%A0,PETSC_NULL_VEC,user%Atu,ierr);CHKERRA(ierr)
  call VecSetFromOptions(user%Atu,ierr);CHKERRA(ierr)

  ! Jacobian
  call MatDuplicate(user%A0,MAT_COPY_VALUES,J,ierr);CHKERRA(ierr)
  user%last_At_jac = 1.d0
  call MatAXPY(J,user%last_At_jac,user%At,DIFFERENT_NONZERO_PATTERN,ierr);CHKERRA(ierr)

  if (my_id==0) write(0,*) 'time for loading matrices:', mpi_wtime() - tstart

  call TSCreate(PETSC_COMM_WORLD,ts,ierr);CHKERRA(ierr)
  call TSSetTime(ts,times(1),ierr);CHKERRA(ierr)
  call TSSetSolution(ts,u0,ierr);CHKERRA(ierr)
  call TSSetProblemType(ts,TS_LINEAR,ierr);CHKERRA(ierr)
  call TSSetRHSFunction(ts,PETSC_NULL_VEC,TDHamiltRHSFunction,user,ierr);CHKERRA(ierr)
  !call TSSetRHSJacobian(ts,J,J,TDHamiltJacFunction,user,ierr);CHKERRA(ierr)
  !call TSRHSJacobianSetReuse(ts,PETSC_TRUE,ierr);CHKERRA(ierr)
  !call TSSetIFunction(ts,PETSC_NULL_VEC,TDLinearIFunction,user,ierr);CHKERRA(ierr)
  !call TSSetIFunction(ts,PETSC_NULL_VEC,TSComputeIFunctionLinear,user,ierr);CHKERRA(ierr)
  !call TSSetIJacobian(ts,J,J,TDLinearIJacobian,user,ierr);CHKERRA(ierr)
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

  subroutine load_array(arr,filename)
    double precision, allocatable :: arr(:)
    character(len=*) :: filename
    integer :: nn
    open(123,file=filename,status='old',action='read')
    read(123,*) nn
    allocate(arr(nn))
    read(123,*) arr
    close(123)
  end

  subroutine load_mat(A,filename)
    Mat              :: A
    character(len=*) :: filename
    call MatCreate(PETSC_COMM_WORLD,A,ierr);CHKERRA(ierr)
    call MatSetFromOptions(A,ierr);CHKERRA(ierr)
    call PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
    call MatLoad(A,binv,ierr);CHKERRA(ierr)
    call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)
  end subroutine

  subroutine load_vec(u,filename)
    Vec              :: u
    character(len=*) :: filename
    call VecCreate(PETSC_COMM_WORLD,u,ierr);CHKERRA(ierr)
    call VecSetFromOptions(u,ierr);CHKERRA(ierr)
    call PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
    call VecLoad(u,binv,ierr);CHKERRA(ierr)
    call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)
  end subroutine
END PROGRAM main

! PetscErrorCode func (TS ts,PetscReal t,Vec u,Vec F,void *ctx);
! t	- current timestep
! u	- input vector
! F	- function vector
! ctx	- [optional] user-defined function context
subroutine TDHamiltRHSFunction(ts,t,u,F,user,ierr)
  use td_type
  use laserfields, only: get_AL
  implicit none
  ! evaluate F = (A0 + E(t)*At)*u
  TS ts
  PetscReal t
  Vec u, F
  type(td_data) user
  PetscErrorCode ierr
  PetscScalar At

  At = get_Al(t)
  call MatMult(user%A0,u,F,ierr);CHKERRA(ierr)
  call MatMult(user%At,u,user%Atu,ierr);CHKERRA(ierr)
  call VecAXPY(F,At,user%Atu,ierr);CHKERRA(ierr)
end

! PetscErrorCode func (TS ts,PetscReal t,Vec u,Mat A,Mat B,void *ctx);
subroutine TDHamiltJacFunction(ts,t,u,A,B,user,ierr)
  use td_type
  use laserfields, only: get_AL
  implicit none
  ! evaluate F = (A0 + E(t)*At)*u
  TS ts
  PetscReal t
  Vec u
  Mat A, B
  type(td_data) user
  PetscErrorCode ierr

  call MatAXPY(A,-user%last_At_jac,user%At,SUBSET_NONZERO_PATTERN,ierr);CHKERRA(ierr)
  user%last_At_jac = get_AL(t)
  call MatAXPY(A,  user%last_At_jac,user%At,SUBSET_NONZERO_PATTERN,ierr);CHKERRA(ierr)
end

subroutine TDLinearIFunction(ts,t,X,Xdot,F,user,ierr)
  use petsc
  use td_type
  implicit none
  TS ts
  PetscReal t
  Vec X,Xdot,F
  type(td_data) user
  PetscErrorCode ierr
  PetscScalar, parameter :: MONE = -1.d0
  
  ! we want to represent Xdot = (A0 + E(t) At) X in its implicit formulation, F = Xdot - (A0 + E(t) At) X
  ! F = (A0 + E(t) At) X
  call TDHamiltRHSFunction(ts,t,X,F,user,ierr);CHKERRA(ierr)
  ! F = Xdot - F = Xdot - A X
  call VecAYPX(F,MONE,Xdot,ierr);CHKERRA(ierr)
end subroutine TDLinearIFunction

subroutine TDLinearIJacobian(ts,t,X,Xdot,shift,J,Jpre,user,ierr)
  use petsc
  use laserfields
  use td_type
  implicit none
  TS ts
  PetscReal t,shift
  PetscScalar shift_scalar
  Vec X,Xdot
  Mat J,Jpre
  type(td_data) user
  PetscErrorCode ierr
  PetscScalar At
  PetscScalar, parameter :: MONE = -1.d0

  shift_scalar = shift

  ! should calculate Jacobian of F(t,X,W+a*X), equivalent to dF/dX + a*dF/dXdot
  ! F = Xdot - (A0+E(t)*At)*X -> dF/dX = -(A0+E(t)*At), dF/dXdot = I
  ! -> J = -(A0+E(t)*At) + a I

  ! J = (A0+E(t)*At)
  call MatCopy(user%A0,J,DIFFERENT_NONZERO_PATTERN,ierr);CHKERRA(ierr)
  user%last_At_jac = get_AL(t)
  call MatAXPY(J,user%last_At_jac,user%At,SUBSET_NONZERO_PATTERN,ierr);CHKERRA(ierr)

  ! J = -(A0+E(t)*At) + shift I
  call MatScale(J,MONE,ierr);CHKERRA(ierr)
  call MatShift(J,shift_scalar,ierr);CHKERRA(ierr)
  if (J /= Jpre) then
     call MatCopy(J,Jpre,SAME_NONZERO_PATTERN,ierr);CHKERRA(ierr)
  end if
end subroutine TDLinearIJacobian
