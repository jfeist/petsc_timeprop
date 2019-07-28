#include "petsc/finclude/petsc.h"
#include "slepc/finclude/slepc.h"

PROGRAM main
  use petsc
  use slepcmfn
  IMPLICIT NONE

  ! solve d_t u = A u

  Mat            :: A, J
  PetscViewer    :: binv
  Vec            :: u0
  MFN            :: mfn
  FN             :: fn
  PetscErrorCode :: ierr
  PetscScalar    :: dt, prescale = 1.d0
  PetscInt       :: steps
  integer        :: my_id, ii
  double precision :: tt, tstart
  double precision, allocatable :: times(:)

  tstart = mpi_wtime()

  call SlepcInitialize(PETSC_NULL_CHARACTER, ierr)
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

  call MFNCreate(PETSC_COMM_WORLD,mfn,ierr);CHKERRA(ierr)
  call MFNSetOperator(mfn,A,ierr);CHKERRA(ierr)
  call MFNGetFN(mfn,fn,ierr);CHKERRA(ierr)
  call FNSetType(fn,FNEXP,ierr);CHKERRA(ierr)
  call MFNSetFromOptions(mfn,ierr);CHKERRA(ierr)

  if (my_id==0) write(6,*) 'time for loading problem etc:', mpi_wtime() - tstart

  tt = mpi_wtime()
  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'usol.petsc',FILE_MODE_WRITE,binv,ierr);CHKERRA(ierr)
  call VecView(u0,binv,ierr);CHKERRA(ierr)
  do ii = 2, size(times)
     dt = times(ii) - times(ii-1)
     call FNSetScale(fn,dt,prescale,ierr);CHKERRA(ierr)
     call MFNSolve(mfn,u0,u0,ierr);CHKERRA(ierr)
     call VecView(u0,binv,ierr);CHKERRA(ierr)
     if (my_id==0) write(6,'(A,G12.4,A,G12.4,A,I8,A)') 'propagated until t = ', times(ii), ' in ', mpi_wtime()-tt, ' seconds'
  end do
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  if (my_id==0) then
    write(6,*) 'time for time propagation:', mpi_wtime() - tt
    write(6,*) 'time for total program run:', mpi_wtime() - tstart
  endif

  call SlepcFinalize(ierr);CHKERRA(ierr)

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
