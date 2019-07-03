#include "petsc/finclude/petsc.h"

PROGRAM main
  use petsc
  IMPLICIT NONE

  Mat            :: HH, HZ, miH, Hperm, Zperm
  PetscViewer    :: binv
  PetscScalar    :: E0, z0, miE0
  Vec            :: vec0, zvec0, Hvec0, vectmp
  MatPartitioning:: part
  IS             :: is,isg,perm
  VecScatter     :: scatt
  TS             :: ts
  PetscErrorCode :: ierr
  integer        :: my_id
  double precision :: tt
  double complex, parameter :: IU = (0.d0,1.d0)

  call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
  call MPI_COMM_RANK(PETSC_COMM_WORLD, my_id, ierr)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  load the matrix from disk
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  call MatCreate(PETSC_COMM_WORLD,HH,ierr);CHKERRA(ierr)
  call MatCreate(PETSC_COMM_WORLD,HZ,ierr);CHKERRA(ierr)
  call MatCreate(PETSC_COMM_WORLD,miH,ierr);CHKERRA(ierr)

  call MatSetFromOptions(HH,ierr);CHKERRA(ierr)
  call MatSetFromOptions(HZ,ierr);CHKERRA(ierr)

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,"HH.petsc",FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call MatLoad(HH,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  call MatCreateVecs(HH,PETSC_NULL_VEC,vectmp,ierr);CHKERRA(ierr)
  call VecSetFromOptions(vectmp,ierr);CHKERRA(ierr)

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,"HZ.petsc",FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call MatLoad(HZ,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)

  call MatPartitioningCreate(MPI_COMM_WORLD,part,ierr);CHKERRA(ierr)
  call MatPartitioningSetAdjacency(part,HH,ierr);CHKERRA(ierr)
  !call MatPartitioningSetType(part,MATPARTITIONINGPARMETIS,ierr);CHKERRA(ierr)
  call MatPartitioningSetFromOptions(part,ierr);CHKERRA(ierr)
  call MatPartitioningApply(part,is,ierr);CHKERRA(ierr)
  call MatPartitioningDestroy(part,ierr);CHKERRA(ierr)

  call ISPartitioningToNumbering(is,isg,ierr);CHKERRA(ierr)
  call ISBuildTwoSided(is,PETSC_NULL_IS,perm,ierr);CHKERRA(ierr)

  call MatCreateSubMatrix(HH,perm,perm,MAT_INITIAL_MATRIX,Hperm,ierr);CHKERRA(ierr)
  call MatDestroy(HH,ierr);
  HH = Hperm;

  call MatCreateSubMatrix(HZ,perm,perm,MAT_INITIAL_MATRIX,Zperm,ierr);CHKERRA(ierr)
  call MatDestroy(HZ,ierr);
  HZ = Zperm;

  call MatCreateVecs(HH,PETSC_NULL_VEC, vec0,ierr);CHKERRA(ierr)
  call MatCreateVecs(HH,PETSC_NULL_VEC,zvec0,ierr);CHKERRA(ierr)
  call MatCreateVecs(HH,PETSC_NULL_VEC,Hvec0,ierr);CHKERRA(ierr)

  call VecSetFromOptions( vec0,ierr);CHKERRA(ierr)
  call VecSetFromOptions(zvec0,ierr);CHKERRA(ierr)
  call VecSetFromOptions(Hvec0,ierr);CHKERRA(ierr)

  call PetscViewerBinaryOpen(PETSC_COMM_WORLD,'vec0.petsc',FILE_MODE_READ,binv,ierr);CHKERRA(ierr)
  call VecLoad(vectmp,binv,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(binv,ierr);CHKERRA(ierr)
  call VecScatterCreate(vectmp,perm,vec0,PETSC_NULL_IS,scatt,ierr);CHKERRA(ierr)
  call VecScatterBegin(scatt,vectmp,vec0,INSERT_VALUES,SCATTER_FORWARD,ierr);CHKERRA(ierr)
  call VecScatterEnd  (scatt,vectmp,vec0,INSERT_VALUES,SCATTER_FORWARD,ierr);CHKERRA(ierr)
  call VecDestroy(vectmp,ierr);CHKERRA(ierr)
  call VecScatterDestroy(scatt,ierr);CHKERRA(ierr)

  call MatDuplicate(HH,MAT_COPY_VALUES,miH,ierr);CHKERRA(ierr)
  call MatScale(miH,-IU,ierr);CHKERRA(ierr)

  ! warmup CUDA (compile CUDA kernels?)
  call MatMult(HH, vec0,Hvec0,ierr);CHKERRA(ierr)
  call MatMult(HZ, vec0,zvec0,ierr);CHKERRA(ierr)
  call MatMult(miH,vec0,Hvec0,ierr);CHKERRA(ierr)
  ! WARNING: PETSc conjugates the SECOND vector argument, i.e., to get <x|y>, we have to call VecDot(y,x)
  ! https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecDot.html
  call VecDot(Hvec0,vec0,E0,ierr);CHKERRA(ierr)
  call VecDot(zvec0,vec0,z0,ierr);CHKERRA(ierr)

  tt = mpi_wtime()
  call MatMult(HH,vec0,Hvec0,ierr);CHKERRA(ierr)
  call MatMult(HZ,vec0,zvec0,ierr);CHKERRA(ierr)
  call VecDot(Hvec0,vec0,E0,ierr);CHKERRA(ierr)
  call VecDot(zvec0,vec0,z0,ierr);CHKERRA(ierr)
  call MatMult(miH,vec0,Hvec0,ierr);CHKERRA(ierr)
  call VecDot(Hvec0,vec0,miE0,ierr);CHKERRA(ierr)
  
  if (my_id==0) then
    write(6,*) 'time for 3 MatMults:', mpi_wtime() - tt
    write(6,*) 'E0 =', E0
    write(6,*) '<-iH>/-i =', IU * miE0
    write(6,*) '<z> =', z0
  endif

  call TSCreate(PETSC_COMM_WORLD,ts,ierr);CHKERRA(ierr)
  call TSSetSolution(ts,zvec0,ierr);CHKERRA(ierr)
  call TSSetProblemType(ts,TS_LINEAR,ierr);CHKERRA(ierr)
  call TSSetRHSFunction(ts,PETSC_NULL_VEC,TSComputeRHSFunctionLinear,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  call TSSetRHSJacobian(ts,miH,miH,TSComputeRHSJacobianConstant,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  !call TSSetIFunction(ts,PETSC_NULL_VEC,TSComputeIFunctionLinear,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  !call TSSetIJacobian(ts,miH,miH,TSComputeIJacobianConstant,PETSC_NULL_FUNCTION,ierr);CHKERRA(ierr)
  call TSSetFromOptions(ts,ierr);CHKERRA(ierr)

  call TSSolve(ts,Hvec0,ierr);CHKERRA(ierr)

  call PetscFinalize(ierr);CHKERRA(ierr)
END PROGRAM main
