/* Program usage: mpiexec -n 1 toy[-help] [all TAO options] */

/* ----------------------------------------------------------------------
min f=(x1-x2)^2 + (x2-2)^2 -2*x1-2*x2
s.t.         x1^2 + x2 = 2
            0 <= x1^2 - x2 <= 1
            -1 <= x1,x2 <= 2
---------------------------------------------------------------------- */

#include<petsctao.h>
#include<math.h>
#include"function.h"
#include"gradfuncx.h"
#include"gradfuncy.h"
#include"hessfuncxx.h"
#include"hessfuncxy.h"
#include"hessfuncyy.h"

static    char help[]="";

/*
     User-defined application context - contains data needed by the
     application-provided call-back routines, FormFunction(),
     FormGradient(), and FormHessian().
*/

/*
     x,d in R^n
     f in R
     bin in R^mi
     beq in R^me
     Aeq in R^(me x n)
     Ain in R^(mi x n)
     H in R^(n x n)
     min f=(1/2)*x'*H*x + d'*x
     s.t.    Aeq*x == beq
                 Ain*x >= bin
*/
typedef struct {
    PetscInt n; /* Length x */
    PetscInt ne; /* number of equality constraints */
    PetscInt ni; /* number of inequality constraints */
    Vec            x,xl,xu;
    Vec            ce,ci,bl,bu;
    Mat            Ae,Ai,H;

    PetscReal M;
    PetscReal m;
    PetscReal mu;
    PetscReal r;

    PetscInt count;
    PetscInt current;
    PetscReal* x_0;
    PetscReal* y_0;
} AppCtx;

/* -------- User-defined Routines --------- */

PetscErrorCode InitializeProblem(AppCtx *);
PetscErrorCode DestroyProblem(AppCtx *);
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal *,Vec,void *);
PetscErrorCode FormHessian(Tao,Vec,Mat,Mat, void*);
PetscErrorCode FormInequalityConstraints(Tao,Vec,Vec,void*);
PetscErrorCode FormEqualityConstraints(Tao,Vec,Vec,void*);
PetscErrorCode FormInequalityJacobian(Tao,Vec,Mat,Mat, void*);
PetscErrorCode FormEqualityJacobian(Tao,Vec,Mat,Mat, void*);



PetscErrorCode main(int argc,char **argv)
{
    PetscErrorCode         ierr;    /* used to check for functions returning nonzeros */
    Tao                    tao;
    //KSP                    ksp;
    //PC                     pc;
    AppCtx                 user;    /* application context */
    
    ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);

    user.M = 10;
    PetscOptionsGetReal(NULL, NULL, "-center_mass", &user.M, NULL);

    user.m = 1;
    PetscOptionsGetReal(NULL, NULL, "-outer_mass", &user.m, NULL);

    user.mu = 0.1;
    PetscOptionsGetReal(NULL, NULL, "-probe_mass", &user.mu, NULL);

    user.r = 0.2;
    PetscOptionsGetReal(NULL, NULL, "-search_radius", &user.r, NULL);

    PetscReal xs[5];
    PetscReal ys[5];
    
    user.count = 5;
    xs[0] =  0.7; ys[0] =  0;
    xs[1] =  1.3; ys[1] =  0;
    xs[2] = -1;   ys[2] =  0;
    xs[3] =  0.2; ys[3] =  1;
    xs[4] =  0.2; ys[4] = -1;

    {
        PetscReal vals[10];
        PetscInt count = 10;
        PetscBool opt = PETSC_FALSE;
        PetscOptionsGetRealArray(NULL, NULL, "-starting_points", vals, &count, &opt);
        
        if (opt)
        {
            for (int i = 0; i < count; ++i)
                if (i % 2)
                    ys[i >> 1] = vals[i];
                else
                    xs[i >> 1] = vals[i];

            user.count = count >> 1;
        }
    }

    //ierr = PetscPrintf(PETSC_COMM_WORLD,"\n---- TOY Problem -----\n");CHKERRQ(ierr);
    //ierr = PetscPrintf(PETSC_COMM_WORLD,"Solution should be f(1,1)=-2\n");CHKERRQ(ierr);

    ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
    
    for (user.current = 0; user.current < user.count; ++user.current)
    {
        user.x_0 = xs + user.current;
        user.y_0 = ys + user.current;

        PetscPrintf(PETSC_COMM_WORLD, "starting at (%f, %f)\n", *user.x_0, *user.y_0);

        ierr = InitializeProblem(&user);CHKERRQ(ierr);

        //ierr = PetscSleep(10);CHKERRQ(ierr);

        
        //ierr = TaoSetType(tao,TAOPDIPM);CHKERRQ(ierr);
        ierr = TaoSetType(tao,TAOIPM);CHKERRQ(ierr);
        //ierr = TaoSetType(tao,TAOLMVM);CHKERRQ(ierr);
        //ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);
        
        ierr = TaoSetInitialVector(tao,user.x);CHKERRQ(ierr);
        ierr = TaoSetVariableBounds(tao,user.xl,user.xu);CHKERRQ(ierr);
        ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void*)&user);CHKERRQ(ierr);
        
        ierr = TaoSetEqualityConstraintsRoutine(tao,user.ce,FormEqualityConstraints,(void*)&user);CHKERRQ(ierr);
        ierr = TaoSetInequalityConstraintsRoutine(tao,user.ci,FormInequalityConstraints,(void*)&user);CHKERRQ(ierr);

        ierr = TaoSetJacobianEqualityRoutine(tao,user.Ae,user.Ae,FormEqualityJacobian,(void*)&user);CHKERRQ(ierr);
        ierr = TaoSetJacobianInequalityRoutine(tao,user.Ai,user.Ai,FormInequalityJacobian,(void*)&user);CHKERRQ(ierr);
        ierr = TaoSetHessianRoutine(tao,user.H,user.H,FormHessian,(void*)&user);CHKERRQ(ierr);
        //ierr = TaoSetTolerances(tao,0,0,0);CHKERRQ(ierr);
        
        ierr = TaoSetMaximumIterations(tao, 10000);CHKERRQ(ierr);
        ierr = TaoSetMaximumFunctionEvaluations(tao, 10000);CHKERRQ(ierr);
        
        {
            KSP ksp;
            PC pc;

            ierr = TaoGetKSP(tao, &ksp);CHKERRQ(ierr);
            ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
            ierr = PCSetType(pc, PCSVD);CHKERRQ(ierr);
        }

        ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
            /*
        ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
        ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
        ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
        
                //This algorithm produces matrices with zeros along the diagonal therefore we need to use
            //SuperLU which does partial pivoting

        ierr = PCFactorSetMatSolverType(pc,MATSOLVERSUPERLU);CHKERRQ(ierr);
        ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
        ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

        //ierr = TaoSetTolerances(tao,0,0,0);CHKERRQ(ierr); */
        ierr = TaoSolve(tao);CHKERRQ(ierr);
        
        const PetscScalar *x;

        ierr = VecGetArrayRead(user.x,&x);CHKERRQ(ierr);
        
        double tempvar=function(user.M, user.m, user.mu, x[0], x[1]);
        
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Min at x=%.16f\ty=%.16f\nValue at minimum %.16f\n",x[0], x[1], tempvar);CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(user.x, &x);
            
        ierr = DestroyProblem(&user);CHKERRQ(ierr);
    }

    ierr = TaoDestroy(&tao);CHKERRQ(ierr);
    ierr = PetscFinalize();
    return ierr;
}

PetscErrorCode InitializeProblem(AppCtx *user)//Initialisierung für Speicher
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    user->n = 2;    //Anzahl freier Variablen
    ierr = VecCreateSeq(PETSC_COMM_SELF,user->n,&user->x);CHKERRQ(ierr);
    ierr = VecDuplicate(user->x,&user->xl);CHKERRQ(ierr);
    ierr = VecDuplicate(user->x,&user->xu);CHKERRQ(ierr);
    //ierr = VecSet(user->x,0.0);CHKERRQ(ierr);                   // <----------
    
    ierr = VecSetValue(user->x, 0, *user->x_0, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(user->x, 1, *user->y_0, INSERT_VALUES);CHKERRQ(ierr);
    
    ierr = VecAssemblyBegin(user->x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(user->x);CHKERRQ(ierr);
    
    ierr = VecSet(user->xl,-5.0);CHKERRQ(ierr);
    ierr = VecSet(user->xu,5.0);CHKERRQ(ierr);
    
    user->ne = 1;   //Anzahl der Nebenbedingungen mit Gleichung
    ierr = VecCreateSeq(PETSC_COMM_SELF,user->ne,&user->ce);CHKERRQ(ierr);

    user->ni = 1;   //Anzahl der Nebenbedingungen mit Ungleichung
    ierr = VecCreateSeq(PETSC_COMM_SELF,user->ni,&user->ci);CHKERRQ(ierr);
    
    ierr = MatCreateSeqDense(PETSC_COMM_WORLD, user->ne, user->n, NULL, &user->Ae);CHKERRQ(ierr);
    //ierr = MatSetSizes(user->Ae, PETSC_DECIDE, PETSC_DECIDE, user->ne, user->n);CHKERRQ(ierr);
    
    ierr = MatCreateSeqDense(PETSC_COMM_WORLD, user->ni, user->n, NULL, &user->Ai);CHKERRQ(ierr);
    //ierr = MatSetSizes(user->Ai, PETSC_DECIDE, PETSC_DECIDE, user->ni, user->n);CHKERRQ(ierr);
    
    ierr = MatSetFromOptions(user->Ae);CHKERRQ(ierr);
    ierr = MatSetFromOptions(user->Ai);CHKERRQ(ierr);
    
    ierr = MatSetUp(user->Ae);CHKERRQ(ierr);
    ierr = MatSetUp(user->Ai);CHKERRQ(ierr);
    
    ierr = MatCreateSeqDense(PETSC_COMM_WORLD, user->n, user->n, NULL, &user->H);
    //ierr = MatSetSizes(user->H, PETSC_DECIDE, PETSC_DECIDE, user->n, user->n);
    
    ierr = MatSetFromOptions(user->H);CHKERRQ(ierr);CHKERRQ(ierr);

    ierr = MatSetUp(user->H);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

PetscErrorCode DestroyProblem(AppCtx *user)//Freiheit für Speicher!!!
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = MatDestroy(&user->Ae);CHKERRQ(ierr);
    ierr = MatDestroy(&user->Ai);CHKERRQ(ierr);
    ierr = MatDestroy(&user->H);CHKERRQ(ierr);

    ierr = VecDestroy(&user->x);CHKERRQ(ierr);
    ierr = VecDestroy(&user->ce);CHKERRQ(ierr);
    ierr = VecDestroy(&user->ci);CHKERRQ(ierr);
    ierr = VecDestroy(&user->xl);CHKERRQ(ierr);
    ierr = VecDestroy(&user->xu);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
    AppCtx *user=(AppCtx*)ctx;

    PetscScalar *g;
    const PetscScalar *x;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
        //VecGetArray[Read]() lockt Zugriff aus speicher
    ierr = VecGetArray(G,&g);CHKERRQ(ierr);
    
    *f=function(user->M, user->m, user->mu, x[0], x[1]);
        //Funktion, von der das Minimum bestimmt werden soll
    
    g[0]=gradfuncx(user->M, user->m, user->mu, x[0], x[1]);    //x-Komponente des Gradienten
    g[1]=gradfuncy(user->M, user->m, user->mu, x[0], x[1]);    //y-Komponente des Gradienten
    
    //PetscPrintf(PETSC_COMM_WORLD,"x = %.16f\ty = %.16f\n", x[0],x[1]);
    //PetscPrintf(PETSC_COMM_WORLD,"x-Grad = %.16f\ty-Grad = %.16f\n", g[0], g[1]);
    
    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
        //VecRestoreArray[Read]() gibt Zugriff wieder frei
    ierr = VecRestoreArray(G,&g);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode FormHessian(Tao tao, Vec X, Mat H, Mat Hpre, void *ctx)
{
    AppCtx *user=(AppCtx*)ctx;

    //Vec DE, DI;
    //const PetscScalar *de, *di;
    //PetscInt zero=0,one=1;
    //PetscScalar two=2.0;
    PetscScalar val[4];
    PetscInt barry[2];
    
    const PetscScalar *x;
    VecGetArrayRead(X,&x);
    
    barry[0]=0;
    barry[1]=1;
    
    PetscErrorCode ierr;

    PetscFunctionBegin;
    //ierr = TaoGetDualVariables(tao,&DE,&DI);CHKERRQ(ierr);

    //ierr = VecGetArrayRead(DE,&de);CHKERRQ(ierr);
    //ierr = VecGetArrayRead(DI,&di);CHKERRQ(ierr);
    
    //di[0]*d/dx^2 Nebenbedingung
    //di[1]*d/dy^2 Nebenbedingung
    
    val[0]=hessfuncxx(user->M, user->m, user->mu, x[0], x[1]);//+2*di[0];        //H_11
    val[1]=hessfuncxy(user->M, user->m, user->mu, x[0], x[1]);                //H_12
    val[2]=hessfuncxy(user->M, user->m, user->mu, x[0], x[1]);                //H_21
    val[3]=hessfuncyy(user->M, user->m, user->mu, x[0], x[1]);//+2*di[0];        //H_22
    
    //PetscPrintf(PETSC_COMM_WORLD,"H = %.16f\t%.16f\n    %.16f\t%.16f\n", val[0], val[1], val[2], val[3]);
    
    //val=2.0 * (1 + de[0] + di[0] - di[1]);
    //ierr = VecRestoreArrayRead(DE,&de);CHKERRQ(ierr);
    //ierr = VecRestoreArrayRead(DI,&di);CHKERRQ(ierr);
    
    ierr = MatSetValues(H,2,barry,2,barry,val,INSERT_VALUES);CHKERRQ(ierr);
    
    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    // NOTE: added
    //ierr = MatCopy(H, Hpre, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

    //ierr = MatGetValues(H,2,barry,2,barry,val);CHKERRQ(ierr);
    //PetscPrintf(PETSC_COMM_WORLD,"H = %.16f\t%.16f\n    %.16f\t%.16f\n", val[0], val[1], val[2], val[3]);
    
    PetscFunctionReturn(0);
}

PetscErrorCode FormInequalityConstraints(Tao tao, Vec X, Vec CI, void *ctx)
{
    AppCtx *user=(AppCtx*)ctx;

    const PetscScalar *x;
    PetscScalar *c;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
    ierr = VecGetArray(CI,&c);CHKERRQ(ierr);
    
    c[0] = -(x[0]-(PetscScalar)*user->x_0)*(x[0]-(PetscScalar)*user->x_0)-(x[1]-(PetscScalar)*user->y_0)*(x[1]-(PetscScalar)*user->y_0)+user->r*user->r; //>=0
    
    //PetscPrintf(PETSC_COMM_WORLD,"c = %.16f\tx = %.16f\ty = %.16f\n\n", c[0], x[0], x[1]);
    
    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
    ierr = VecRestoreArray(CI,&c);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
        
PetscErrorCode FormEqualityConstraints(Tao tao, Vec X, Vec CE,void *ctx)
{
    PetscScalar *x,*c;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = VecGetArray(X,&x);CHKERRQ(ierr);
    ierr = VecGetArray(CE,&c);CHKERRQ(ierr);
    c[0] = 0;
    ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    ierr = VecRestoreArray(CE,&c);CHKERRQ(ierr);
    PetscFunctionReturn(0);
    
}

PetscErrorCode FormInequalityJacobian(Tao tao, Vec X, Mat JI, Mat JIpre, void *ctx)
{   
    AppCtx *user=(AppCtx*)ctx;
    
    PetscInt rows[user->ni];    //ni Nebenbedingunge
    PetscInt cols[user->n];     //n freie Variablen x, y...
    PetscScalar vals[user->ni*user->n];
    const PetscScalar *x;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
    rows[0] = 0;            //rows[1] = 1;
    cols[0] = 0;            cols[1] = 1;
    vals[0] = -2*(x[0]-*user->x_0);
    vals[1] = -2*(x[1]-*user->y_0);     //erste ableitungen der Nebenbedingungen
    
    //PetscPrintf(PETSC_COMM_WORLD,"val[0] = %.16f\tval[1] = %.16f\tx = %.16f\ty = %.16f\t ni = %d n = %d\n\n", vals[0], vals[1], x[0], x[1], user->ni, user->n);
    //vals[2] = -2*x[0];  vals[3] = +1.0;
    
    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
    
    ierr = MatSetValues(JI, 1,rows,2,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    
    ierr = MatAssemblyBegin(JI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(JI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode FormEqualityJacobian(Tao tao, Vec X, Mat JE, Mat JEpre, void *ctx)
{   
    AppCtx *user=(AppCtx*)ctx;
    
    //user->variable = (*user).variable
    //&user =adresse von user-Pointer = **user
    //*(&user)=user???
    
    
    PetscInt rows[user->ne];        //für ne Nebenbedingungen
    
    PetscScalar vals[user->ni*user->n];      //für n freie Variablen x, y...
    const PetscScalar *x;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
    rows[0] = 0;        rows[1] = 1;
    vals[0] = 0;        vals[1] = 0;
    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
    ierr = MatSetValues(JE,1,rows,2,rows,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(JE,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(JE,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


/*TEST

     build:
            requires: !complex !define(PETSC_USE_CXX)

     test:
            requires: superlu
            args: -tao_smonitor -tao_view -tao_gatol 1.e-5

TEST*/


//  ./toy -tao_view -tao_test_hessian

//  /home/andre/compori2122/petscbuild/petsc-3.15.4/src/tao/interface
