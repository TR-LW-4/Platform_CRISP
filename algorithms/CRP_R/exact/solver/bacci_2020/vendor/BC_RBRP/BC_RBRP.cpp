/*
Copyright 2020 Tiziano Bacci, Sara Mattia, Paolo Ventura
This file is part of BC-RBRP.

    BC-RBRP is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    BC-RBRP is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with BC-RBRP.  If not, see <http://www.gnu.org/licenses/>.
*/
/* The present file include the main function for calling the BC-RBRP or
   the Bounded Beam Search heuristic.
   Usage "Usage: ./BC_RBRP.exe instanceName <solS=> <optS=> <verbS=>\n
   <solS=> , if 0 uses BC-BRP, if 1 uses Bounded Beam Search
   <optS=> , if 1 solves with BC-BRP the integral problem , if 0 solves with BC-BRP the continuous (relaxed) problem
   <verbS=> , if 1 displays solution of BC-BRP, 0 otherwise
*/
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <sys/time.h>
#include <iostream>
#include<string.h>
using namespace std;
#define Alloc(p,a,t) do						\
    if (!(p = (t *) malloc ((size_t) ((a) * sizeof (t))))) {	\
      printf("run out of memory [Alloc(%s,%d,%s)]\n",#p,a,#t);	\
      exit(1);							\
    } while (0)



int readCommandLine (int argc, char **argv, char *filename, int *solverSwitch, int *optimalitySwitch, int *verbose);
int readInstance (char *fileName, int *n, int *w, int *h, int ***A);
double rBRP_MIP3 (int n, int w, int h, int **D, int ub, double *optCpuTime, int optimalitySwitch, int verbose, double *slb,int *svar,int *scon,double *snode,int *scuts,double *stimecall,int *sstatus,int ***solution);
int rBRP_BSheu (int n, int w, int h, int **D, int ***solution);
void printSolution(int n, int w, int h, int ***solution);

int main (int argc, char **argv) {
  int n, w, h, **A;//n = number of blocks, w = number of stacks, h = height of each stack, A = yard (w x h)
  char fileName[1000];
  int solverSwitch, optimalitySwitch, verbose;
  
  double start, end;
  timeval tim;
  //OUTPUT
  double optCpuTime, brpVal;//computing time, optimal cost
  int ub;//initial upper bound provided by Bounded Beam Search algorithm
  double lb;//final best bound
  int var;//number of variables of the model
  int con;//number of constraints of the model
  double node;//numer of nodes explored during the branch-and-cut
  int cuts;//number of cuts added
  double timecall;//time spent in adding cuts
  int status;//status of the solver Gurobi
  int ***solution;//solution of BBS/BC-RBRP algorithm: solution[t][j][k]=i if in stack j^th, slot k^th is located item i at time period t, -1 if the slot is empty
    
  
  //read input parameters
  readCommandLine (argc, argv, fileName, &solverSwitch, &optimalitySwitch, &verbose);
  //read instance
  readInstance (fileName, &n, &w, &h, &A);

  Alloc(solution, n+1, int**);
  for(int i = 0; i < n+1; i++){
    Alloc(solution[i], w, int*);
    for(int j = 0; j < w; j++)
      Alloc(solution[i][j], h, int);
  }

  switch (solverSwitch) {
  case 0://solve exactly by using BC-RBRP
    gettimeofday(&tim, NULL);
    start=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));
    ub = (double)rBRP_BSheu (n, w, h, A, solution);    
    printf ("Heuristic solution value = %d\n", ub);
    brpVal = rBRP_MIP3 (n, w, h, A, ub, &optCpuTime, optimalitySwitch, verbose, &lb,&var,&con,&node,&cuts,&timecall,&status,solution);
    printf ("Optimal solution = %.0lf, found in %2.2f secs.\n", brpVal, optCpuTime);
    if(verbose){
      printf("\nBC-RBRP solution\n\n");
      printSolution(n, w, h, solution);
      printf("\n");
    }
    gettimeofday(&tim, NULL);
    end=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));
    break; 
  case 1://solve by using heuristic BBS
    gettimeofday(&tim, NULL);
    start=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));
    ub = (double)rBRP_BSheu (n, w, h, A, solution);    
    gettimeofday(&tim, NULL);
    end=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));
    printf ("Huristic solution = %d, found in %2.2f secs.\n", ub, end-start);
    if(verbose){
      printf("\nBBS solution\n\n");
      printSolution(n, w, h, solution);
      printf("\n");
    }
  default: ;
  }
  for (int i = 0; i < w ; i++) 
    free (A[i]);
  free (A);
 
  for(int i = 0; i < n+1; i++){
    for(int j = 0; j < w; j++)
      free(solution[i][j]);
    free(solution[i]);
  }
  free(solution);
  
  
  return (0);
}


int readCommandLine (int argc, char **argv, char *filename, int *solverSwitch, int *optimalitySwitch, int *verbose) {
  int i, check;
  if (argc < 2) {
    printf ("Usage: ./BC_RBRP.exe fileName <solS=> <optS=> <verbS=>\n");
    printf ("where:\n");
    printf ("solS (solver switch)          = 0 BC-RBRP, exact method for the restricted Block Relocation Problem\n");
    printf ("                              = 1 BBS,  heuristic method for the restricted Block Relocation Problem\n");
    printf ("optS (optimality switch)      = 1 BC-RBRP solves instance to optimality with integers variables\n");
    printf ("                              = 0 BC-RBRP solves the linear relaxation with continuous variables\n");
    printf ("verbS (verbosity switch)      = 1 BC-RBRP output solution\n");
    printf ("                              = 0 BC-RBRP no output solution\n");
    printf ("                              Default: 0\n\n");
    exit(0);
  }

  *solverSwitch = 0;
  *optimalitySwitch = 1;
  *verbose = 0;
  
  sscanf (argv[1], "%s", filename);
  for (i = 2; i < argc; ++i) {
    check = 0;
    if (!strncmp (argv[i], "solS", 4)) {
      sscanf (argv[i], "solS=%d", solverSwitch);
      check ++;
    }
    if (!strncmp (argv[i], "optS", 4)) {
      sscanf (argv[i], "optS=%d", optimalitySwitch);
      check ++;
    }
    if (!strncmp (argv[i], "verbS", 4)) {
      sscanf (argv[i], "verbS=%d", verbose);
      check ++;
    }
    if (!check) {
      printf ("Wrong input parameter: %s\n", argv[i]);
      exit(0);
    }
  }


  return (0);
}
// read input instance according to the Bacci's website format
int readInstance (char *fileName, int *n, int *w, int *h, int ***A) {
  FILE *iF;
  int i, j, k;
  iF = fopen (fileName, "r");
  
  fscanf (iF, "%d %d %d\n", w, h, n);
  *A = (int**) malloc ((*w) * sizeof (int*));
  for (i = 0; i < (*w); ++i) {
    (*A)[i] = (int*) malloc ((*h) * sizeof (int));
    fscanf (iF, "%d", &k);
    for (j = 0; j < k; ++j)
      fscanf (iF, "%d", &(*A)[i][j]);
    for (j = k; j < *h; ++j)
	(*A)[i][j] = -1;
  }

  fflush(iF);
  fclose(iF);
  return (0);
}

//Print final solution of the BBS/BC-RBRP algorithm
void printSolution(int n, int w, int h, int ***solution){
  int nn = n;
  int digits = 0;
  int *resh;
  Alloc(resh, h, int);
  int count = 0, tcount = 0;
  while (nn != 0) {
    nn /= 10; 
    ++digits;
  }
  int tdigits;
  int tnn;
  for(int t = 0; t < n+1; t++){
    printf("**** Time period = %d\n\n",t+1);
    for(int k = h-1; k >= 0; k--){
      for(int j = 0; j < w; j++){
	if(solution[t][j][k] > 0){

	  if(solution[t][j][k] == t+1){
	    tcount = 0;
	    for(int s = k+1; s < h; s++)
	      if(solution[t][j][s] > 0){
		resh[tcount] = solution[t][j][s];
		tcount++;
		count++;
	      }
	  }
	  
	  tnn = solution[t][j][k];
	  tdigits = 0;
	  while (tnn != 0) {
	    tnn /= 10; 
	    ++tdigits;
	  }
	}else{
	  tnn = 0;
	  tdigits = 0;
	}
	printf("[");
	for(int i = 0; i < (digits-tdigits);i++)
	  printf(" ");
	if(solution[t][j][k] > 0)
	  printf("%d",solution[t][j][k]);
	//else
	//printf(" ");
	printf("]");

      }
      printf("\n");
    }
    printf("\n");
    printf("Reshuffles = %d\n",count);
    if(tcount){
      printf("\nReshuffled blocks at time period %d =",t+1);
      for(int s = 0; s < tcount; s++)
	printf(" %d",resh[s]); 
    }
    printf("\n\n");
  }
  free(resh);
}
