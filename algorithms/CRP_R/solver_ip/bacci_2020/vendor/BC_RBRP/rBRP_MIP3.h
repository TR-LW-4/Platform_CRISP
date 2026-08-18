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
/* The present file containes the function (callback for Gurobi solver) for dinamically generating rows in the ILP model of the BC-RBRP, as described in Bacci et al. (2020).
 */
#include "gurobi_c++.h"
#include <sys/time.h>


#define Alloc(p,a,t) do						\
    if (!(p = (t *) malloc ((size_t) ((a) * sizeof (t))))) {	\
      printf("run out of memory [Alloc(%s,%d,%s)]\n",#p,a,#t);	\
      exit(1);							\
    } while (0)


typedef struct {
  int w;      // stack where the current block is located
  int h;      // height where the current block is located
  int lb;     // height of the highest blocking item
  int **yard; // block[i].yard is the yard remaining when items 1,...,i-1 have been retrieved
  int maxMin; // max of the minimum on the stacks (excluded block[i].w) of block[i].yard
  int *minS; //  minS[j] = minimum item in stack j (n+1 if j is empty)
  int pi;    // it is the first time when the current block is moved (reshuffled or retrieved)
  int zub;    // upper bound on the number of reshuffles due to the retrieval of the current block
  int *pos;   // pos[i] = stack where item i is located at current time
  int *resh;  // resh[i] = 1 iff item i is reshuffled at current time 
  int *from;  // resh[i] = 1 iff item i is reshuffled at current time 
} T_block;



int constructDataStructure (T_block *block, int n, int w, int h, int **yard, int UBT);
int addVariables (int n, int w, int h, GRBModel *model, GRBVar ***varx, GRBVar ***vary, int intProbSwitch, T_block *block);
int addObjectiveFunction (int n, int w, GRBModel *model, GRBVar ***vary);
int addConstraints (GRBEnv env, GRBModel *model, GRBVar ***varx, GRBVar ***vary, int n, int w, int h, T_block *block, int verbose);
int setMipParameters (GRBModel *model);
int setLazyConstraints (GRBModel *model, T_block *block, int n, int w, int h);
int copyInitialSolution (GRBModel *model, GRBVar ***varx, GRBVar ***vary, int n, int w, int h, int ***solution);
int checkSol (int n, int w, int h, int **Yard, GRBModel model, GRBVar ***varx, GRBVar **varz, int verbose);
int saveSolution(int n, int w, int h, int **Yard, T_block *block, GRBModel model, GRBVar ***varx, GRBVar ***vary, int ***solution);
int findStackMin (int n, int card, int *v);
int adjbin (int n, int* v, int *w, int val);
int min (int a, int b);

class mycallback: public GRBCallback
{
  public:
  int n, w, h;
    GRBVar ***varx;
    GRBVar ***vary; 
    GRBModel *model;
    T_block *block;
    int *nlazy;
    double *timelazy;
    mycallback(int ns, int ws, int hs, GRBVar ***varxs, GRBVar ***varys, T_block *blocks, GRBModel *models, int *nlazys, double *timelazys) {
      varx = varxs;
      vary = varys;
      n = ns;
      w = ws;
      h = hs;
      block = blocks;
      model = models;
      nlazy = nlazys;
      timelazy = timelazys;

    }
  protected:
    void callback () {
      if(where == 4){
	try {
	  int cutsNo = 0;
	  int i, i1, i3, j, t, k, k1;
	  timeval tim;
	  gettimeofday(&tim, NULL);
	  double start=tim.tv_sec+(tim.tv_usec/1000000.0);


	  //SAVE CURRENT INTEGER SOLUTION
	  for(t=1; t < n; t++){//For each time slot t
	    for(i=t; i<n; i++){
	      if(block[i].pi < t) {
		if(getSolution(varx[i][block[t-1].pos[i]][t]) > 0.5)
		  block[t].pos[i] = block[t-1].pos[i];
		else{
		  for(j=0; j<w; j++)
		    if(j!=block[t-1].pos[i])
		      if(getSolution(varx[i][j][t]) > 0.5)
			block[t].pos[i] = j;
		}
		block[t].resh[i] = 0;
		block[t].from[i] = -1;
		if(i>t)
		  for(j=0; j<w; j++)
		    if(getSolution(vary[i][j][t]) > 0.5){
		      block[t].resh[i] = 1;
		      block[t].from[i] = j;
		    }
	      }
	    }
	    
	    if(t>0){
	      int blockt, t1, i2;
	     
	      cutsNo = 0;
	      //Check feasibility of all surrogate constraints (18)	    
	      for(i1=t; i1 < n; i1++)
		if(block[i1].pi<t)
		  for(i2 = t; i2 < i1; i2++)
		    if(block[i2].pi<t)
		      if(block[t].pos[i1] == block[t].pos[i2])
			if(block[t].minS[block[t].pos[i2]] > i2)
			  if(block[t-1].pos[i2] == block[t].pos[i2])
			    if(block[t-1].pos[i1] != block[t].pos[i1])
			      {
		      
				double sumy = 0.0;		      
				for(i3 = t; i3 <= i2; i3++)
				  if(block[i3].pi<t)
				    sumy += getSolution(vary[i1][block[t-1].pos[i2]][i3]);
				if(sumy < 0.5){
				  GRBLinExpr xpr = 0;
				  xpr += varx[i2][block[t-1].pos[i2]][t-1];
				  xpr -= varx[i1][block[t-1].pos[i2]][t-1];
				  xpr += varx[i1][block[t-1].pos[i2]][t];
				  for(i3 = t; i3 <= i2; i3++)
				    if(block[i3].pi<t)
				      xpr -= vary[i1][block[t-1].pos[i2]][i3];
				  addLazy(xpr <= 1);
				  i2 = i1+1;
				  (*nlazy)++;
				  cutsNo++;
				}
			      }
	      
	      if(cutsNo==0)//if no surrogate constraints (18) are added
		for(i1=t; i1 < n; i1++)// for each block i > t, check whether i is above t or not at time t	
		  if(block[i1].pi<t){
		    if(i1>t)
		      if(block[t].pi <t)
			if(block[t].pos[i1]==block[t].pos[t]){
			  t1 = t;
			  int changestack = 0;
			  int startstack = block[t].pos[t];
			  while(t1>0 && (block[t1].pos[i1]==block[t1].pos[t])){
			    if(block[t1].pos[t]!=startstack){
			      startstack = block[t1].pos[t];
			      changestack++;
			    }
			    t1--;
			  }
			  if(t1>0){
			    if(block[t1].pos[t]==block[t1+1].pos[t])
			      blockt = 1;
			    else
			      blockt = 0;
			    if((changestack%2)>0){
			      if(blockt==1)
				blockt = 0;
			      else
				blockt = 1;
			    }
			  }else{
			    blockt=0;
			    if(block[0].pos[i1]==block[0].pos[t]){
			      for(k=0; k<h-1;k++)
				if(block[0].yard[block[0].pos[t]][k] == t+1)
				  for(k1=k+1; k1<h;k1++)
				    if(block[0].yard[block[0].pos[t]][k1] == i1+1)
				      blockt = 1;
			    }else{
			      if(block[0].pos[t]==block[1].pos[t])
				blockt=1;
			    }
			    if(block[0].pos[t]!=block[1].pos[t] && block[0].pos[i1]==block[0].pos[t])
			      changestack++;
		
			    if((changestack%2)>0){
			      if(blockt==1)
				blockt = 0;
			      else
				blockt = 1;
			    }
			  }
			  int tbar;
			  if(t1 > 0)
			    tbar = t1+1;
			  else{
			    if(block[t1].pos[i1]==block[t1].pos[t])
			      tbar = t1;
			    else
			      tbar = t1+1;
			  }
				
			  if(blockt == 0 && block[t].resh[i1] == 1){//if i does not block t a time t but i is reshuffled
			    //Add S t-up constraints (9)
			    GRBLinExpr xpr = 0;

			    if(tbar > 0){
			      if(block[tbar].pos[i1]!=block[tbar-1].pos[i1]){
				xpr -= (1 - varx[i1][block[tbar].pos[t]][tbar]);
				xpr -= varx[i1][block[tbar].pos[t]][tbar-1];
				xpr -= (1 - varx[t][block[tbar].pos[t]][tbar-1]);
			      }
			      if(block[tbar].pos[t]!=block[tbar-1].pos[t]){
				xpr -= (1 - varx[i1][block[tbar].pos[i1]][tbar-1]);
				xpr -= varx[t][block[tbar].pos[i1]][tbar-1];
				xpr -= (1 - varx[t][block[tbar].pos[i1]][tbar]);
			      }
			    }
	     
			    for(t1=tbar;t1<t;t1++){
			      if(block[t1].resh[i1] == 0){
				for(j = 0; j < w; j++){
				xpr -= vary[i1][j][t1];
				xpr -= vary[t][j][t1];
				}
			      }else{
				xpr -= 1;
				for(j = 0; j < w; j++)
				  xpr += vary[i1][j][t1];
				xpr -= 1;
				for(j = 0; j < w; j++)
				xpr += vary[t][j][t1];
			      }
			    }
			    xpr -= (1 - varx[i1][block[t].pos[i1]][t]);
			    xpr -= (1 - varx[t][block[t].pos[t]][t]);
			    for(j = 0; j < w; j++)
			      xpr += vary[i1][j][t];
			    addLazy(xpr <= 0);
			    cutsNo++;
                            (*nlazy)++;
			  }
			  if(blockt == 1 && block[t].resh[i1] == 0){//if i blocks t a time t but i is not reshuffled
			    //Add S i-up constraints (8)
			    GRBLinExpr xpr = 0;			    
			    if(tbar > 0){
		     
			      if(block[tbar].pos[i1]!=block[tbar-1].pos[i1]){
				xpr -= (1 - varx[i1][block[tbar].pos[t]][tbar]);
				xpr -= varx[i1][block[tbar].pos[t]][tbar-1];
				xpr -= (1 - varx[t][block[tbar].pos[t]][tbar-1]);
			      }
			      if(block[tbar].pos[t]!=block[tbar-1].pos[t]){
				xpr -= (1 - varx[i1][block[tbar].pos[i1]][tbar-1]);
				xpr -= varx[t][block[tbar].pos[i1]][tbar-1];
				xpr -= (1 - varx[t][block[tbar].pos[i1]][tbar]);
			      }
			    }
	     
			    for(t1=tbar;t1<t;t1++){
			      if(block[t1].resh[i1] == 0){
				for(j = 0; j < w; j++){
				xpr -= vary[i1][j][t1];
				xpr -= vary[t][j][t1];
			      }
			      }else{
				xpr -= 1;
				for(j = 0; j < w; j++)
				  xpr += vary[i1][j][t1];
				xpr -= 1;
				for(j = 0; j < w; j++)
				  xpr += vary[t][j][t1];
			      }
			    }
		    
			    xpr -= (1 - varx[i1][block[t].pos[i1]][t]);
			    xpr -= (1 - varx[t][block[t].pos[t]][t]);
			    xpr -= vary[i1][block[t].pos[t]][t];
			    addLazy(xpr <= -1);
			    cutsNo++;
                            (*nlazy)++;
			  }
			}
	  
		  }
	    }

	    if(cutsNo > 0)
	      t = n+1;
	   
	  }
	  gettimeofday(&tim, NULL);
	  double end=tim.tv_sec+(tim.tv_usec/1000000.0);
	  (*timelazy) = (*timelazy) + (end - start);

       
	} catch (GRBException e) {
	  std::cout << "Error number: " << e.getErrorCode() << std::endl;
	  std::cout << e.getMessage() << std::endl;
	} catch (...) {
	  std::cout << "Error during callback" << std::endl;
	}
      }
    }
};
