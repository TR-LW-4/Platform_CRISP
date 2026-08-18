/*
Copyright 2020 Tiziano Bacci, Sara Mattia Paolo Ventura
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
/* This file contains the code for creating and running the ILP model of the BC-RBRP
   with Gurobi 8.1.1. All the constrains are numbered according to the paper by Bacci et al. (2020).
   One thread and a time limit of one hour are set. 
   The callback procedure for dinamically generating rows in included in rBRP_MIP3.h .
*/ 
#include "rBRP_MIP3.h"
#include <stdlib.h>
double rBRP_MIP3(
		 int n,//number of blocks
		 int w,//number of stakcs
		 int h,//height of each stack
		 int **Yard,//yard (wxh)
		 int ub,
		 double *timeBrp,
		 int optimalitySwitch,
		 int verbose,
		 double *slb,
		 int *svar,
		 int *scon,
		 double *snode,
		 int *scuts,
		 double *stimecall,
		 int *sstatus,
		 int ***solution
		 )
{
  int i, j;
  GRBEnv env = GRBEnv();
  GRBModel model = GRBModel(env);
  GRBVar ***varx;
  GRBVar ***vary;
  double objVal;
  T_block *block;
  varx = new GRBVar**[n];
  for(i = 0; i < n; i++){
    varx[i] = new GRBVar*[w];
    for(j = 0; j < w; j++){
      varx[i][j] = new GRBVar[i+1];
    }
  }

  vary = new GRBVar**[n];
  for(i = 0; i < n; i++){
    vary[i] = new GRBVar*[w];
    for(j = 0; j < w; j++){
      vary[i][j] = new GRBVar[i];
    }
  }

  Alloc (block, n, T_block);
  constructDataStructure (block, n, w, h, Yard, ub);
  
  // CONSTRUCT THE ILP MODEL
  addVariables (n, w, h, &model, varx, vary, optimalitySwitch, block);
  addObjectiveFunction (n, w, &model, vary);
  addConstraints (env, &model, varx, vary, n, w, h, block, verbose);
  
  // SOLVE THE MODEL
  setMipParameters (&model);
  if(optimalitySwitch)
    copyInitialSolution (&model, varx, vary, n, w, h, solution);  
  int nlazy = 0;//total number of lazy cuts added
  double timelazy = 0.0; //total time spent in the lazy callback
  setLazyConstraints (&model, block, n, w, h);  
  mycallback cb = mycallback(n, w, h, varx, vary, block, &model, &nlazy, &timelazy);//callback for adding cuts to integer solutions
  
  model.setCallback(&cb);
  model.update();
  //model.write("brp.lp");//uncomment to write .lp file
  timeval tim;
  gettimeofday(&tim, NULL);
  double time1=tim.tv_sec+(tim.tv_usec/1000000.0);
  model.optimize();
  gettimeofday(&tim, NULL);
  double time2=tim.tv_sec+(tim.tv_usec/1000000.0);
  (*timeBrp) = (time2 - time1);
  int optimstatus = model.get(GRB_IntAttr_Status);
  if (optimstatus == GRB_OPTIMAL) {
    objVal = model.get(GRB_DoubleAttr_ObjVal);
    double bestb = model.get(GRB_DoubleAttr_ObjBound);
    int nvar = model.get(GRB_IntAttr_NumVars);
    int ncon = model.get(GRB_IntAttr_NumConstrs);
    double nodes = model.get(GRB_DoubleAttr_NodeCount);
    if(optimalitySwitch)
      saveSolution (n, w, h, Yard, block, model, varx, vary, solution);
    std::cout << "Optimum " << objVal << " time " << (*timeBrp) <<
      " initial_ub " << ub << " best_bound " << bestb <<
      " num_variables " << nvar << " num_constraints " << ncon <<
      " num_nodes " << nodes << " num_lazy " << nlazy <<
      " time_callback " << timelazy << " status " << optimstatus
      << std::endl;
    *slb = bestb;
    *svar = nvar;
    *scon = ncon;
    *snode = nodes;
    *scuts = nlazy;
    *stimecall = timelazy;
    *sstatus = optimstatus;
  } else if (optimstatus == GRB_INF_OR_UNBD) {
    std::cout << "Model is infeasible or unbounded" << std::endl;
    objVal = ((double)ub);
    exit(0);
  } else if (optimstatus == GRB_INFEASIBLE) {
    std::cout << "Model is infeasible" << std::endl;
    objVal = ((double)ub);
    exit(0);
  } else if (optimstatus == GRB_UNBOUNDED) {
    std::cout << "Model is unbounded" << std::endl;
    objVal = ((double)ub);
    exit(0);
  } else if (optimstatus == GRB_TIME_LIMIT){
    objVal = model.get(GRB_DoubleAttr_ObjVal);
    double bestb = model.get(GRB_DoubleAttr_ObjBound);
    int nvar = model.get(GRB_IntAttr_NumVars);
    int ncon = model.get(GRB_IntAttr_NumConstrs);
    double nodes = model.get(GRB_DoubleAttr_NodeCount);
    (*timeBrp) = 3600.5;
    if(optimalitySwitch)
      saveSolution (n, w, h, Yard, block, model, varx, vary, solution);
    std::cout << "Optimum " << objVal << " time " << (*timeBrp) <<
      " initial_ub " << ub << " best_bound " << bestb <<
      " num_variables " << nvar << " num_constraints " << ncon <<
      " num_nodes " << nodes << " num_lazy " << nlazy <<
      " time_callback " << timelazy << " status " << optimstatus
      << std::endl;
    *slb = bestb;
    *svar = nvar;
    *scon = ncon;
    *snode = nodes;
    *scuts = nlazy;
    *stimecall = timelazy;
    *sstatus = optimstatus;
  } else if (optimstatus == GRB_CUTOFF){
    std::cout << "Model objective exceeds cutoff" << std::endl;
    objVal = ((double)ub);
    exit(0);
  } else {
    std::cout << "Optimization was stopped with status = "
	 << optimstatus << std::endl;
    exit(0);
    return 0;
  }
    

  //free variables x
  for(i = 0; i < n; i++){
    for(j = 0; j < w; j++)
      delete[] varx[i][j];
    delete[] varx[i];
  }
  delete[] varx;

  //free variables y
  for(i = 0; i < n; i++){
    for(j = 0; j < w; j++)
      delete[] vary[i][j];
    delete[] vary[i];
  }
  delete[] vary;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < w; ++j)
      free (block[i].yard[j]);
    free (block[i].yard);
    free (block[i].minS);
    free (block[i].pos);
    free (block[i].resh);
    free (block[i].from);
  }
  free (block);
  return (objVal);
}


int constructDataStructure (T_block *block, int n, int w, int h, int **yard, int ub) {
  int i, j, k, k1;
  for (i = 0; i < n; ++i) {
    block[i].w = -1;
    block[i].h = -1;
    Alloc (block[i].minS, w, int);
    Alloc (block[i].yard, w, int*);
    for (j = 0; j < w; ++j)
      Alloc (block[i].yard[j], h, int);
    Alloc (block[i].pos, n, int);
    Alloc (block[i].resh, n, int);
    Alloc (block[i].from, n, int);
  }
  for (j = 0; j < w; ++j)
    for (k = 0; k < h; ++k) {
      i = yard[j][k] - 1;
      if (i >= 0) {
	block[i].w = j;
	block[i].h = k;
	for (block[i].lb = k; (block[i].lb+1 < h) && (yard[j][block[i].lb+1] - 1 > i); block[i].lb++);
	for (k1 = 0; k1 < k; ++k1)
	  if (yard[j][k1] < i + 1)
	    block[i].lb = k;
      }
    }
  
  for (j = 0; j < w; ++j)
    for (k = 0; k < h; ++k)
      block[0].yard[j][k] = yard[j][k];
  for (i = 1; i < n; ++i) {
    for (j = 0; j < w; ++j)
      for (k = 0; k < h; ++k)
	block[i].yard[j][k] = block[i-1].yard[j][k];
    j = block[i-1].w;
    for (k = block[i-1].h; k <= block[i-1].lb; ++k)
      block[i].yard[j][k] = -1;
  }
  for (i = 0; i < n; ++i) {
    block[i].maxMin = -1;
    for (j = 0; j < w; ++j) {
      if (j != block[i].w) {
	block[i].minS[j] = findStackMin (n, h, block[i].yard[j]) - 1;
	if (block[i].minS[j] > block[i].maxMin)
	  block[i].maxMin = block[i].minS[j];
      }
      else{
	block[i].minS[j] = n;
	for(k = 0; k <= block[i].h; k++)
	  if((block[i].yard[j][k] > 0) && (block[i].yard[j][k] - 1 < block[i].minS[j]))
	    block[i].minS[j] = block[i].yard[j][k] - 1;	

	if (block[i].minS[j] > block[i].maxMin)
	  block[i].maxMin = block[i].minS[j];
      }
    }
  }

  block[0].pi = 0;
  for (i = 1; i < n; i++) {
    block[i].pi = i; 
    for (j = 0; j < w; j++)
      for (k = 0; k < h; k++)
	if (yard[j][k] == i + 1)
	  for (k1 = 0; k1 < k; k1++)
	    if ((yard[j][k1] < i + 1) && (yard[j][k1] < block[i].pi + 1))
	      block[i].pi = yard[j][k1] - 1;
  }

  return (0);
}



int addVariables (int n, int w, int h, GRBModel *model, GRBVar ***varx, GRBVar ***vary, int optimalitySwitch, T_block *block) {
  int i, j, s, t; 
  char varName[64];
  for(t = s = 0; t < n; t++)
    for(i = t; i < n; i++)//add binary variables x
      for(j = 0; j < w; j++,s++){
	  sprintf(varName,"X_%d_%d_%d", i+1, j+1, t+1);
	  if(optimalitySwitch == 0)
	    varx[i][j][t] = model->addVar(0.0, 1.0, 1.0, GRB_CONTINUOUS, varName);
	  else
	    varx[i][j][t] = model->addVar(0.0, 1.0, 1.0, GRB_BINARY, varName);
	}

  
  for(i = s = 0; i < n; i++)//add binary variables y
    for(j = 0; j < w; j++)
      for(t = 0; t < i; t++, s++){
	sprintf(varName,"Y_%d_%d_%d", i+1, j+1, t+1);
	if(optimalitySwitch == 0)
	vary[i][j][t] = model->addVar(0.0, 1.0, 1.0, GRB_CONTINUOUS, varName);
	else
	vary[i][j][t] = model->addVar(0.0, 1.0, 1.0, GRB_BINARY, varName);
      }


  model->update();
 
  return (0);
}


int addObjectiveFunction (int n, int w, GRBModel *model, GRBVar ***vary) { 
  int i, j, t;
  //objective function (1)
  GRBLinExpr xpr = 0;
  for(i = 0; i < n; i++) 
    for(t = 0; t < i; t++)
      for(j = 0; j < w; j++){
      xpr += vary[i][j][t];
    }
  model->setObjective(xpr, GRB_MINIMIZE);
  model->update();

  return (0);
}




int addConstraints (GRBEnv env, GRBModel *model, GRBVar ***varx, GRBVar ***vary, int n, int w, int h, T_block *block, int verbose) {
  int i, i1, cont, j, t;
  //Constraints initially added to the model
  
  // assignment constraints (2)
  for(i = cont = 0; i < n; i++)
    for (t = 0; t <= i; t++, cont++){
      GRBLinExpr xpr = 0;
      for(j = 0; j < w; j++)
	xpr += varx[i][j][t];      
      model->addConstr(xpr, GRB_EQUAL, 1.0);
    }
  // height constraints (3)
  for(j = cont = 0; j < w; j++) 
    for(t = 0; t < n; t++, cont++) {
      GRBLinExpr xpr = 0;
      for(i = t; i < n; i++)
	xpr += varx[i][j][t];      
      model->addConstr(xpr, GRB_LESS_EQUAL, ((double)h));
    }

  // do reshuffle constraints (4)
  for(i = cont = 0; i < n; i++)
    for(j = 0; j < w; j++)
      for(t = 0; t < i; ++t) {
	GRBLinExpr xpr = 0;
	xpr += varx[i][j][t];
	xpr -= varx[i][j][t+1];
	xpr -= vary[i][j][t];
	model->addConstr(xpr, GRB_LESS_EQUAL, 0.0);
      }

  // no reshuffle constraints (5)
  for(i = cont = 0; i < n; i++)
    for(j = 0; j < w; j++)
      for(t = 0; t < i; ++t, ++cont) {
	GRBLinExpr xpr = 0;
	xpr += varx[i][j][t+1];
	xpr += vary[i][j][t];
	model->addConstr(xpr, GRB_LESS_EQUAL, 1.0);
      }

  // restricted constraints (6)
  for(i = cont = 0; i < n; i++)
    for (t = 0; t < i; ++t) 
      for(j = 0; j < w; j++, cont++) {
	GRBLinExpr xpr = 0;
	xpr -= varx[t][j][t];
	xpr += vary[i][j][t];
	model->addConstr(xpr, GRB_LESS_EQUAL, 0.0);
      }

  // no reshuffle constraints (7)
  for(i = cont = 0; i < n; i++)
    for (t = 0; t < i; ++t) 
      for(j = 0; j < w; j++, cont++) {
	GRBLinExpr xpr = 0;
	xpr -= varx[i][j][t];
	xpr += vary[i][j][t];
	model->addConstr(xpr, GRB_LESS_EQUAL, 0.0);
      }



  //*****INITIAL CONFIGURATION ******
  
  
  // initial position constraints (10), (13)
  for (i = 0; i < n; ++i)
    for (t = 0; t <= block[i].pi; ++t) {
      GRBLinExpr xpr = 0;
      xpr += varx[i][block[i].w][t];      
      model->addConstr(xpr, GRB_EQUAL, 1.0);
    }

  // no intial blocks constraints (14)
  for (i = 0; i < n; ++i)
    for (i1 = 0; i1 < i; ++i1)
      if ((block[i].pi <= block[i1].pi) && (block[i1].pi == i1)) {
	GRBLinExpr xpr = 0;
	xpr += varx[i][block[i1].w][i1+1];      
	model->addConstr(xpr, GRB_EQUAL, 0.0);
      }

  
  // no reshufles from current stack constraints (15), (16)
    for(i = 0; i < n; i++)
    if(block[i].pi < i)
      for(t = block[i].pi +1; t < i; t++){
      int minb = i;
      if(t == block[i].pi+1)
	for(i1 = t; i1 < i; i1++)
	  if(block[i1].pi < block[i].pi || (block[i1].pi == block[i].pi && block[i1].h > block[i].h))
	    if(minb > i1)
	      minb = i1;
      if(t > block[i].pi+1)
	for(i1 = t; i1 < i; i1++)
	  if(block[i1].pi < t)
	    if(minb > i1)
	      minb = i1;

      for(j = 0; j < w; j++)
	if((t == block[i].pi+1 && j != block[i].w) || t > block[i].pi+1)
	  if(t < min(minb, block[t].minS[j])){
	    // constraints (15)
	    GRBLinExpr xpr = 0;
	    xpr += varx[i][j][t];
	    xpr -= varx[i][j][t+1];
	    model->addConstr(xpr, GRB_LESS_EQUAL, 0.0);
	  }

      for(j = 0; j < w; j++)
	if((t == block[i].pi+1 && j != block[i].w) || t > block[i].pi+1)
	  if(minb > block[t].minS[j])
	    minb = block[t].minS[j];
             
      for(j = 0; j < w; j++)
	if((t == block[i].pi+1 && j != block[i].w) || t > block[i].pi+1)
	  if(t < minb){
	    // constraints (16)
	    GRBLinExpr xpr = 0;
	    xpr -= varx[i][j][t];
	    xpr += varx[i][j][t+1];
	    model->addConstr(xpr, GRB_EQUAL, 0.0);	   
	  }
      }
   
    //reshuffle because the minimum of a stack constraints (17)
    for(i=0; i<n; i++)
      if(block[i].pi < i)
	for(j = 0; j < w; j++)
	  if(j!=block[i].w && block[block[i].pi].minS[j] < i){
	    GRBLinExpr xpr = 0;      	
	    xpr += varx[i][j][block[i].pi+1];
	    xpr -= vary[i][j][block[block[i].pi].minS[j]];
	    for(i1 = block[i].pi+1; i1 < block[block[i].pi].minS[j]; i1++)
	      if(block[i1].pi < block[i].pi || (block[i1].pi == block[i].pi && block[i1].h > block[i].h))
		xpr -= vary[i][j][i1];      
	    model->addConstr(xpr, GRB_LESS_EQUAL, 0.0);
	  }

 

  //reshuffle because a previous moved block constraints (18)
  for(i = 0; i < n; i++)
    if(block[i].pi < i)
      for(i1 = block[i].pi+1; i1 < i; i1++)
	if(block[i1].pi < block[i].pi || (block[i1].pi == block[i].pi && block[i1].h > block[i].h))
	  for(j = 0; j < w; j++){
	    if(j != block[i].w && block[block[i].pi].minS[j] > i1){
	      GRBLinExpr xpr = 0;
	      xpr += varx[i1][j][block[i].pi+1];
	      xpr += varx[i][j][block[i].pi+1];
	      for(int i2 = block[i].pi+1; i2 <= i1; i2++)
		if(block[i2].pi < block[i].pi || (block[i2].pi == block[i].pi && block[i2].h > block[i].h))
		  xpr -= vary[i][j][i2];
	      model->addConstr(xpr, GRB_LESS_EQUAL, 1.0);
	    }
	  }

  model->update();

  return (0);
}


int setMipParameters (GRBModel *model) {
  model->set(GRB_DoubleParam_TimeLimit, 3600.0);
  model->set(GRB_IntParam_Threads, 1);
  model->set(GRB_IntParam_LazyConstraints , 1);
  model->update();
  return (0);
}


int copyInitialSolution (GRBModel *model, GRBVar ***varx, GRBVar ***vary, int n, int w, int h, int ***solution) {
  //Copy the initial solution provided by the BBS algorithm
  int i, j, k, t;
  int stack, slot;
  double ***startx, ***starty;
  FILE *iF;

  Alloc (startx, n, double**);
  for (i = 0; i < n; i++){
    Alloc (startx[i], w, double*);
    for (j = 0; j < w; j++)
      Alloc (startx[i][j], n + 1, double); 
  }

  Alloc (starty, n, double**);
  for (i = 0; i < n; i++){
    Alloc (starty[i], w, double*);
    for (j = 0; j < w; j++)
      Alloc (starty[i][j], n + 1, double); 
  }

  for (i = 0; i < n; i++)
    for (j = 0; j < w; j++)
      for (t = 0; t < n + 1; t++){
	startx[i][j][t] = 0.0;
	starty[i][j][t] = 0.0;
      }



  for (t = 0; t < n + 1; t++){
    stack = -1;
    slot = -1;
    for(j = 0; j < w; j++) 
      for(k = 0 ; k < h; k++)
	if(solution[t][j][k] > 0){
	  startx[solution[t][j][k]-1][j][t] = 1.0;
	  if(solution[t][j][k] == t + 1){
	    stack = j;
	    slot = k;
	  }
	  if((j == stack) && (k > slot)){
	    starty[solution[t][j][k]-1][j][t] = 1.0;
	  }
	}
  }      

  
  for(i = 0; i < n; i++)//add initial solution
    for(j = 0; j < w; j++)
      for(t = 0; t <= i; t++){
	varx[i][j][t].set(GRB_DoubleAttr_Start, startx[i][j][t]);
	if(t<i)
	  vary[i][j][t].set(GRB_DoubleAttr_Start, starty[i][j][t]);
      }

  model->update();

  for (i = 0; i < n; i++) {
    for (j = 0; j < w; j++)
      free (startx[i][j]);
    free (startx[i]);
  }
  free (startx);
  for (i = 0; i < n; i++) {
    for (j = 0; j < w; j++)
      free (starty[i][j]);
    free (starty[i]);
  }
  free (starty);

  return (0);
}


int setLazyConstraints (GRBModel *model, T_block *block, int n, int w, int h) {
  int i, t;
  //Prepare initial configuration for the dynamic constraints generation
  for (t = 0; t < n; ++t)
    for (i = 0; i < n; ++i)
      if (t <= block[i].pi)
	block[t].pos[i] = block[i].w;

  for (t = 0; t < n; ++t)
    for (i = 0; i < n; ++i)
      if (t <= block[i].pi){
	block[t].resh[i] = 0;
	block[t].from[i] = -1;
      }
  
  for (t = 0; t < n; ++t)
    for (i = 0; i < n; ++i)
      if (t == block[i].pi && block[i].pi < i){
	block[t].resh[i] = 1;
	block[t].from[i] = block[i].w;
      }
  return (0);
}


int findStackMin (int n, int card, int *v) {
  int min = n+1;
  int i;
  for (i = 0; (i < card) && (v[i] > 0); ++i)
    if (min > v[i])
      min = v[i];
  return (min);
}


int adjbin (int n, int* v, int *w, int val) {
  int var = -1;
  int pos = -1;
  do 
    {
      if (v[++pos] ^= 1) {
	if ((pos == n - 1) || !v[pos + 1]) {
	  var = 1;
	}
	break;
      } 
    } while (pos < n - 1);
  w[pos] += var;
  if (!pos) val++;
  else { 
    if (var > 0) val = val - (pos - var); 
    else val = val - (pos + var);
  }
  return(val);
}



int saveSolution (int n, int w, int h, 
		  int **Yard, T_block *block, GRBModel model, GRBVar ***varx, GRBVar ***vary, int ***solution) {
  int i, i1, j, t,k;
  double ***valx, ***valy;
  double eps = 0.5;
  
  Alloc (valx, n, double**);
  for (i = 0; i < n; ++i) {
    Alloc (valx[i], w, double*);
    for (j = 0; j < w; ++j)
      Alloc (valx[i][j], n+1, double);
  }

  Alloc (valy, n, double**);
  for (i = 0; i < n; ++i) {
    Alloc (valy[i], w, double*);
    for (j = 0; j < w; ++j)
      Alloc (valy[i][j], n+1, double);
  }

  for (i = 0; i < n; ++i)
    for (j = 0; j < w; ++j)
      for (t = 0; t <= i; ++t)
	valx[i][j][t] = varx[i][j][t].get(GRB_DoubleAttr_X);


  for (i = 0; i < n; ++i)
    for (j = 0; j < w; ++j)
      for (t = 0; t < i; ++t)
	valy[i][j][t] = vary[i][j][t].get(GRB_DoubleAttr_X);

  for (t = 0; t < n+1; ++t)
    for(j=0; j<w; j++)
      for(k=0; k<h; k++)
	solution[t][j][k] = -1;
  
  for(j=0; j<w; j++)
    for(k=0; k<h; k++)
      solution[0][j][k] = Yard[j][k];

  for(t=0;t<n-1;t++){
    
    //copy yard time t to time t+1
    for(j=0; j<w; j++)
      for(k=0; k<h; k++)
	solution[t+1][j][k] = solution[t][j][k];
    //search item t and remove it after reshuffle all blocking items
    for(j = 0; j < w; j++)
      if(valx[t][j][t] > eps){
	//find stack of item t
	int slot;
	for(k = 0; k < h; k++)
	  if(solution[t][j][k] == t+1)
	    slot = k;
	
	k = h-1;
	while(solution[t][j][k] < 0)
	  k--;

	//while the slot is not the one of item t
	while(k > slot){
	  i1 = solution[t][j][k]-1;
	  
	  //remove blocking item
	  solution[t+1][j][k] = -1;
	  int j1;
	  for(j1 = 0; j1 < w; j1++)
	    if(valx[i1][j1][t+1] > eps){
	      int k1 = 0;
	      while(solution[t+1][j1][k1] > 0)
		k1++;
	      solution[t+1][j1][k1] = i1+1;
	    }
	  k--;
	}
	
	solution[t+1][j][slot] = -1;

	
      }
  }
  /*
  for (t = 0; t < n; ++t) {
    printf ("t = %d **************************************************\n", t+1);
    for (j = 0; j < w; ++j){
      for(k=0;k<n;k++)
	if(yardfrac[t][j][k]>0)
	  printf("%d ",yardfrac[t][j][k]);
      printf("\n");
    }
    printf("\n");
    for (i = t; i < n; ++i)
      for (j = 0; j < w; ++j)
	if (valx[i][j][t] > eps)
	  printf ("X[%d][%d][%d] = %f\n", i+1, j+1, t+1, valx[i][j][t]);
    printf ("\n");
    for (i = t+1; i < n; ++i)
      for(j = 0; j < w; j++)
	if (valy[i][j][t] > eps)
	  printf ("Y[%d][%d][%d] = %f\n", i+1, j+1, t+1, valy[i][j][t]);
    printf ("\n");

  }
*/
  for(i = 0; i < n; i++){
    for(j = 0; j < w; j++)
      free(valx[i][j]);
    free(valx[i]);
  }
  free(valx);

  for(i = 0; i < n; i++){
    for(j = 0; j < w; j++)
      free(valy[i][j]);
    free(valy[i]);
  }
  free(valy);


  return (0);
}

  

int min (int a, int b) {
  if (a < b) return a;
  else return b;
}
