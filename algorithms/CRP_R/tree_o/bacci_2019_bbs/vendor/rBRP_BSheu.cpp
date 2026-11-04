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
/* The present file includes all functions needed for the Bounded Beam
 Search algorithm for the restricted Block Relocation Problem introduced
 in Bacci et al. (2019)
*/
  
#include <inttypes.h>
#include <time.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#define std_min(a,b) ((a)<(b)?(a):(b))


#define Alloc(p,a,t) do						\
    if (!(p = (t *) malloc ((size_t) ((a) * sizeof (t))))) {	\
      printf("run out of memory [Alloc(%s,%d,%s)]\n",#p,a,#t);	\
      exit(1);							\
    } while (0)


typedef struct{
  int UB;                   //UB of the node
  int LB;                   //LB of the node
  int RS;                   //cost to reach node from initial configuration
  int **yard;               // current yard of the node -> yard[j][k] = i if in stack j, slot k is allocated item i (yard[j][k] = -1 if empty slot)
  int *minstack;            // minstack[j] = i if i is the item with higher retrieval priority in stack j
  int *freeslot;            // freestack[j] = q if q is the number of empty slots in stack j
  int **map;                // map[i][0] = j if item i is allocated in stack j, map[i][1] = k if item i is allocated in slot k
  int *solution;            // vector solution collects the destination stacks for each reshuffle
  int blNo;                 //number of blocking items in the current yard  
  int minItem;             //the next item to be retrieved in the current yard
}nodebeam; // collects all the best nodes of a level during the beam search

typedef struct{
  int it; //item to reshuffle in the current node
  int UB; //UB of the node
  int LB; //LB of the node
  int dstack; //destination stack of the item to reshuffle in the current node          
  int father; // father of the current node
  int order;  // order is used to order the nodes according to the best UB  (in case of tie to the best LB) 
  int minItem; //the next item to be retrieved in the current node
}nodesearch; // collects all nodes of a level during the beam search



int copyYard(int n, int w, int h, int **yard, int *minstack, int *freeslot, int **map, int **tyard, int *tminstack, int *tfreeslot, int **tmap, int stack); // function to copy yard, minstack, freeslot, map -> in -> tyard, tminstack, tfreeslot, tmap 
void constructSolution(int n, int w, int h, int **yard, int GUB, int bestSolCard, int *bestSol, nodebeam *VetBeam, int ***bbs_solution); // function to print solution in file solutionBBS.txt

//functions of the fast heuristic algorithms for computing upper bounds
int Difference1(int n, int w, int h, int minItem, int blNo, int **yardIn, int *Min_Stack, int *free_slot, int **Mat, int BestUB, int *dstack); // computes an upper bound accoding to heuristic Difference1
int GAH(int n, int w, int h, int minItem, int blNo, int **yardIn, int *Min_Stack, int *free_slot, int **Mat, int **vet, int *B, int *idB, int BestUB, int *dstack); // computes an upper bound accoding to heuristic GAH
int ChainF(int n, int w, int h, int minItem, int blNo, int **yardIn, int *Min_Stack, int *free_slot, int **Mat, int BestUB, int *dstack); // computes an upper bound accoding to heuristic ChainF
//*********************************************************************

int removeItems(int n, int w, int h, int minItem, int **yard, int *minstack, int *freeslot, int **map, int *item); // to retrieve, according to the exit order, all the items thta are not blocked in yard

int UBALB(int n, int w, int h, int minItem, int **yard, int *minstack, int *freeslot, int **map, int *phi, int *minst, int *freesl, int *L, int *R); // computes lower bound UBALB by Bacci et al.
//***** Functions used by lower bound UBALB ****************************************
void merge(int *arr, int l, int m, int r, int *L, int *R);
void mergeSort(int *arr, int l, int r, int *L, int *R);
void mergeCrit1(int *arr, int l, int m, int r, int *L, int *R, int *id);
void mergeSortCrit1(int *arr, int l, int r, int *L, int *R, int *id);
//***** Functions used by lower bound UBALB ****************************************

//Function for the Bounded Beam Search algorithm
int rBRP_BSheu (//INPUT
		int n,//number of items in the initial yard (n>1)
		int w,//number of stacks int the initial yard
		int h,//number of slots for each stack in the initial yard
		int **yard, //initial yard -> yard[j][k]=i if in stack j^th, slot k^th is located item i, -1 if the slot is empty
		int ***bbs_solution, //solution of BBS algorithm: bbs_solution[t][j][k]=i if in stack j^th, slot k^th is located item i at time period t, -1 if the slot is empty
		double timelimit //time limit in seconds (default: 5.0)
		)
{
  
  
  int i, j, k, q, item, check, st;

  int beta;//the beam width
  
  if(n < 40)
    beta = 800;
  if(n >= 40 && n < 60)
    beta = 500;
  if(n >= 60 && n < 80)
    beta = 300;
  if(n >= 80 && n < 100)
    beta = 200;
  if(n >= 100 && n < 120)
    beta = 100;
  if(n >= 120)
    beta = 50;

  int blNo,ub;
  int searchNode, beamNode;
  int minItem;
  int GUB = n * n;
  double timeH = 0.0, time1, time2; // timeH : total elapsed time during the beam search algorithm
  timeval tim;
  int stack, slot, newiteration, checkempty;
  int cardBeam = 0, cardBest = 0, cardSearch = 0;
  
  int *tminstack, *tfreeslot, **tyard, **tmap;//temporary vectors used to copy informations about the yard
  
  nodebeam *VetBeam, *VetBest, *tVet; //VetBeam collect the best beta nodes that are used to construct the next level of the search tree.
                                      //VetBest collects all the best beta nodes taken by the previous construction of one level.
                                      //At the end of each iteration all the nodes in VetBest are copied into VetBeam.
                                      //VetBeam and VetBest are distinct in order to avoid, during the construction of one level,
                                      //to memory for all possible nodes the yard configuration 
  nodesearch *VetSearch;              //VetSearch collects all the possible nodes obtained from VetBeam during the construction of the next level
  int *bestSol, bestSolCard, dstack;

  
  //************** vectors for GAH heuristic ************************************************
  int **vet,*B,*idB;//vet[][] collect all the blocking items that have to be reshuffled - vet[i][0] = i, vet[i][1] = index of the i^th blocking item, vet[i][2] = slot where the i^th blocking item is assigned in the origin stack, vet[i][3] = destination stack of the i^th blocking item, vet[i][4] = destination slot of the i^th blocking item
  //B[j] = number of blocking item that in the origin stack are located under the current blocking item and whose destination stack is the current stack considered
  //idB[j] = if B[j]<=1 then idB[j] is the index of the blocking item that in the origin stack are located under the current blocking item and whose destination stack is the current stack considered
  //*******************************************************************************************

  //************** vectors for UBALB **********
  int *phi, *minst, *freesl, *L, *R;
  //*******************************************



  gettimeofday(&tim, NULL);
  time1=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));

  Alloc(phi, n, int);
  Alloc(freesl, w, int);
  Alloc(minst, n, int);
  Alloc(L, n, int);
  Alloc(R, n, int);

  
  Alloc(vet,h,int*);
  for(i=0;i<h;i++)
    Alloc(vet[i],5,int);
  Alloc(B,w,int);
  Alloc(idB,w,int);
  


  Alloc (bestSol, n * n,int);

  Alloc(VetBeam, beta, nodebeam);
  Alloc(VetBest, beta, nodebeam);
  Alloc(VetSearch, beta * w, nodesearch);
  
  for(q = 0; q < beta; q++) {
    Alloc(VetBeam[q].yard, w, int*);
    Alloc(VetBest[q].yard, w, int*);
    for(j = 0; j < w; j++) {
      Alloc(VetBeam[q].yard[j], h, int);
      Alloc(VetBest[q].yard[j], h, int);
    }
    
    Alloc(VetBeam[q].map, n, int*);
    for (j = 0; j < n; ++j)
      Alloc(VetBeam[q].map[j], 2, int); 

    Alloc(VetBest[q].map, n, int*);
    for (j = 0; j < n; ++j)
      Alloc(VetBest[q].map[j], 2, int); 

    
    Alloc(VetBeam[q].minstack, w, int);
    Alloc(VetBest[q].minstack, w, int);
    Alloc(VetBeam[q].freeslot, w, int);
    Alloc(VetBest[q].freeslot, w, int);

    Alloc(VetBeam[q].solution, n * n,int);
    Alloc(VetBest[q].solution, n * n,int);
    VetBeam[q].solution[0] = 0;
    VetBest[q].solution[0] = 0;
    
  }

  Alloc(tminstack, w, int);
  Alloc(tfreeslot, w, int);

  Alloc(tyard, w, int*);
  for(i = 0; i < w; i++)
    Alloc(tyard[i], h, int);
  
  Alloc(tmap, n, int*);
  for(i = 0; i < n; i++)
    Alloc(tmap[i], 2, int);

  cardBeam = 1;


  // INITIALIZE THE ROOT NODE
  VetBeam[0].RS = 0; 
  VetBeam[0].blNo = 0;
  VetBeam[0].minItem = 1;
  for(j = 0; j < w; j++) {
    VetBeam[0].minstack[j] = n + 1;
    VetBeam[0].freeslot[j] = 0;
    for(k = 0 ; k < h; k++){
      VetBeam[0].yard[j][k] = yard[j][k];
      if(VetBeam[0].yard[j][k] == -1) 
	VetBeam[0].freeslot[j] ++;
      else {
	VetBeam[0].map[yard[j][k]-1][0] = j;
	VetBeam[0].map[yard[j][k]-1][1] = k;
	if(VetBeam[0].yard[j][k] < VetBeam[0].minstack[j])
	  VetBeam[0].minstack[j] = VetBeam[0].yard[j][k];
	for (i = check = 0; ((i < k) && (!check)); ++i)
	  if (yard[j][i] < yard[j][k]) {
	    VetBeam[0].blNo++;
	    check = 1;
	  }
      }    
    }
  }
  gettimeofday(&tim, NULL);
  time2=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));
  timeH = timeH + (time2 - time1);
  // CALCULATE THE ROOT NODE UB
  gettimeofday(&tim, NULL);
  time1=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));
    
  copyYard (n, w, h, VetBeam[0].yard, VetBeam[0].minstack, VetBeam[0].freeslot, VetBeam[0].map, tyard, tminstack, tfreeslot, tmap,-1);
  dstack = 0;
  if(n >= 1000 && n < 10000)
    ub = Difference1(n, w, h, VetBeam[0].blNo, VetBeam[0].minItem, tyard, tminstack, tfreeslot, tmap, n*n,&dstack);
  if(n >= 10000)
    ub=GAH(n, w, h, VetBeam[0].blNo, VetBeam[0].minItem, tyard, tminstack, tfreeslot, tmap, vet, B, idB, n*n,&dstack);
  if(n < 1000)   
    ub = ChainF(n, w, h, VetBeam[0].blNo, VetBeam[0].minItem, tyard, tminstack, tfreeslot, tmap, n*n,&dstack);
  
  VetBeam[0].UB = ub;
    
  gettimeofday(&tim, NULL);
  time2=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));
  timeH = timeH + (time2 - time1);

  GUB = VetBeam[0].UB;
  bestSolCard = 0;
  newiteration = 1;
   
  if(timeH > timelimit)
    goto freeMemory;
  


 
  
  while(newiteration) {
    cardSearch = 0;
    // SCROLL THE FATHER's LIST
    for(q = 0; ((q < cardBeam) && (newiteration)); q++) {
      
      minItem = VetBeam[q].minItem;
      minItem = removeItems (n, w, h, minItem, VetBeam[q].yard, VetBeam[q].minstack, VetBeam[q].freeslot, VetBeam[q].map, &item);

      if(minItem < n){
    
	if(VetBeam[q].blNo<0)
	  VetBeam[q].blNo = 0;
	
	gettimeofday(&tim, NULL);
	time1=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));
		
	copyYard (n, w, h, VetBeam[q].yard, VetBeam[q].minstack, VetBeam[q].freeslot, VetBeam[q].map, tyard, tminstack, tfreeslot, tmap,-1);
	VetBeam[q].LB = VetBeam[q].blNo + UBALB(n, w, h, minItem, tyard, tminstack, tfreeslot, tmap, phi, minst, freesl, L, R) + VetBeam[q].RS;
		
	gettimeofday(&tim, NULL);
	time2=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));
       

	timeH = timeH + time2 - time1;
	if (VetBeam[q].LB < GUB) {
	  stack = VetBeam[q].map[item-1][0];
	  slot = VetBeam[q].map[item-1][1];
	  VetBeam[q].freeslot[stack]++;
	  VetBeam[q].yard[stack][slot] = -1;
	
	  checkempty = 0;    // 1 if item already assigned to an empty stack
	  // CREATE THE SONS OF FATHER q

	  for(st = 0; st < w; st++) 	    
	    if ((st != stack) && (VetBeam[q].freeslot[st]) && (((VetBeam[q].freeslot[st] == h) && (!checkempty ))||(VetBeam[q].freeslot[st] < h))) {


	      if (VetBeam[q].freeslot[st] == h)
		checkempty = 1;	      
	    
	      VetSearch[cardSearch].order = cardSearch;	
	      VetSearch[cardSearch].it = item;
	      VetSearch[cardSearch].minItem = minItem;
	      
	      // CALCULATE THE LB OF THE st-h SON OF FATHER q	      
	      VetSearch[cardSearch].LB = VetBeam[q].LB;

	      gettimeofday(&tim, NULL);
	      time1=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));

	      
	      copyYard (n, w, h, VetBeam[q].yard, VetBeam[q].minstack, VetBeam[q].freeslot, VetBeam[q].map, tyard, tminstack, tfreeslot, tmap,stack);
		

		
	      tmap[item-1][0] = st;
	      tmap[item-1][1] = h - tfreeslot[st];
	      tyard[st][h-tfreeslot[st]] = item;
	      blNo = VetBeam[q].blNo;
	      if (tminstack[st] > item) {
		blNo--;
		tminstack[st] = item;
	      }
	      tfreeslot[st]--;
		
		  
		
	      // CALCULATE THE UB OF THE st-h SON OF FATHER q
	      dstack = 0;
	      if(n >= 1000 && n < 10000)
		ub = Difference1(n, w, h, blNo, minItem, tyard, tminstack, tfreeslot, tmap, GUB,&dstack);
	      if(n >= 10000)
		ub=GAH(n, w, h, blNo, minItem, tyard, tminstack, tfreeslot, tmap, vet, B, idB, GUB,&dstack);
	      if(n < 1000)
		ub = ChainF(n, w, h, blNo, minItem, tyard, tminstack, tfreeslot, tmap, GUB,&dstack);
	      
		
	      VetSearch[cardSearch].UB = VetBeam[q].RS + 1 + ub;
		
		
		
	      VetSearch[cardSearch].dstack = st;
	      VetSearch[cardSearch].father = q;
		
	      if(VetSearch[cardSearch].UB < GUB) {     
		GUB = VetSearch[cardSearch].UB;
		bestSolCard = VetBeam[q].RS + 1;
		for (j = 0; j < bestSolCard - 1; ++j)
		  bestSol[j] = VetBeam[q].solution[j];
		bestSol[bestSolCard-1] = st;
	      }
		
	      
	      cardSearch++;
		
	      gettimeofday(&tim, NULL);
	      time2=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));
		
	      timeH = timeH + time2 - time1;
		
	      if(timeH > timelimit) 
		goto freeMemory;
	      
		
	      
	      
	    }
	  
	}
	
      }
      
    }
    // FOR ALL THE NEW NODES: LB < GUB <= UB 
    // ORDER THE NEW NODES
   
    gettimeofday(&tim, NULL);
    time1=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));

    for(i = 0; i < cardSearch - 1; i++)
      for(j = i + 1; j < cardSearch; j++) {

	if(VetSearch[VetSearch[j].order].UB < VetSearch[VetSearch[i].order].UB ||
	   (VetSearch[VetSearch[j].order].UB == VetSearch[VetSearch[i].order].UB &&
	    VetSearch[VetSearch[j].order].LB < VetSearch[VetSearch[i].order].LB)) {
	  k = VetSearch[i].order;
	  VetSearch[i].order = VetSearch[j].order;
	  VetSearch[j].order = k;

	}

      }

    gettimeofday(&tim, NULL);
    time2=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));

    timeH = timeH + time2 - time1;
    
    if(timeH > timelimit)
      goto freeMemory;
    
    
    // CREATE THE NEW BEAM NODES
    if (cardSearch < beta)
      cardBest = cardSearch;
    else
      cardBest = beta;

    if(cardBest<beta){
      gettimeofday(&tim, NULL);
      time1=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));
    }


    for (i = 0; i < cardBest; ++i) {

      searchNode = VetSearch[i].order;
      beamNode = VetSearch[searchNode].father;
      VetBest[i].UB = VetSearch[searchNode].UB;
      VetBest[i].RS = VetBeam[beamNode].RS + 1;
      VetBest[i].minItem = VetSearch[searchNode].minItem;

      copyYard (n, w, h, VetBeam[beamNode].yard, VetBeam[beamNode].minstack, VetBeam[beamNode].freeslot, VetBeam[beamNode].map,
		VetBest[i].yard, VetBest[i].minstack, VetBest[i].freeslot, VetBest[i].map,-1);

      item = VetSearch[searchNode].it;

      // UPDATE DESTINATION STACK
      stack = VetSearch[searchNode].dstack;
      slot = h - VetBest[i].freeslot[stack];
      VetBest[i].freeslot[stack]--;
      VetBest[i].yard[stack][slot] = item;
      VetBest[i].blNo = VetBeam[beamNode].blNo;
      if (item < VetBest[i].minstack[stack]) {
	VetBest[i].blNo--;
	VetBest[i].minstack[stack] = item;
      }
      
      VetBest[i].map[item-1][0] = stack;
      VetBest[i].map[item-1][1] = slot;
      
      // UPDATE SOLUTION
      for (j = 0; j < VetBeam[beamNode].RS; ++j)
	VetBest[i].solution[j] = VetBeam[beamNode].solution[j];
      VetBest[i].solution[VetBeam[beamNode].RS] = stack;


    }

    if(cardBest<beta){
      gettimeofday(&tim, NULL);
      time2=(double)(tim.tv_sec+(tim.tv_usec/1000000.0));
    }

    if(cardBest<beta)
      timeH = timeH + time2 - time1;
    
    if(timeH > timelimit)
     goto freeMemory;
    


    
    if (!cardSearch)
      newiteration = 0;
    tVet = VetBeam;
    VetBeam = VetBest;
    VetBest = tVet;
    cardBeam = cardBest;
  }
  


  goto freeMemory;
    
 freeMemory:
    
  constructSolution(n, w, h, yard, GUB, bestSolCard, bestSol, VetBeam, bbs_solution);
  
    
    free(phi);
    free(minst);
    free(freesl);
    free(L);
    free(R);

  
    for(q = 0; q < beta; q++) {
      for(j = 0; j < w; j++) {
	free(VetBeam[q].yard[j]);
	free(VetBest[q].yard[j]);
      }
    
      free(VetBeam[q].yard);
      free(VetBest[q].yard);    
    
    
      free(VetBeam[q].minstack);
      free(VetBeam[q].freeslot);    
    
      free(VetBest[q].minstack);
      free(VetBest[q].freeslot);

      free(VetBeam[q].solution);        
      free(VetBest[q].solution);
    
      for (j = 0; j < n; ++j)
	free(VetBeam[q].map[j]);
      free(VetBeam[q].map);

      for (j = 0; j < n; ++j)
	free(VetBest[q].map[j]);
      free(VetBest[q].map);

    }
 
    free(VetBeam);
    free(VetBest);
    free(VetSearch);
 
    free(tminstack);
    free(tfreeslot);

    for(i = 0; i < w; i++) 
      free(tyard[i]);
    free(tyard);
    for(i = 0; i < n; i++) 
      free(tmap[i]);
    free(tmap);
    free(bestSol);
    for(i=0;i<h;i++)
      free(vet[i]);
    free(vet);
    free(B);
    free(idB); 

    return (GUB);
}

//Functions for computing upper and lower bounds
//heuristic Difference1 introduced in Unluyurt and Aydin (2012)
int Difference1(//INPUT
		int n,           
		int w,           
		int h,
		int blNo, 
		int minItem,     // minimum item in the yard
		int **yard,
		int *minstack,
		int *freeslot,
		int **map,
		int BestUB,
		int *dstack
		)
{
  int i, j, rit, stack, slot, t;
  int UBT;      //value of the solution
  int bbs, bns, bbs2; //bbs = best blocking stack, bns = best non blocking stack
  int bbv, bnv, bbv2; //bbv = best blocking stack UB, bnv = best non blocking stack UB
  int bs;   // bv = best UB value, bs = best UB stack

  UBT = 0;
  
  while(minItem <= n){
    if(UBT + blNo >= BestUB)    
      return (n * n);
    
    stack = map[minItem-1][0];//saves stack and slot of the next item t to be retrieved, removes t and updates configuration
    slot = map[minItem-1][1];

    for (t = h - freeslot[stack] - 1; t > slot; --t) {
      
      rit = yard[stack][t];
      bbv = -1;
      bnv = bbv2 =  n * n;
      bbs = bns = bbs2 = -1;
      yard[stack][t] = -1;	
      freeslot[stack]++;
      for(j = 0; j < w; j++) {
	if((j != stack) && (freeslot[j])) {
	  if(minstack[j] < rit) {
	    if ((yard[j][h - freeslot[j] - 1]) < rit && (yard[j][h - freeslot[j] - 1]) > bbv) {
	      bbv = (yard[j][h - freeslot[j] - 1]);
	      bbs = j;
	    }
	     
	    if ((yard[j][h - freeslot[j] - 1]) < bbv2) {
	      bbv2 = (yard[j][h - freeslot[j] - 1]);
	      bbs2 = j;
	    }
	     	      
	  }
	  else {
	    if (minstack[j] < bnv) {
	      bnv = minstack[j];
	      bns = j;
	    }
	  }
	}
      }
      if (bns != -1) {
	bs = bns;
	minstack[bs] = rit;
	blNo--;
      }
      else{
	if(bbs != -1)
	  bs = bbs;
	else
	  bs = bbs2;
      }
     
      UBT++;
      if((*dstack)==-1)
	(*dstack) = bs;
      
      yard[bs][h-freeslot[bs]] = rit;
      map[rit-1][0] = bs;
      map[rit-1][1] = h - freeslot[bs];
      freeslot[bs] --;
    }

    yard[stack][slot] = -1;
    freeslot[stack]++;
    minstack[stack] = n + 1;
    for(i = 0; i < slot; i++)
      if(yard[stack][i] < minstack[stack])
	minstack[stack] = yard[stack][i];

    minItem++;
  }
  return UBT;
}


//heuristic GAH introduced in Wu and Ting (2012)
int GAH(//INPUT
	int n,           
	int w,           
	int h,
	int blNo, 
	int minItem,     // minimum item in the yard
	int **yard,
	int *minstack,
	int *freeslot,
	int **map,
	int **vet,
	int *B,
	int *idB,
	int BestUB,
	int *dstack
	)
{
  int i, j, rit, stack, slot, t, nblocks, p, oldmin;
  int UBT;      //value of the solution
  int bbs, bns; //bbs = best blocking stack, bns = best non blocking stack
  int bbv, bnv; //bbv = best blocking stack UB, bnv = best non blocking stack UB
  int bs;   // bv = best UB value, bs = best UB stack
  int nextit;
 
 
  
  UBT = 0;
  
  while(minItem <= n && blNo > 0){
    if(UBT + blNo >= BestUB)    
      return (n * n);
    
     
    stack = map[minItem-1][0];//saves stack and slot of the next item t to be retrieved, removes t and updates configuration
    slot = map[minItem-1][1];

    if(slot<h-1 && yard[stack][slot+1]!=-1){

      nblocks=0;
      for (t = h - freeslot[stack] - 1; t > slot; --t) {
	vet[nblocks][0]=nblocks;
	vet[nblocks][1]=yard[stack][t];
	vet[nblocks][2]=t;
	vet[nblocks][3]=-1;
	vet[nblocks][4]=-1;
	nblocks++;
	if((*dstack) == -1 && t == h - freeslot[stack] - 1)
	  nextit = yard[stack][t];
      }    

    
      for (t = 0; t < nblocks-1; t++) {
	for (i = t+1; i < nblocks; i++) {
	  if(vet[vet[t][0]][1]<vet[vet[i][0]][1]){
	    j=vet[t][0];
	    vet[t][0]=vet[i][0];
	    vet[i][0]=j;
	  }
	}
      }    

      for(j = 0; j < w; j++) {
	B[j]=0;
      }
    
      for (t = 0; t < nblocks; t++) {
	rit = vet[vet[t][0]][1];
	bnv = n * n;
	bns = -1;

	for (i = 0; i < nblocks; i++) {
	  if(vet[vet[t][0]][2]>vet[vet[i][0]][2] && vet[vet[t][0]][1]<vet[vet[i][0]][1] && vet[vet[i][0]][3]!=-1){
	    B[vet[vet[i][0]][3]]++;
	  }
	}

	for(j = 0; j < w; j++) {
	  if((j != stack) && (freeslot[j]) && B[j]==0) {
	    if (minstack[j] >= rit && minstack[j] < bnv) {
	      bnv = minstack[j];
	      bns = j;
	    }
	  }
	  B[j]=0;
	}
	if (bns != -1) {
	  bs = bns;
	  minstack[bs] = rit;
	  blNo--;
	  UBT++;
	  if((*dstack) == -1 && rit == nextit)
	    (*dstack) = bs;

	  yard[stack][vet[vet[t][0]][2]] = -1;	
	  freeslot[stack]++;      
	  yard[bs][h-freeslot[bs]] = rit;
	  vet[vet[t][0]][3]=bs;
	  vet[vet[t][0]][4]= h - freeslot[bs];
	  map[rit-1][0] = bs;
	  map[rit-1][1] = h - freeslot[bs];
	  freeslot[bs] --;


	}

      }


    
      for (t = nblocks - 1; t >= 0; t--) {
	if(vet[vet[t][0]][3]==-1){

	  for(j = 0; j < w; j++) {
	    B[j]=0;
	    idB[j]=0;
	  }

	  
	  for (i = 0; i < nblocks; i++) {
	    if(vet[vet[t][0]][2]>vet[vet[i][0]][2] && vet[vet[t][0]][1]<vet[vet[i][0]][1] && vet[vet[i][0]][3]!=-1){
	      if(B[vet[vet[i][0]][3]]==0)
		idB[vet[vet[i][0]][3]]=vet[vet[i][0]][1];
	      B[vet[vet[i][0]][3]]++;
	    }
	  }
	  rit = vet[vet[t][0]][1];
	  bbv = - n * n;
	  bnv = n * n;
	  bbs = bns = -1;
	  yard[stack][vet[vet[t][0]][2]] = -1;	
	  freeslot[stack]++;
	  for(j = 0; j < w; j++) {
	    if((j != stack) && (freeslot[j])) {
	    
	      if(B[j]<=1)
		p=std_min(minstack[j]-rit,rit-idB[j]);
	      else
		p=-n+1;

	      if(p < 0) {
		if (p > bbv) {
		  bbv = p;
		  bbs = j;
		}
	      }
	      else {
		if (p < bnv) {
		  bnv = p;
		  bns = j;
		}
	      }
	    }
	  }
	  if (bns != -1) {
	    bs = bns;
	  }
	  else
	    bs = bbs;

	
	  UBT++;
	  j=-1;
	  vet[vet[t][0]][3]=bs;

	  for (i = 0; i < nblocks; i++) {
	    if(bs==vet[vet[i][0]][3] && vet[vet[t][0]][2]>vet[vet[i][0]][2] && vet[vet[i][0]][2]>j && i!=t){
	      j=vet[vet[i][0]][2];
	      p=vet[vet[i][0]][1];
	    }
	  }

	

	  if(j!=-1){
	    oldmin = n + 1;
	    for(i=0;i<map[p-1][1];i++){
	      if(oldmin > yard[bs][i])
		oldmin = yard[bs][i];
	    }
	    for(i=h-freeslot[bs];i>map[p-1][1];i--){
	      yard[bs][i]=yard[bs][i-1];
	      map[yard[bs][i]-1][1]=i;
	      if(yard[bs][i]>rit && yard[bs][i]<=oldmin)
		blNo++;
	    }
	    vet[vet[t][0]][4]=map[p-1][1]-1;
	  }else
	    vet[vet[t][0]][4]=h-freeslot[bs];
	
	  if(minstack[bs]>rit){
	    minstack[bs] = rit;
	    blNo--;
	  }

	  if((*dstack) == -1 && rit == nextit)
	    (*dstack) = bs;
	
	  yard[bs][vet[vet[t][0]][4]] = rit;
	  map[rit-1][0] = bs;
	  map[rit-1][1] = vet[vet[t][0]][4];
	  freeslot[bs] --;




	}
      }
    }
	
    yard[stack][slot] = -1;
    freeslot[stack]++;
    minstack[stack] = n + 1;
    for(i = 0; i < slot; i++)
      if(yard[stack][i] < minstack[stack])
	minstack[stack] = yard[stack][i];


    minItem++;
  }

  return UBT;
}

//heuristic ChainF introduced in Jovanovic and Voß (2014).
int ChainF(//INPUT
	       int n,           
	       int w,           
	       int h,
	       int blNo, 
	       int minItem,     // minimum item in the yard
	       int **yard,
	       int *minstack,
	       int *freeslot,
	       int **map,
	       int BestUB,       
	       int *dstack  //to compute final solution
	       )
{
  int i, j, rit, stack, slot, t;
  int UBT;      //value of the solution
  int bbs, bns; //bbs = best blocking stack, bns = best non blocking stack
  int bbv, bnv; //bbv = best blocking stack UB, bnv = best non blocking stack UB
  int bs;   // bv = best UB value, bs = best UB stack
  int bbv2,bbv3,bbs2,bbs3;
  int nextrit;
  int bnsnx; //bbs = best blocking stack, bns = best non blocking stack
  int bnvnx; //bbv = best blocking stack UB, bnv = best non blocking stack UB
  int bsnx;   // bv = best UB value, bs = best UB stack
  int bbsnx,bbvnx,bbv2nx,bbv3nx,bbs2nx,bbs3nx,nextstack;

  UBT = 0;
  
  while(minItem <= n && blNo > 0){
    
    if(UBT + blNo >= BestUB)    
      return (n * n);
    
     
    stack = map[minItem-1][0];//saves stack and slot of the next item t to be retrieved, removes t and updates configuration
    slot = map[minItem-1][1];

    for (t = h - freeslot[stack] - 1; t > slot; --t) {
      rit = yard[stack][t];
      nextrit=-1;
      if(t>slot+1){
	nextrit=yard[stack][t-1];
	nextstack=stack;
      }
      
      bbv = bbv2 = -1;
      bnv = bbv3 = n * n;
      bbs = bns = bbs2 = bbs3 = -1;
      bbvnx = bbv2nx = -1;
      bnvnx = bbv3nx = n * n;
      bbsnx = bnsnx = bbs2nx = bbs3nx = -1;
      yard[stack][t] = -1;	
      freeslot[stack]++;
      for(j = 0; j < w; j++) {
	if((j != stack) && (freeslot[j])) {
	  if(minstack[j] < rit) {
	    if (minstack[j] > bbv) {
	      bbv = minstack[j];
	      bbs = j;
	    }
	    if (freeslot[j]>1 && minstack[j] > bbv2) {
	      bbv2 = minstack[j];
	      bbs2 = j;
	    }
	    if (minstack[j] < bbv3) {
	      bbv3 = minstack[j];
	      bbs3 = j;
	    }
	  }
	  else {
	    if (minstack[j] < bnv) {
	      bnv = minstack[j];
	      bns = j;
	    }
	  }
	  if(nextrit!=-1 && nextrit>rit && j!=nextstack){
	    if(minstack[j] < nextrit) {
	      if (minstack[j] > bbvnx) {
		bbvnx = minstack[j];
		bbsnx = j;
	      }
	      if (freeslot[j]>1 && minstack[j] > bbv2nx) {
		bbv2nx = minstack[j];
		bbs2nx = j;
	      }
	      if (minstack[j] < bbv3nx) {
		bbv3nx = minstack[j];
		bbs3nx = j;
	      }
	    }
	    else {
	      if (minstack[j] < bnvnx) {
		bnvnx = minstack[j];
		bnsnx = j;
	      }
	    }
	  }
	}
      }
      if (bns != -1) {
	bs = bns;

	if(nextrit!=-1 && nextrit>rit){
	  if (bnsnx != -1) {
	    bsnx = bnsnx;
	    if(bs==bsnx){
	      bbv = bbv2 = -1;
	      bnv = bbv3 = n * n;
	      bbs = bns = bbs2 = bbs3 = -1;
	      for(j = 0; j < w; j++) {
		if((j != stack) && (j != bsnx) && (freeslot[j])) {
		  if(minstack[j] < rit) {
		    if (minstack[j] > bbv) {
		      bbv = minstack[j];
		      bbs = j;
		    }
		    if (freeslot[j]>1 && minstack[j] > bbv2) {
		      bbv2 = minstack[j];
		      bbs2 = j;
		    }
		    if (minstack[j] < bbv3) {
		      bbv3 = minstack[j];
		      bbs3 = j;
		    }
		  }
		  else {
		    if (minstack[j] < bnv) {
		      bnv = minstack[j];
		      bns = j;
		    }
		  }
		}
	      }
	      if (bns != -1) {
		bs = bns;
	      }else{
		if(bbs!=-1){
		  bs = bbs;
		  if(freeslot[bs]==1 && bbs2!=-1)
		    bs = bbs2;
		  if(freeslot[bs]==1)
		    bs = bbs3;
		}
	      }
	    }
	  }

	}

	if(minstack[bs] > rit){
	  minstack[bs] = rit;
	  blNo--;
	}
      }
      else{
	bs = bbs;
	if(freeslot[bs]==1 && bbs2!=-1)
	  bs = bbs2;
	if(freeslot[bs]==1)
	  bs = bbs3;
	

      }

    
      UBT++;
      if((*dstack)==-1)
	(*dstack) = bs;
      
      yard[bs][h-freeslot[bs]] = rit;
      map[rit-1][0] = bs;
      map[rit-1][1] = h - freeslot[bs];
      freeslot[bs] --;
    }

    yard[stack][slot] = -1;
    freeslot[stack]++;
    minstack[stack] = n + 1;
    for(i = 0; i < slot; i++)
      if(yard[stack][i] < minstack[stack])
	minstack[stack] = yard[stack][i];

    minItem++;
  }
 
  return UBT;
}

//Lower bound UBALB introduced in Bacci et al. (2019)
int UBALB(
		  int n,
		  int w,
		  int h,
		  int minItem,
		  int **yard,
		  int *minstack,
		  int *freeslot,
		  int **map,
		  int *phi,
		  int *minst,
		  int *freesl,
		  int *L,//for merge sort
		  int *R//for merge sort
		  )
{
  int i, j, t, stack, slot, tfree, nblocks,emptyStacks=0,checkFirst=-1;
  int LBT;//value of the lower bound
  int NotFullHeavyStack=0,idmaxminst;
  int FullStack = 0;
 

  //*******************
  //At the beginning the algorithm copies yardIn in yard and  searches for the first item to be retrieved in yard (which has the minimum index)
  LBT = 0;
  int maxminstack = n+1;
  


  while(minItem <= n && emptyStacks<h){
    if (map[minItem-1][0] != -1) {

      stack = map[minItem-1][0];
      slot = map[minItem-1][1];
      tfree=freeslot[stack];
      nblocks=0;

      for (t = h - tfree - 1; t > slot; --t) {
	if(yard[stack][t]<maxminstack){
	  phi[nblocks]=yard[stack][t];
	  nblocks++;
	}else
	  LBT++;
	map[yard[stack][t]-1][0]=-1;
	yard[stack][t] = -1;
	freeslot[stack]++;
      }

      if(nblocks > 0 && emptyStacks < nblocks){
	
	mergeSort(phi, 0, nblocks-1, L, R);
	
	if(checkFirst==-1){
	  maxminstack = -1;
	  FullStack = 0;
	  emptyStacks = 0;
	  for(j=0;j<w;j++){
	    if((j==stack && tfree>0) || (j!=stack && freeslot[j] > 0)){
	      minst[FullStack] = j;
	      FullStack++;
	      if(j!=stack && (minstack[j] > maxminstack)){
		maxminstack = minstack[j];
		idmaxminst = j;
	      }

	    }
	    if(j!=stack && freeslot[j] == h)
	      emptyStacks++;
	    freesl[j] = freeslot[j];
	  }
	  checkFirst=1;


	}

	if(freeslot[idmaxminst]<nblocks  || phi[0]>maxminstack){
	 
	  mergeSortCrit1(minstack, 0, FullStack-1, L, R, minst);

	  NotFullHeavyStack = 0;
	  if(minst[NotFullHeavyStack]==stack)
	    NotFullHeavyStack++;


	  for(i=0; i<nblocks; i++){
	  
	    if(phi[i]<=minstack[minst[NotFullHeavyStack]]){
	      freesl[minst[NotFullHeavyStack]]--;
	      if(freesl[minst[NotFullHeavyStack]]==0 && i!=(nblocks-1)){
		NotFullHeavyStack++;
		if(minst[NotFullHeavyStack]==stack)
		  NotFullHeavyStack++;
	      }
	    }else
	      LBT++;
	  }
	  for(j=0;j<NotFullHeavyStack+1;j++)
	    freesl[minst[j]] = freeslot[minst[j]];
	}
      }
 
      map[yard[stack][slot]-1][0]=-1;
      yard[stack][slot] = -1;
      freeslot[stack]++;
      if(freeslot[stack] == h)
	emptyStacks++;
      minstack[stack] = n + 1;
      for(i = 0; i < slot; i++)
	if(yard[stack][i] < minstack[stack])
	  minstack[stack] = yard[stack][i];
      if(minstack[stack]>maxminstack){
	maxminstack = minstack[stack];
	idmaxminst=stack;
      }
      if(checkFirst==1 && tfree == 0){

	minst[FullStack] = stack;
	FullStack++;
       	
      }
      
      freesl[stack]=freeslot[stack];
 
    }
    minItem++;
  }
  
 

  return LBT;

}

//Function for removing unblocked blocks
int removeItems(
		int n,
		int w,
		int h,
		int minItem, 
		int **yard,
		int *minstack,
		int *freeslot,
		int **map,
		int *item         // next item to be reshuffled
		)
  
{
  int k, check = 0;
  int stack, slot;
  
  while((check==0) && (minItem <= n)) {
    stack = map[minItem-1][0];
    slot = map[minItem-1][1];
    if ((slot == h - 1) || (yard[stack][slot+1] < 0)) {
      yard[stack][slot] = -1;
      freeslot[stack] ++;
      minstack[stack] = n + 1;
      for (k = 0; k < slot; ++k)
	if (yard[stack][k] < minstack[stack])
	  minstack[stack] = yard[stack][k];
      minItem ++;
    }
    else {
      check = 1;
      *item = yard[stack][h-freeslot[stack]-1];
    }
  }
  return (minItem);
}


int copyYard (int n, int w, int h, int **yard, int *minstack, int *freeslot, int **map, int **tyard, int *tminstack, int *tfreeslot, int **tmap, int stack) {
  int i, j;
  for (i = 0; i < w; ++i) {
    tminstack[i] = minstack[i];
    tfreeslot[i] = freeslot[i];
    for (j = 0; j < h; ++j) {
      tyard[i][j] = yard[i][j];
      if (yard[i][j] != -1) {
	tmap[yard[i][j]-1][0] = i;
	tmap[yard[i][j]-1][1] = j;
      }
    }
  }	
  return (0);
}


void merge(int *arr, int l, int m, int r, int *L, int *R)
{
  int i, j, k;
  int n1 = m - l + 1;
  int n2 =  r - m;
  /* create temp arrays */
  //int L[n1], R[n2];
 
  /* Copy data to temp arrays L[] and R[] */
  for (i = 0; i < n1; i++)
    L[i] = arr[l + i];
  for (j = 0; j < n2; j++)
    R[j] = arr[m + 1+ j];
 
  /* Merge the temp arrays back into arr[l..r]*/
  i = 0; // Initial index of first subarray
  j = 0; // Initial index of second subarray
  k = l; // Initial index of merged subarray
  while (i < n1 && j < n2)
    {
      if (L[i] > R[j])
        {
	  arr[k] = L[i];
	  i++;
        }
      else
        {
	  arr[k] = R[j];
	  j++;
        }
      k++;
    }
 
  /* Copy the remaining elements of L[], if there
     are any */
  while (i < n1)
    {
      arr[k] = L[i];
      i++;
      k++;
    }
 
  /* Copy the remaining elements of R[], if there
     are any */
  while (j < n2)
    {
      arr[k] = R[j];
      j++;
      k++;
    }
}
 
/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
void mergeSort(int *arr, int l, int r, int *L, int *R)
{
  if (l < r)
    {
      // Same as (l+r)/2, but avoids overflow for
      // large l and h
      int m = l+(r-l)/2;
 
      // Sort first and second halves
      mergeSort(arr, l, m, L, R);
      mergeSort(arr, m+1, r, L, R);
 
      merge(arr, l, m, r, L, R);
    }
}

void mergeCrit1(int *arr, int l, int m, int r, int *L, int *R, int *id)
{
  int i, j, k;
  int n1 = m - l + 1;
  int n2 =  r - m;
 
  /* create temp arrays */
  //int L[n1], R[n2];
  /* Copy data to temp arrays L[] and R[] */
  for (i = 0; i < n1; i++)
    L[i] = id[l + i];
  for (j = 0; j < n2; j++)
    R[j] = id[m + 1+ j];
 
  /* Merge the temp arrays back into arr[l..r]*/
  i = 0; // Initial index of first subarray
  j = 0; // Initial index of second subarray
  k = l; // Initial index of merged subarray
  while (i < n1 && j < n2)
    {
      if (arr[L[i]] > arr[R[j]])
        {
	  id[k] = L[i];
	  i++;
        }
      else
        {
	  id[k] = R[j];
	  j++;
        }
      k++;
    }
 
  /* Copy the remaining elements of L[], if there
     are any */
  while (i < n1)
    {
      id[k] = L[i];
      i++;
      k++;
    }
 
  /* Copy the remaining elements of R[], if there
     are any */
  while (j < n2)
    {
      id[k] = R[j];
      j++;
      k++;
    }
}
 
/* l is for left index and r is right index of the
   sub-array of arr to be sorted */
void mergeSortCrit1(int *arr, int l, int r, int *L, int *R, int *id)
{
  if (l < r)
    {
      // Same as (l+r)/2, but avoids overflow for
      // large l and h
      int m = l+(r-l)/2;
 
      // Sort first and second halves
      mergeSortCrit1(arr, l, m, L, R, id);
      mergeSortCrit1(arr, m+1, r, L, R, id);
 
      mergeCrit1(arr, l, m, r, L, R, id);
    }
}



void constructSolution(int n, int w, int h, int **yard, int GUB, int bestSolCard, int *bestSol, nodebeam *VetBeam, int ***bbs_solution){
 
  int *minStack, *freeSlot, **map,**yardtmp,dstack;
  int *tminStack, *tfreeSlot, **tmap,**tyardtmp;
  int check, blNo, minItem, reshItem, oldStack, oldSlot; 
  int i, j,k, i1, slot, cont, k1;
  int **vet,*B,*idB;
  FILE *output;
  
  Alloc (minStack, w, int);
  Alloc (freeSlot, w, int);
  Alloc (map, n, int*);
  for (i = 0; i < n; ++i)
    Alloc (map[i], 2, int);
  Alloc (yardtmp, w, int*);
  for (i = 0; i < w; ++i)
    Alloc (yardtmp[i], h, int);

  Alloc (tminStack, w, int);
  Alloc (tfreeSlot, w, int);
  Alloc (tmap, n, int*);
  for (i = 0; i < n; ++i)
    Alloc (tmap[i], 2, int);
  Alloc (tyardtmp, w, int*);
  for (i = 0; i < w; ++i)
    Alloc (tyardtmp[i], h, int);

  Alloc(vet,h,int*);
  for(i=0;i<h;i++)
    Alloc(vet[i],5,int);
  Alloc(B,w,int);
  Alloc(idB,w,int);

  
  blNo = 0;
  minItem = 1;
  for(j = 0; j < w; j++) {
    minStack[j] = n + 1;
    freeSlot[j] = 0;
    for(k = 0 ; k < h; k++){
      yardtmp[j][k] = yard[j][k];
      if(yardtmp[j][k] == -1) 
	freeSlot[j] ++;
      else {
	map[yardtmp[j][k]-1][0] = j;
	map[yardtmp[j][k]-1][1] = k;
	if(yardtmp[j][k] < minStack[j])
	  minStack[j] = yardtmp[j][k];
	for (i = check = 0; ((i < k) && (!check)); ++i)
	  if (yardtmp[j][i] < yardtmp[j][k]) {
	    blNo++;
	    check = 1;
	  }
      }    
    }
  }
  for (i = 0; i < bestSolCard; ++i) {
    minItem = removeItems (n, w, h, minItem, yardtmp, minStack, freeSlot, map, &reshItem);
    oldStack = map[minItem-1][0];
    oldSlot = map[minItem-1][1];
    assert (map[reshItem-1][0] == oldStack);
    assert (map[reshItem-1][1] > oldSlot);
    assert ((map[reshItem-1][1] == h - 1) || (yardtmp[map[reshItem-1][0]][map[reshItem-1][1]+1] == -1));
    assert (freeSlot[bestSol[i]] > 0);
    yardtmp[map[reshItem-1][0]][map[reshItem-1][1]] = -1;
    freeSlot[oldStack] ++;
    yardtmp[bestSol[i]][h-freeSlot[bestSol[i]]] = reshItem;
    if (reshItem < minStack[bestSol[i]]) {
      minStack[bestSol[i]] = reshItem;
      blNo--;
    }
    map[reshItem-1][0] = bestSol[i];
    map[reshItem-1][1] = h-freeSlot[bestSol[i]];
    freeSlot[bestSol[i]] --;
    //printf ("****** Item %d is reshuffled to stack %d\n", reshItem, bestSol[i]+1);
  }
  while (bestSolCard < GUB) {
    dstack = -1;
    minItem = removeItems (n, w, h, minItem, yardtmp, minStack, freeSlot, map, &reshItem);
    oldStack = map[minItem-1][0];
    oldSlot = map[minItem-1][1];
    //printf ("oldStack = %d, oldSlot = %d\n", oldStack, oldSlot);
    assert (map[reshItem-1][0] == oldStack);
    assert (map[reshItem-1][1] > oldSlot);
    assert ((map[reshItem-1][1] == h - 1) || (yardtmp[map[reshItem-1][0]][map[reshItem-1][1]+1] == -1));
      
    copyYard (n, w, h, yardtmp, minStack, freeSlot, map, tyardtmp, tminStack, tfreeSlot, tmap,-1);
    if(n >= 1000 && n < 10000)
      VetBeam[0].UB = Difference1(n, w, h, blNo, minItem, tyardtmp, tminStack, tfreeSlot, tmap, n*n,&dstack);
    if(n >= 10000)
      VetBeam[0].UB = GAH(n, w, h, blNo, minItem, tyardtmp, tminStack, tfreeSlot, tmap, vet, B, idB, n*n,&dstack);
    if(n < 1000)
      VetBeam[0].UB = ChainF(n, w, h, blNo, minItem, tyardtmp, tminStack, tfreeSlot, tmap, n*n,&dstack);

    bestSol[bestSolCard] = dstack;
      
    assert (freeSlot[bestSol[bestSolCard]] > 0);
    yardtmp[map[reshItem-1][0]][map[reshItem-1][1]] = -1;
    freeSlot[oldStack] ++;
    yardtmp[bestSol[bestSolCard]][h-freeSlot[bestSol[bestSolCard]]] = reshItem;
    if (reshItem < minStack[bestSol[bestSolCard]]) {
      minStack[bestSol[bestSolCard]] = reshItem;
      blNo--;
    }
    map[reshItem-1][0] = bestSol[bestSolCard];
    map[reshItem-1][1] = h-freeSlot[bestSol[bestSolCard]];
    freeSlot[bestSol[bestSolCard]] --;
    //printf ("****** Item %d is reshuffled to stack %d\n", reshItem, bestSol[bestSolCard]+1);
    bestSolCard++;
  }
  for(j = 0; j < w; j++) 
    for(k = 0 ; k < h; k++)
      yardtmp[j][k] = yard[j][k];

  for(j = 0; j < w; j++) 
    for(k = 0 ; k < h; k++)
      bbs_solution[0][j][k] = yardtmp[j][k];

  cont = 0;
  for(i=0; i < n; i++){

    for(j = 0; j < w; j++) {
      for(k = 0 ; k < h; k++){
	if(yardtmp[j][k] == i+1){
	  slot = k;
	  yardtmp[j][k] = -1;
	  for(i1=h-1; i1 >slot ; i1--)
	    if(yardtmp[j][i1]>0){
	      k1 = 0;
	      while(yardtmp[bestSol[cont]][k1]>0)
		k1++;
	      yardtmp[bestSol[cont]][k1] = yardtmp[j][i1];
	      yardtmp[j][i1] = -1;
	      cont++;
	    }
		    
	  j = w+1;
	  k = h+1;
	}
      }
    }

    for(j = 0; j < w; j++) 
      for(k = 0 ; k < h; k++)
	bbs_solution[i+1][j][k] = yardtmp[j][k];
	  
	  
  }
  
  
  free (minStack);
  free (freeSlot);
  for (i = 0; i < n; ++i)
    free (map[i]);
  free (map);
  for (i = 0; i < w; ++i)
    free (yardtmp[i]);
  free (yardtmp);

  free (tminStack);
  free (tfreeSlot);
  for (i = 0; i < n; ++i)
    free (tmap[i]);
  free (tmap);
  for (i = 0; i < w; ++i)
    free (tyardtmp[i]);
  free (tyardtmp);

  for(i=0;i<h;i++)
    free(vet[i]);
  free(vet);
  free(B);
  free(idB); 

}
