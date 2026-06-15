/*
 * Copyright 2018-2019 Shunji Tanaka and Stefan Voss.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials
 *      provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  $Id: solve.c,v 1.7 2019/06/11 12:13:41 tanaka Exp tanaka $
 *  $Revision: 1.7 $
 *  $Date: 2019/06/11 12:13:41 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "define.h"
#include "heuristics.h"
#include "print.h"
#include "problem.h"
#include "solution.h"
#include "solve.h"

/* Lower bound: */
/*   1: 2-blocking */
/*   2: 2-blocking + 4-blocking */
/*   3: relaxation/bb (lower bound: LB2) */
#ifndef LOWER_BOUND
/* #define LOWER_BOUND (1) */
/* #define LOWER_BOUND (2) */
/* #define LOWER_BOUND (3) */
#define LOWER_BOUND (3)
#endif /* !LOWER_BOUND */
#ifndef PURE_BB
/* #define PURE_BB */
#endif /* !PURE_BB */
#ifndef HEURISTIC
#ifdef PURE_BB
#define HEURISTIC (2)
#else /* !PURE_BB */
#define HEURISTIC (1)
#endif /* !PURE_BB */
#endif /* !HEURISTIC */
#ifndef LB_BB_DOMINANCE
#define LB_BB_DOMINANCE
#endif /* !LB_BB_DOMINANCE */
#ifndef LOWER_BOUND_COMPUTATION_TIME_CHECK
/* #define LOWER_BOUND_COMPUTATION_TIME_CHECK */
#endif /* !LOWER_BOUND_COMPUTATION_TIME_CHECK */


typedef struct {
  int src_stack;
  int dst_stack;
  int index;
  int ship_n_block;
  uchar blocking;
  int n_blocking;
#if 1
  int score;
#endif
  int lb;
} child_node_t;

typedef struct {
  state_t *state;
  int **ship_n_tier;
  yard_stack_state_t **ystate;
  uchar ***blocking4_matrix;
  child_node_t *child_node;
} bbwork_t;

static bbwork_t *bbwork;
static solution_t *partial_solution;
static int *vposition_list;

#if LOWER_BOUND == 2
static int *lb_blocking4_bucket;
#elif LOWER_BOUND == 3
static int **lb_ship_n_tier;
static yard_stack_state_t **lb_ystate;
static int **lb_ship_block;
static int **lb_n_blocking4;
static int **lb_blocking4_bucket;
#ifdef LB_BB_DOMINANCE
static uchar **lb_dominated;
#endif /* LB_BB_DOMINANCE */
#endif /* LOWER_BOUND == 3 */
static int initial_lb;

#ifndef RESTRICTED
#ifdef DOMINANCE_CHECK
static int *dominance_check;
static int dominance_table[4][4] =
  { { 0, 1, 0, 0 },
    { 1, 1, 2, 1 },
    { 0, 2, 0, 0 },
    { 0, 1, 0, 0 } };

static struct {
  int src;
  int dst;
} preloc[3];
#endif /* DOMINANCE_CHECK */
#endif /* !RESTRICTED */

static ulint n_node;
static uint count;
#if LOWER_BOUND == 3
#ifdef LOWER_BOUND_COMPUTATION_TIME_CHECK
static uint lb_count;
#endif /* LOWER_BOUND_COMPUTATION_TIME_CHECK */
#endif /* LOWER_BOUND == 3 */

#if LOWER_BOUND >= 2
static void initialize_work_memory_for_lb(problem_t *, int);
static void free_work_memory_for_lb(void);
#endif /* LOWER_BOUND >= 2 */
static void initialize_work_memory_for_bb(problem_t *, int);
static void free_work_memory_for_bb(void);

#ifdef RESTRICTED
#ifdef PURE_BB
static uchar bb(problem_t *, state_t *state, solution_t *, int, int, int);
#else /* !PURE_BB */
static uchar bb(problem_t *, state_t *state, solution_t *, int *, int, int);
#endif /* !PURE_BB */
#else /* !RESTRICTED */
#ifdef PURE_BB
static uchar bb(problem_t *, state_t *state, solution_t *, int, int);
#else /* !PURE_BB */
static uchar bb(problem_t *, state_t *state, solution_t *, int *, int);
#endif /* !PURE_BB */
#endif /* !RESTRICTED */

#if LOWER_BOUND == 2
static int lower_bound2(problem_t *, state_t *);
#elif LOWER_BOUND == 3
#ifdef RESTRICTED
static int lower_bound3(problem_t *, state_t *, int, int);
#else /* !RESTRICTED */
static int lower_bound3(problem_t *, state_t *, int);
#endif /* !RESTRICTED */
#endif /* LOWER_BOUND == 3 */

uchar solve(problem_t *problem, solution_t *solution)
{
  uchar ret;
  state_t *state = initialize_state(problem, NULL);

  move_all_blocks(problem, state);

  /* initial upper bound */
#ifdef RESTRICTED
  heuristics(problem, state, solution, -1, 1<<20);
#else /* !RESTRICTED */
  heuristics(problem, state, solution, 1<<20);
#endif /* !RESTRICTED */

#if LOWER_BOUND >= 2
  initialize_work_memory_for_lb(problem, solution->n_relocation + 1);
#endif /* LOWER_BOUND >= 2 */

  if(state->ship_n_block < problem->ship_n_block) {
#if LOWER_BOUND == 1
    initial_lb = state->n_blocking;
#elif LOWER_BOUND == 2
    initial_lb = lower_bound2(problem, state);
#elif LOWER_BOUND == 3
#ifdef RESTRICTED
    initial_lb = lower_bound3(problem, state, -1, solution->n_relocation);
#else /* !RESTRICTED */
    initial_lb = lower_bound3(problem, state, solution->n_relocation);
#endif /* !RESTRICTED */
#endif /* LOWER_BOUND == 3 */

    fprintf(stderr, "initial lb=%d ub=%d\n", initial_lb,
	    solution->n_relocation);
  } else {
    initial_lb = 0;
    fprintf(stderr, "initial lb=%d ub=%d\n", initial_lb,
	    solution->n_relocation);
    fprintf(stderr, "Trivial optimal solution (0 relocation).\n");
  }

  if(lower_bound_only == TRUE) {
#if LOWER_BOUND >= 2
    free_work_memory_for_lb();
#endif /* LOWER_BOUND >= 2 */
    free_state(state);
    return(FALSE);
  }

  n_node = 1;
  count = 0;
  ret = TRUE;

  if(initial_lb == -1) {
    ret = TLIMIT;
  } else if(initial_lb == solution->n_relocation) {
    fprintf(stderr, "Initial upper bound is optimal.\n");
  } else {
#ifdef PURE_BB
    initialize_work_memory_for_bb(problem, solution->n_relocation + 1);
#ifdef RESTRICTED
    ret = bb(problem, state, solution, initial_lb, 1, -1);
#else /* !RESTRICTED */
    ret = bb(problem, state, solution, initial_lb, 1);
#endif /* !RESTRICTED */
#else /* !PURE_BB */
    int ub = initial_lb;

    initialize_work_memory_for_bb(problem, solution->n_relocation + 1);

    /* main loop */
    for(; ub < solution->n_relocation; ++ub) {
      fprintf(stderr, "cub=%d ", ub);
      print_time(problem);
#ifdef RESTRICTED
      if((ret = bb(problem, state, solution, &ub, 1, -1)) == TLIMIT) {
	break;
      }
#else /* !RESTRICTED */
      if((ret = bb(problem, state, solution, &ub, 1)) == TLIMIT) {
	break;
      }
#endif /* !RESTRICTED */
      initial_lb = 0;
    }
#endif /* !PURE_BB */
    free_work_memory_for_bb();
  }

#if LOWER_BOUND >= 2
  free_work_memory_for_lb();
#endif /* LOWER_BOUND >= 2 */
  free_state(state);

  fprintf(stderr, "nodes=%llu\n", n_node);

  return((ret == TLIMIT)?FALSE:TRUE);
}

#if LOWER_BOUND >= 2
void initialize_work_memory_for_lb(problem_t *problem, int max_depth)
{
#if LOWER_BOUND == 2
  lb_blocking4_bucket
    = (int *) malloc((size_t) problem->yard_n_block*sizeof(int));
#else /* LOWER_BOUND == 3 */
  int i, j;

  int max_lb_depth = problem->yard_n_block/2 + 1;
  int lb_work_size = problem->ship_n_stack + 2*problem->yard_n_block;

  lb_ship_n_tier = (int **) malloc((size_t) 3*max_lb_depth*sizeof(int *));
  lb_n_blocking4 = lb_ship_n_tier + max_lb_depth;
  lb_ship_block = lb_ship_n_tier + 2*max_lb_depth;
  lb_ship_n_tier[0] = (int *) malloc((size_t) max_lb_depth*lb_work_size
				     *sizeof(int));
  lb_n_blocking4[0] = lb_ship_n_tier[0] + problem->ship_n_stack;
  lb_ship_block[0] =  lb_n_blocking4[0] + problem->yard_n_block;
  lb_ystate = (yard_stack_state_t **) malloc((size_t) max_lb_depth
					     *sizeof(yard_stack_state_t *));
  lb_ystate[0] = (yard_stack_state_t *) malloc((size_t) max_lb_depth
					       *problem->yard_n_stack
					       *sizeof(yard_stack_state_t));

#ifdef LB_BB_DOMINANCE
  lb_dominated = (uchar **) malloc((size_t) max_lb_depth*sizeof(uchar *));
  lb_dominated[0] = (uchar *) malloc((size_t) max_lb_depth
				     *problem->yard_n_stack);
#endif /* LB_BB_DOMINANCE */

  lb_blocking4_bucket = (int **) malloc((size_t) max_lb_depth
					*sizeof(int *));
  lb_blocking4_bucket[0] = (int *) malloc((size_t) max_lb_depth
					  *problem->yard_n_block*sizeof(int));
  for(i = 1; i < max_lb_depth; ++i) {
    lb_ship_n_tier[i] = lb_ship_n_tier[i - 1] + lb_work_size;
    lb_ship_block[i] = lb_ship_block[i - 1] + lb_work_size;
    lb_ystate[i] = lb_ystate[i - 1] + problem->yard_n_stack;
#ifdef LB_BB_DOMINANCE
    lb_dominated[i] = lb_dominated[i - 1] + problem->yard_n_stack;
#endif /* LB_BB_DOMINANCE */
    lb_n_blocking4[i] = lb_n_blocking4[i - 1] + lb_work_size;
    lb_blocking4_bucket[i]
      = lb_blocking4_bucket[i - 1] + problem->yard_n_block;
  }

  for(i = 0; i < problem->ship_n_stack; ++i) {
    for(j = 0; j < problem->ship_n_tier[i]; ++j) {
      lb_ship_block[0][problem->ship_block[i][j]] = j + 1;
    }
  }
#endif /* LOWER_BOUND == 3 */
}

void free_work_memory_for_lb(void)
{
#if LOWER_BOUND == 2
  free(lb_blocking4_bucket);
#else /* LOWER_BOUND == 3 */
  free(lb_blocking4_bucket[0]);
  free(lb_blocking4_bucket);
#ifdef LB_BB_DOMINANCE
  free(lb_dominated[0]);
  free(lb_dominated);
#endif /* LB_BB_DOMINANCE */
  free(lb_ystate[0]);
  free(lb_ystate);
  free(lb_ship_n_tier[0]);
  free(lb_ship_n_tier);
#endif /* LOWER_BOUND == 3 */
}
#endif /* LOWER_BOUND >= 2 */

void initialize_work_memory_for_bb(problem_t *problem, int max_depth)
{
  int i, j, k;
  int n = problem->ship_n_stack + problem->yard_n_block;
  int max_n_child = problem->yard_n_stack*(problem->yard_n_stack - 1) + 1;

  bbwork = (bbwork_t *) malloc((size_t) max_depth*sizeof(bbwork_t));
  bbwork[0].ystate
    = (yard_stack_state_t **) malloc((size_t) max_depth*max_n_child
				     *sizeof(yard_stack_state_t *));
  bbwork[0].ystate[0]
    = (yard_stack_state_t *) malloc((size_t) max_depth*max_n_child
				    *problem->yard_n_stack
				    *sizeof(yard_stack_state_t));

  bbwork[0].ship_n_tier = (int **) malloc((size_t) max_depth*max_n_child
					  *sizeof(int *));
  bbwork[0].ship_n_tier[0] = (int *) malloc((size_t) max_depth*max_n_child*n
					    *sizeof(int));

  bbwork[0].blocking4_matrix = (uchar ***) malloc((size_t) max_depth
						  *max_n_child
						  *sizeof(uchar **));
  bbwork[0].blocking4_matrix[0] = (uchar **) malloc((size_t) max_depth
						    *max_n_child
						    *problem->yard_n_block
						    *sizeof(uchar *));
  bbwork[0].blocking4_matrix[0][0] = (uchar *) malloc((size_t) max_depth
						      *max_n_child
						      *problem->yard_n_block
						      *problem->yard_n_block);

  bbwork[0].child_node = (child_node_t *) malloc((size_t) max_depth
						 *max_n_child
						 *sizeof(child_node_t));

  for(i = 0; i < max_depth; ++i) {
    if(i > 0) {
      bbwork[i].ystate = bbwork[i - 1].ystate + max_n_child;
      bbwork[i].ystate[0]
	= bbwork[i - 1].ystate[0] + max_n_child*problem->yard_n_stack;

      bbwork[i].ship_n_tier = bbwork[i - 1].ship_n_tier + max_n_child;
      bbwork[i].ship_n_tier[0] = bbwork[i - 1].ship_n_tier[0] + max_n_child*n;

      bbwork[i].blocking4_matrix
	= bbwork[i - 1].blocking4_matrix + max_n_child;
      bbwork[i].blocking4_matrix[0]
	= bbwork[i - 1].blocking4_matrix[0] + max_n_child*problem->yard_n_block;
      bbwork[i].blocking4_matrix[0][0]
	= bbwork[i - 1].blocking4_matrix[0][0]
	+ max_n_child*problem->yard_n_block*problem->yard_n_block;

      bbwork[i].child_node = bbwork[i - 1].child_node + max_n_child;

      for(j = 0; j < max_n_child; ++j) {
	if(j > 0) {
	  bbwork[i].ystate[j] = bbwork[i].ystate[j - 1] + problem->yard_n_stack;
	  bbwork[i].ship_n_tier[j] = bbwork[i].ship_n_tier[j - 1] + n;
	  bbwork[i].blocking4_matrix[j]
	    = bbwork[i].blocking4_matrix[j - 1] + problem->yard_n_block;
	  bbwork[i].blocking4_matrix[j][0]
	    = bbwork[i].blocking4_matrix[j - 1][0]
	    + problem->yard_n_block*problem->yard_n_block;
	}

	for(k = 1; k < problem->yard_n_block; ++k) {
	  bbwork[i].blocking4_matrix[j][k]
	    = bbwork[i].blocking4_matrix[j][k - 1] + problem->yard_n_block;
	}
      }
    }
  }

  vposition_list = (int *) malloc((size_t) problem->yard_n_stack*sizeof(int));

#ifdef DOMINANCE_CHECK
#ifndef RESTRICTED
  dominance_check = (int *) malloc((size_t) problem->yard_n_stack*sizeof(int));
#endif /* RESTRICTED */
#endif /* DOMINANCE_CHECK */
  partial_solution = create_solution();
}

void free_work_memory_for_bb(void)
{
  free_solution(partial_solution);
#ifdef DOMINANCE_CHECK
#ifndef RESTRICTED
  free(dominance_check);
#endif /* RESTRICTED */
#endif /* DOMINANCE_CHECK */
  free(vposition_list);
  free(bbwork[0].child_node);
  free(bbwork[0].blocking4_matrix[0][0]);
  free(bbwork[0].blocking4_matrix[0]);
  free(bbwork[0].blocking4_matrix);
  free(bbwork[0].ship_n_tier[0]);
  free(bbwork[0].ship_n_tier);
  free(bbwork[0].ystate[0]);
  free(bbwork[0].ystate);
  free(bbwork);
}

#ifdef RESTRICTED
#ifdef PURE_BB
uchar bb(problem_t *problem, state_t *state, solution_t *solution, int plb,
	 int depth, int target_stack)
#else /* !PURE_BB */
uchar bb(problem_t *problem, state_t *state, solution_t *solution, int *ub,
	 int depth, int target_stack)
#endif /* !PURE_BB */
#else /* !RESTRICTED */
#ifdef PURE_BB
uchar bb(problem_t *problem, state_t *state, solution_t *solution, int plb,
	 int depth)
#else /* !PURE_BB */
uchar bb(problem_t *problem, state_t *state, solution_t *solution, int *ub,
	 int depth)
#endif /* !PURE_BB */
#endif /* !RESTRICTED */
{
  int i, j, k, l;
  int max_n_child = problem->yard_n_stack*(problem->yard_n_stack - 1) + 1;
  int n_child = 0;
  int ship_n_block = state->ship_n_block;
  int ship_stack;
  int n_blocking = state->n_blocking;
  uchar blocking;
  uchar ret, empty_flag;
  block_t rblock, dblock;
#ifdef DOMINANCE_CHECK
  int last_reloc;
#ifdef RESTRICTED
  int last_target_depth;
#else /* !RESTRICTED */
  int *pdominance_table;
#endif /* !RESTRICTED */
#endif /* DOMINANCE_CHECK */
  yard_stack_state_t *ystate = state->ystate, *nystate;
  block_t **yard_block = state->yard_block;
  coordinate_t yard_position_backup, *yard_position = state->yard_position;
  int *ship_n_tier = state->ship_n_tier, *nship_n_tier;
  int *n_blocking4 = state->n_blocking4, *nn_blocking4;
  uchar **blocking4_matrix = state->blocking4_matrix, **nblocking4_matrix;
  child_node_t *cn = bbwork[depth].child_node;
  
#ifdef PURE_BB
  if(depth > solution->n_relocation) {
    return(FALSE);
  }
#else /* !PURE_BB */
  if(depth > *ub) {
    return(FALSE);
  }
#endif /* !PURE_BB */

#if LOWER_BOUND == 3
  if(tlimit > 0 && ++count == 2000) {
    count = 0;
    if(get_time(problem) >= (double) tlimit) {
      return(TLIMIT);
    }
  }
#else /* LOWER_BOUND <= 2 */
  if(tlimit > 0 && ++count == 200000) {
    count = 0;
    if(get_time(problem) >= (double) tlimit) {
      return(TLIMIT);
    }
  }
#endif /* LOWER_BOUND <= 2 */

#if 0
  printf("depth=%d\n", depth);
  print_state(problem, state, stdout);
  for(i = 0; i < problem->yard_n_stack; ++i) {
    printf("[%d,%d]", state->ystate[i].n_target, state->ystate[i].n_stacked);
  }
  printf("\n");
#ifdef DOMINANCE_CHECK
  for(i = 0; i < problem->yard_n_stack; ++i) {
    printf("<%d,%d>", state->ystate[i].last_modified/MAX_N_RELOCATION,
	   state->ystate[i].last_modified%MAX_N_RELOCATION);
  }
  printf("\n");
  for(i = 0; i < partial_solution->n_target; ++i) {
    printf("%d:", partial_solution->target[i]);
  }
  printf("\n");

#endif /* DOMINANCE_CHECK */
  printf("n_blocking=%d\n", state->n_blocking);

  print_solution_relocation(problem, partial_solution, stdout);
#endif

#ifdef DOMINANCE_CHECK
#ifdef RESTRICTED
  if(target_stack == -1) {
    add_target(partial_solution, depth);
  }
#else /* !RESTRICTED */
  memset((void *) dominance_check, 0, problem->yard_n_stack*sizeof(int));
  preloc[0].src = preloc[1].src = preloc[2].src = -1;

  if(depth >= 2) {
    int dst;

    for(i = 0; i < problem->yard_n_stack; ++i) {
      if(ystate[i].last_modified > MAX_N_RELOCATION) {
	last_reloc = ystate[i].last_modified - MAX_N_RELOCATION;
	dst = partial_solution->relocation[last_reloc - 1].dst;
	if(ystate[dst].last_modified == last_reloc) {
	  if(i > preloc[0].src) {
	    preloc[2] = preloc[1];
	    preloc[1] = preloc[0];
	    preloc[0].src = i; 
	    preloc[0].dst = dst;
	  } else if(i > preloc[1].src) {
	    preloc[2] = preloc[1];
	    preloc[1].src = i; 
	    preloc[1].dst = dst;
	  } else if(i > preloc[2].src) {
	    preloc[2].src = i;
	    preloc[2].dst = dst;
	  }
	}
      }
    }

    if(preloc[0].src >= 0) {
      dominance_check[preloc[0].src] = dominance_check[preloc[0].dst] = 1;
      if(preloc[1].src >= 0) {
	dominance_check[preloc[1].src] = dominance_check[preloc[1].dst] = 2;
	if(preloc[2].src >= 0) {
	  dominance_check[preloc[2].src] = dominance_check[preloc[2].dst] = 3;
	}
      }
    }
  }
#endif /* !RESTRICTED */
#endif /* DOMINANCE_CHECK */

  for(i = 0; i < problem->yard_n_stack; ++i) {
    yard_stack_state_t ystate_backup = ystate[i];

#ifdef RESTRICTED
    if(target_stack >= 0 && target_stack != i) {
      continue;
    } else if(ystate[i].n_stacked == problem->yard_s_height) {
      continue;
    }
#else /* !RESTRICTED */
    /* empty stack */
    if(ystate[i].n_tier == 0) {
      continue;
    }
#endif /* !RESTRICTED */

    /* rblock: the block to be relocated */
    rblock = yard_block[i][ystate[i].n_tier - 1];
    yard_position_backup = yard_position[rblock.no];

#ifdef DOMINANCE_CHECK
    last_reloc = state->last_relocation[rblock.no];
#endif /* DOMINANCE_CHECK */

#ifndef RESTRICTED
#ifdef DOMINANCE_CHECK
    if(last_reloc > 0) {
      if(ystate[partial_solution->relocation[last_reloc - 1].src].last_modified
	 == MAX_N_RELOCATION + last_reloc) {
	continue;
      }

      if(ystate[i].last_modified == last_reloc) {
	uchar flag = FALSE;

	for(j = 0; j < i; ++j) {
	  if(ystate[j].n_tier <= ystate[i].n_tier - 1
	     && (ystate[j].last_modified + MAX_N_RELOCATION)%MAX_N_RELOCATION
	     < last_reloc) {
	    flag = TRUE;
	    break;
	  }
	}
	if(flag == TRUE) {
	  continue;
	}

	for(j = i + 1; j < problem->yard_n_stack; ++j) {
	  if(ystate[j].n_tier < ystate[i].n_tier - 1
	     && (ystate[j].last_modified + MAX_N_RELOCATION)%MAX_N_RELOCATION
	     < last_reloc) {
	    flag = TRUE;
	    break;
	  }
	}
	if(flag == TRUE) {
	  continue;
	}
      } else {
	uchar flag = FALSE;

	for(j = 0; j < problem->yard_n_stack; ++j) {
	  if(ystate[j].n_tier < problem->yard_s_height
	     && (ystate[j].last_modified + MAX_N_RELOCATION)%MAX_N_RELOCATION
	     < last_reloc) {
	    flag = TRUE;
	    break;
	  }
	}

	if(flag == TRUE) {
	  continue;
	}
      }
    }

    pdominance_table = dominance_table[dominance_check[i]];
#endif /* DOMINANCE_CHECK */
#endif /* !RESTRICTED */

    if(ystate[i].n_stacked < problem->yard_s_height) {
      --ystate[i].n_stacked;
    }
    --ystate[i].n_tier;

    ship_stack = problem->ship_position[rblock.no].s;

    for(j = 0; j < problem->yard_n_stack;
	vposition_list[j++] = problem->yard_s_height);
    for(j = ship_n_tier[ship_stack]; j < problem->ship_position[rblock.no].t;
	++j) {
      k = problem->ship_block[ship_stack][j];
      vposition_list[yard_position[k].s]
	= min(vposition_list[yard_position[k].s], yard_position[k].t);
    }

#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
    last_target_depth = -1;
    if(ystate[i].n_stacked == 0) {
      for(j = partial_solution->n_target - 2; j >= 0; --j) {
	int src_stack
	  = partial_solution->relocation[partial_solution->target[j] - 1].src;
	if(src_stack == i) {
	  break;
	}
	if(src_stack > i) {
	  last_target_depth = partial_solution->target[j];
	  break;
	}
      }
#if 0
      printf("last_target_depth=%d (%d, %d)\n", last_target_depth,
	     ystate[i].last_modified%MAX_N_RELOCATION,
	     (int) ystate[i].last_modified/MAX_N_RELOCATION);
#endif
      if(last_target_depth <= ystate[i].last_modified%MAX_N_RELOCATION
	 || last_target_depth
	 <= (int) ystate[i].last_modified/MAX_N_RELOCATION) {
	last_target_depth = -1;
      }
    }
#if 0
    printf("last_target_depth=%d\n", last_target_depth);
#endif
    last_target_depth = max(last_target_depth, last_reloc);
#endif /* DOMINANCE_CHECK */
#endif /* RESTRICTED */

#if 0
    printf("src_stack=%d\n", i + 1);
#endif
    /* enumerate the candidates for the destination stack */
    empty_flag = FALSE;
    for(j = 0; j < problem->yard_n_stack; ++j) {
      int ship_stack;
      int lb;

      /* destination = source or no space in the destination stack */
      if(j == i || ystate[j].n_tier == problem->yard_s_height) {
	/* destination == source or no space */
	continue;
      }

      if(ystate[j].n_tier == 0) {
	if(empty_flag == TRUE) {
	  continue;
	}
	empty_flag = TRUE;
      }
#if 0
      printf("dst_stack=%d\n", j + 1);
#endif
#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
      if(ystate[j].last_modified%MAX_N_RELOCATION < last_target_depth) {
#if 0
	printf("dominated. %d=>%d (%d %d)\n", i + 1, j + 1, last_target_depth,
	       ystate[j].last_modified%MAX_N_RELOCATION);
#endif
	continue;
      }
#endif /* DOMINANCE_CHECK */
#else /* !RESTRICTED */
#ifdef DOMINANCE_CHECK
      if((ystate[j].last_modified + MAX_N_RELOCATION)%MAX_N_RELOCATION
	 < last_reloc) {
	continue;
      }
      if(i < preloc[pdominance_table[dominance_check[j]]].src) {
	continue;
      }
#endif /* DOMINANCE_CHECK */
#endif /* !RESTRICTED */

      ship_stack = problem->ship_position[rblock.no].s;
      blocking = 0;
      if(ship_stack < 0) {
	if(ystate[j].n_tier > 0) {
	  blocking = 1;
	}
      } else {
	for(k = 0; k < ystate[j].n_tier; ++k) {
	  int no = yard_block[j][k].no;
	  if(problem->ship_position[no].s == ship_stack
	     && problem->ship_position[no].t
	     < problem->ship_position[rblock.no].t) {
	    blocking = 1;
	    break;
	  }
	}
      }

      /* LB after relocation */
      state->n_blocking = n_blocking - rblock.blocking + blocking;
#if 0
      printf("%d->%d: blocking=%d (%d, %d, %d)\n", i, j,
	     state->n_blocking, n_blocking, rblock.blocking, blocking);
#endif

      ++n_node;

#if 0
      if(n_node > 1U<<30) {
	print_time(problem);
	exit(1);
      }
#endif

#ifdef PURE_BB
      if(depth + state->n_blocking >= solution->n_relocation) {
	continue;
      }
#else /* !PURE_BB */
      if(depth + state->n_blocking > *ub) {
	continue;
      }
#endif /* !PURE_BB */

      /* the block is relocated to the destination stack*/
      dblock = yard_block[j][ystate[j].n_tier];
      yard_block[j][ystate[j].n_tier].no = rblock.no;
      yard_block[j][ystate[j].n_tier].blocking = blocking;
      if(ystate[j].n_tier > 0) {
	yard_block[j][ystate[j].n_tier].n_blocking
	  = yard_block[j][ystate[j].n_tier - 1].n_blocking + blocking;
      }

      yard_position[rblock.no].s = j;
      yard_position[rblock.no].t = ystate[j].n_tier;

      /* create and copy the state for a child node */
      state->ship_n_block = ship_n_block;
      state->ystate = nystate = bbwork[depth].ystate[n_child];
      state->ship_n_tier = nship_n_tier = bbwork[depth].ship_n_tier[n_child];
      state->n_blocking4 = nn_blocking4
	= state->ship_n_tier + problem->ship_n_stack;
      state->blocking4_matrix = nblocking4_matrix
	= bbwork[depth].blocking4_matrix[n_child];

      memcpy((void *) nystate, (void *) ystate,
	     (size_t) problem->yard_n_stack*sizeof(yard_stack_state_t));
      memcpy((void *) nship_n_tier, (void *) ship_n_tier,
	     (size_t) (problem->ship_n_stack
		       + problem->yard_n_block)*sizeof(int));
      memcpy((void *) nblocking4_matrix[0], (void *) blocking4_matrix[0],
	     (size_t) problem->yard_n_block*problem->yard_n_block);

      ++nystate[j].n_tier;
      if(nystate[j].n_stacked < problem->yard_s_height) {
	++nystate[j].n_stacked;
      }

      if(nn_blocking4[rblock.no] > 0) {
	nn_blocking4[rblock.no] = 0;
	for(k = 0; k < problem->yard_n_block; ++k) {
	  nn_blocking4[k] -= nblocking4_matrix[rblock.no][k];
	  nblocking4_matrix[rblock.no][k] = nblocking4_matrix[k][rblock.no] = 0;
	}
      }

      if(!blocking) {
	for(k = nystate[j].n_tier - 1; k >= 0; --k) {
	  int ship_stack2 = problem->ship_position[yard_block[j][k].no].s;
	  if(ship_stack2 == ship_stack) {
	    continue;
	  }

	  for(l = problem->ship_position[yard_block[j][k].no].t + 1;
	      l < problem->ship_n_tier[ship_stack2]; ++l) {
	    int no = problem->ship_block[ship_stack2][l];
	    int yard_stack = yard_position[no].s;
	    if(yard_stack != i
	       && !yard_block[yard_position[no].s][yard_position[no].t].blocking
	       && yard_position[no].t > vposition_list[yard_stack]
	       && !nblocking4_matrix[rblock.no][no]) {
	      nblocking4_matrix[rblock.no][no]
		= nblocking4_matrix[no][rblock.no] = 1;
	      ++nn_blocking4[rblock.no];
	      ++nn_blocking4[no];
	    }
	  }
	}
      }

#ifdef DOMINANCE_CHECK
#ifdef RESTRICTED
      if(ystate[i].n_stacked > 0) {
	if((int) (nystate[i].last_modified/MAX_N_RELOCATION)
	   < nystate[j].last_modified%MAX_N_RELOCATION) {
	  nystate[i].last_modified
	    = (nystate[j].last_modified%MAX_N_RELOCATION)*MAX_N_RELOCATION
	    + nystate[i].last_modified%MAX_N_RELOCATION;
	}
      }
      nystate[j].last_modified = depth;
#else /* !RESTRICTED */
      nystate[i].last_modified = MAX_N_RELOCATION + depth;
      nystate[j].last_modified = depth;
#endif /* !RESTRICTED */
      state->last_relocation[rblock.no] = depth;
#endif /* DOMINANCE_CHECK */

      /* partial solution up to the current node */
      partial_solution->n_relocation = depth - 1;
      add_relocation(partial_solution, i, j, rblock.no);

      if(nystate[i].n_stacked == 0) {
	if(move_all_blocks2(problem, state, partial_solution) == TRUE) {
	  yard_block[j][ystate[j].n_tier] = dblock;
	  continue;
	}

#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
	nystate[i].last_modified = depth;
#endif /* DOMINANCE_CHECK */
#endif /* RESTRICTED */

	/* all the movable blocks are moved */
	if(state->ship_n_block == problem->ship_n_block) {
	  if(partial_solution->n_relocation < solution->n_relocation) {
	    /* better upper bound is found */
	    copy_solution(solution, partial_solution);
	    fprintf(stderr, "ub=%d ", solution->n_relocation);
	    print_time(problem);

#ifndef PURE_BB
	    if(solution->n_relocation <= *ub) {
	      /* When a solution as good as *ub is found, */
	      /* the search is terminated */

	      /* recover from the backup */
	      state->ystate = ystate;
	      state->ship_n_tier = ship_n_tier;
	      state->n_blocking4 = n_blocking4;
	      state->blocking4_matrix = blocking4_matrix;

	      return(TRUE);
	    }
#endif /* !PURE_BB */
	  }

	  yard_block[j][ystate[j].n_tier] = dblock;
	  continue;
	}
      }

#if HEURISTIC == 3
#ifdef RESTRICTED
      heuristics(problem, state, partial_solution,
		 (state->ship_n_block > ship_n_block)?(-1):i,
		 solution->n_relocation);
#else /* !RESTRICTED */
      heuristics(problem, state, partial_solution, solution->n_relocation);
#endif /* !RESTRICTED */

      if(partial_solution->n_relocation < solution->n_relocation) {
	/* better upper bound is found */
	copy_solution(solution, partial_solution);
	fprintf(stderr, "ub=%d depth=%d ", solution->n_relocation, depth);
	print_time(problem);

	if(solution->n_relocation <= *ub) {
	  /* When a solution as good as *ub is found, */
	  /* the search is terminated */

	  /* recover from the backup */
	  state->ystate = ystate;
	  state->ship_n_tier = ship_n_tier;
	  state->n_blocking4 = n_blocking4;
	  state->blocking4_matrix = blocking4_matrix;

	  return(TRUE);
	}
      }
#endif /* HEURISTIC == 3 */

#if LOWER_BOUND == 1
      lb = state->n_blocking;
#elif LOWER_BOUND == 2
      lb = lower_bound2(problem, state);
#elif LOWER_BOUND == 3
#ifdef PURE_BB
#ifdef RESTRICTED
      lb = lower_bound3(problem, state,
			(state->ship_n_block > ship_n_block)?(-1):i,
			min(solution->n_relocation - depth, plb + 1));
#else /* !RESTRICTED */
      lb = lower_bound3(problem, state,
			min(solution->n_relocation - depth, plb + 1));
#endif /* !RESTRICTED */
#else /* !PURE_BB */
#ifdef RESTRICTED
      lb = lower_bound3(problem, state,
			(state->ship_n_block > ship_n_block)?(-1):i,
			*ub - depth + 1);
#else /* !RESTRICTED */
      lb = lower_bound3(problem, state, *ub - depth + 1);
#endif /* !RESTRICTED */
#endif /* !PURE_BB */
#endif /* LOWER_BOUND == 3 */

#ifdef LOWER_BOUND_COMPUTATION_TIME_CHECK
      if(lb < 0) {
	ret = TLIMIT;
	goto bb_end;
      }
#endif /* LOWER_BOUND_COMPUTATION_TIME_CHECK */

#if LOWER_BOUND >= 2
#ifdef PURE_BB
      if(depth + lb >= solution->n_relocation) {
	yard_block[j][ystate[j].n_tier] = dblock;
	continue;
      }
#else /* !PURE_BB */
      if(depth + lb > *ub) {
	yard_block[j][ystate[j].n_tier] = dblock;
	continue;
      }
#endif /* !PURE_BB */
#endif /* LOWER_BOUND >= 2 */

#if HEURISTIC == 2
#ifdef RESTRICTED
      heuristics(problem, state, partial_solution,
		 (state->ship_n_block > ship_n_block)?(-1):i,
		 solution->n_relocation);
#else /* !RESTRICTED */
      heuristics(problem, state, partial_solution, solution->n_relocation);
#endif /* !RESTRICTED */

      if(partial_solution->n_relocation < solution->n_relocation) {
	/* better upper bound is found */
	copy_solution(solution, partial_solution);
	fprintf(stderr, "ub=%d depth=%d ", solution->n_relocation, depth);
	print_time(problem);
      }
#endif /* HEURISTIC == 2 */

#if HEURISTIC == 1
      if(lb + depth == *ub - 1 || lb + depth <= initial_lb) {
	/* upper bound computation */
#ifdef RESTRICTED
	heuristics(problem, state, partial_solution,
		   (state->ship_n_block > ship_n_block)?(-1):i,
		   solution->n_relocation);
#else /* !RESTRICTED */
	heuristics(problem, state, partial_solution, solution->n_relocation);
#endif /* !RESTRICTED */

	if(partial_solution->n_relocation < solution->n_relocation) {
	  /* better upper bound is found */
	  copy_solution(solution, partial_solution);
	  fprintf(stderr, "ub=%d depth=%d ", solution->n_relocation, depth);
	  print_time(problem);

	  if(solution->n_relocation <= *ub) {
	    /* When a solution as good as *ub is found, */
	    /* the search is terminated */

	    /* recover from the backup */
	    state->ystate = ystate;
	    state->ship_n_tier = ship_n_tier;
	    state->n_blocking4 = n_blocking4;
	    state->blocking4_matrix = blocking4_matrix;

	    return(TRUE);
	  }
	}
      }
#endif /* HEURISTIC == 1 */

      /* list of the child nodes */
      cn[max_n_child - 1].src_stack = i;
      cn[max_n_child - 1].dst_stack = j;
      cn[max_n_child - 1].index = n_child;
      cn[max_n_child - 1].n_blocking = state->n_blocking;
      cn[max_n_child - 1].lb = lb;
      cn[max_n_child - 1].blocking = blocking;
      cn[max_n_child - 1].ship_n_block = state->ship_n_block;

#if 1
      cn[max_n_child - 1].score = problem->yard_s_height
	*(problem->yard_s_height
	  *(problem->yard_s_height
	    *(problem->yard_s_height - ystate[i].n_target)
	    + ystate[i].n_stacked) + ystate[j].n_target) + ystate[j].n_stacked;
#endif
#if 0
      cn[max_n_child - 1].score = problem->yard_s_height
	*(problem->yard_s_height
	  *(problem->yard_s_height*ystate[i].n_stacked
	    + (problem->yard_s_height - ystate[i].n_target))
	  + ystate[j].n_stacked) + ystate[j].n_target;
#endif

#if 0
      {
	int nonb_height;

	if(ystate[i].n_stacked < problem->yard_s_height) {
	  nonb_height = ystate[i].n_stacked
	    - (yard_block[i][ystate[i].n_tier - 1].n_blocking
	       - yard_block[i][ystate[i].n_tier - ystate[i].n_stacked - 1]
	       .n_blocking);
	} else {
	  nonb_height = problem->yard_s_height;
	}

	cn[max_n_child - 1].score = problem->yard_s_height
	  *(problem->yard_s_height
	    *(problem->yard_s_height
	      *(problem->yard_s_height - ystate[i].n_target) + nonb_height)
	    + ystate[j].n_target) + ystate[j].n_stacked;
      }
#endif

#if 0
      {
	int nonb_height;

	if(ystate[i].n_stacked < problem->yard_s_height) {
	  nonb_height = ystate[i].n_stacked
	    - (yard_block[i][ystate[i].n_tier - 1].n_blocking
	       - yard_block[i][ystate[i].n_tier - ystate[i].n_stacked - 1]
	       .n_blocking);
	} else {
	  nonb_height = problem->yard_s_height;
	}

	cn[max_n_child - 1].score = problem->yard_s_height
	  *(problem->yard_s_height
	    *((problem->yard_s_height + 1)*nonb_height
	      + (problem->yard_s_height - ystate[i].n_target))
	    + ystate[j].n_stacked) + ystate[j].n_target;
      }
#endif

#if 0
      {
	int nonb_height;
	int spr = problem->max_ship_tier;

	if(ystate[i].n_stacked < problem->yard_s_height) {
	  nonb_height = ystate[i].n_stacked
	    - (yard_block[i][ystate[i].n_tier - 1].n_blocking
	       - yard_block[i][ystate[i].n_tier - ystate[i].n_stacked - 1]
	       .n_blocking);
	} else {
	  nonb_height = problem->yard_s_height;
	}

	for(k = 0; k < ystate[j].n_tier; ++k) {
	  int pr = problem->ship_position[yard_block[j][k].no].t
	    - ship_n_tier[problem->ship_position[yard_block[j][k].no].s];
	  spr = min(spr, pr);
	}

	cn[max_n_child - 1].score = problem->max_ship_tier
	  *(problem->yard_s_height
	    *(problem->yard_s_height*nonb_height
	      + (problem->yard_s_height - ystate[i].n_target))
	    + ystate[j].n_stacked) + spr;
      }
#endif

      /* the node with the largest lower bound is branched first */

      /* insertion sort */
      for(k = n_child - 1; k >= 0
	    && (cn[k].lb > lb
		|| (cn[k].lb == lb && cn[k].score > cn[max_n_child - 1].score));
	  --k) {
	cn[k + 1] = cn[k];
      }
      cn[k + 1] = cn[max_n_child - 1];

      ++n_child;

      yard_block[j][ystate[j].n_tier] = dblock;
    }

    ystate[i] = ystate_backup;
    yard_position[rblock.no] = yard_position_backup;
#ifdef DOMINANCE_CHECK
    state->last_relocation[rblock.no] = last_reloc;
#endif /* DOMINANCE_CHECK */
  }

  /* branching */
  ret = FALSE;
  for(j = 0; j < n_child; ++j) {
    int src_stack = cn[j].src_stack;
    int dst_stack = cn[j].dst_stack;

    /* bounding */
#ifdef PURE_BB
    if(cn[j].lb + depth >= solution->n_relocation) {
      continue;
    }
#else /* !PURE_BB */
    if(cn[j].lb + depth > *ub) {
      continue;
    }
#endif /* !PURE_BB */

    /* update the information for the child node */
    rblock = yard_block[src_stack][ystate[src_stack].n_tier - 1];
    dblock = yard_block[dst_stack][ystate[dst_stack].n_tier];

    yard_block[dst_stack][ystate[dst_stack].n_tier].no = rblock.no;
    yard_block[dst_stack][ystate[dst_stack].n_tier].blocking = cn[j].blocking;
    if(ystate[dst_stack].n_tier > 0) {
      yard_block[dst_stack][ystate[dst_stack].n_tier].n_blocking
	= yard_block[dst_stack][ystate[dst_stack].n_tier - 1].n_blocking
	+ cn[j].blocking;
    }

    yard_position_backup = yard_position[rblock.no];
    yard_position[rblock.no].s = dst_stack;
    yard_position[rblock.no].t = ystate[dst_stack].n_tier;

    state->n_blocking = cn[j].n_blocking;
    state->ship_n_block = cn[j].ship_n_block;
    state->ystate = bbwork[depth].ystate[cn[j].index];
    state->ship_n_tier = bbwork[depth].ship_n_tier[cn[j].index];
    state->n_blocking4 = state->ship_n_tier + problem->ship_n_stack;
    state->blocking4_matrix = bbwork[depth].blocking4_matrix[cn[j].index];

#ifdef DOMINANCE_CHECK
    last_reloc = state->last_relocation[rblock.no];
    state->last_relocation[rblock.no] = depth;
#endif /* DOMINANCE_CHECK */

    /* update the partial solution */
    partial_solution->n_relocation = depth - 1;
    add_relocation(partial_solution, src_stack, dst_stack, rblock.no);

#if 0
    print_state(problem, state, stdout);
#endif

#ifdef RESTRICTED
#ifdef PURE_BB
    ret = bb(problem, state, solution, cn[j].lb, depth + 1,
	     (state->ship_n_block > ship_n_block)?(-1):src_stack);
#else /* !PURE_BB */
    ret = bb(problem, state, solution, ub, depth + 1,
	     (state->ship_n_block > ship_n_block)?(-1):src_stack);
#endif /* !PURE_BB */
#else /* !RESTRICTED */
#ifdef PURE_BB
    ret = bb(problem, state, solution, cn[j].lb, depth + 1);
#else /* !PURE_BB */
    ret = bb(problem, state, solution, ub, depth + 1);
#endif /* !PURE_BB */
#endif /* !RESTRICTED */

    yard_block[dst_stack][ystate[dst_stack].n_tier] = dblock;
    yard_position[rblock.no] = yard_position_backup;
#ifdef DOMINANCE_CHECK
    state->last_relocation[rblock.no] = last_reloc;
#endif /* DOMINANCE_CHECK */

    /* recursive call for branching */
    if(ret != FALSE) {
      /* an optimal solution is found, or time limit is reached */
      break;
    }
  }

#ifdef LOWER_BOUND_COMPUTATION_TIME_CHECK
 bb_end:
#endif /* LOWER_BOUND_COMPUTATION_TIME_CHECK */
  /* recover from the backup */
  state->ship_n_block = ship_n_block;
#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
  if(target_stack == -1) {
    --partial_solution->n_target;
  }
#endif /* DOMINANCE_CHECK */
#endif /* RESTRICTED */
  state->n_blocking = n_blocking;
  state->ystate = ystate;
  state->ship_n_tier = ship_n_tier;
  state->n_blocking4 = n_blocking4;
  state->blocking4_matrix = blocking4_matrix;

  return(ret);
}

#if LOWER_BOUND == 2
int lower_bound2(problem_t *problem, state_t *state)
{
  int i, c = 0;
  int n_cycle4 = 0, bucket_size = 0;

  for(i = 0; i < problem->yard_n_block; ++i) {
    bucket_size = max(bucket_size, state->n_blocking4[i]);
  }
  for(i = 0; i <= bucket_size; lb_blocking4_bucket[i++] = 0);

  for(i = 0; i < problem->yard_n_block; ++i) {
    ++lb_blocking4_bucket[state->n_blocking4[i]];
    n_cycle4 += state->n_blocking4[i];
  }
  n_cycle4 /= 2;

  if(n_cycle4 > 0) {
    for(i = bucket_size; n_cycle4 > 0 && i > 0; --i) {
      if(n_cycle4 > i*lb_blocking4_bucket[i]) {
	n_cycle4 -= i*lb_blocking4_bucket[i];
	c += lb_blocking4_bucket[i];
      } else {
	c += (n_cycle4 + i - 1)/i;
	break;
      }
    }
  }

  return(state->n_blocking + c);
}
#elif LOWER_BOUND == 3
static int bucket_size;

#ifdef RESTRICTED
static int lower_bound3_sub(problem_t *, state_t *, int, int, int, int, int *);
#else /* !RESTRICTED */
static int lower_bound3_sub(problem_t *, state_t *, int, int, int, int *);
#endif /* !RESTRICTED */

#ifdef RESTRICTED
int lower_bound3(problem_t *problem, state_t *state, int target_stack, int ub)
#else /* !RESTRICTED */
int lower_bound3(problem_t *problem, state_t *state, int ub)
#endif /* !RESTRICTED */
{
  int i;
  int c = 0, n_cycle4 = 0, cn_cycle4;
  int ship_n_block_backup = state->ship_n_block;
  int *ship_n_tier_backup = state->ship_n_tier;
  yard_stack_state_t *ystate_backup = state->ystate;

#ifdef LOWER_BOUND_COMPUTATION_TIME_CHECK
  lb_count = 0;
#endif /* LOWER_BOUND_COMPUTATION_TIME_CHECK */

  bucket_size = 0;
  for(i = 0; i < problem->yard_n_block; ++i) {
    bucket_size = max(bucket_size, state->n_blocking4[i]);
  }
  for(i = 0; i <= bucket_size; lb_blocking4_bucket[0][i++] = 0);

  for(i = 0; i < problem->yard_n_block; ++i) {
    ++lb_blocking4_bucket[0][state->n_blocking4[i]];
    n_cycle4 += state->n_blocking4[i];
  }
  n_cycle4 /= 2;

  if(n_cycle4 > 0) {
    for(i = bucket_size, cn_cycle4 = n_cycle4; i > 0; --i) {
      if(cn_cycle4 > i*lb_blocking4_bucket[0][i]) {
	cn_cycle4 -= i*lb_blocking4_bucket[0][i];
	c += lb_blocking4_bucket[0][i];
      } else {
	c += (cn_cycle4 + i - 1)/i;
	break;
      }
    }
  }

  if(state->n_blocking + c >= ub) {
    return(ub);
  }

#if 0
  printf("depth=%d lb=%d+%d ub=%d\n", 0, state->n_blocking, c, ub);
#endif

  memcpy((void *) lb_ship_n_tier[0], (void *) ship_n_tier_backup,
	 (size_t) (problem->ship_n_stack + problem->yard_n_block)*sizeof(int));
  memcpy((void *) lb_ystate[0], (void *) ystate_backup,
	 (size_t) problem->yard_n_stack*sizeof(yard_stack_state_t));
#ifdef LB_BB_DOMINANCE
  memset((void *) lb_dominated[0], 0, (size_t) problem->yard_n_stack);
#ifdef RESTRICTED
  if(target_stack >= 0) {
    memset((void *) lb_dominated[1], 0, (size_t) problem->yard_n_stack);
  }
#endif /* RESTRICTED */
#endif /* LB_BB_DOMINANCE */

  i = ub;
#ifdef RESTRICTED
  lower_bound3_sub(problem, state, 1, state->n_blocking, n_cycle4,
		   target_stack, &ub);
#else /* !RESTRICTED */
  lower_bound3_sub(problem, state, 1, state->n_blocking, n_cycle4, &ub);
#endif /* !RESTRICTED */

  state->ship_n_tier = ship_n_tier_backup;
  state->n_blocking4 = ship_n_tier_backup + problem->ship_n_stack;
  state->ystate = ystate_backup;
  state->ship_n_block = ship_n_block_backup;

  return(ub);
}

#ifdef RESTRICTED
int lower_bound3_sub(problem_t *problem, state_t *state, int depth,
		     int current_lb, int n_cycle4, int target_stack,
		     int *min_lb)
#else /* !RESTRICTED */
int lower_bound3_sub(problem_t *problem, state_t *state, int depth,
		     int current_lb, int n_cycle4, int *min_lb)
#endif /* !RESTRICTED */
{
  int i, j, k;
  int ship_n_block_backup = state->ship_n_block;
  int cn_cycle4, nn_cycle4;
  int *cship_n_tier = lb_ship_n_tier[depth];
  int *cn_blocking4 = lb_ship_n_tier[depth] + problem->ship_n_stack;
  int *cship_block = lb_ship_block[depth];
  int *pblocking4_bucket = lb_blocking4_bucket[depth - 1];
  int *cblocking4_bucket = lb_blocking4_bucket[depth];
#ifdef LB_BB_DOMINANCE
  uchar *pdominated = lb_dominated[depth - 1];
  uchar *cdominated = lb_dominated[depth];
#endif /* LB_BB_DOMINANCE */
  yard_stack_state_t *pystate = lb_ystate[depth - 1];
  yard_stack_state_t *cystate = lb_ystate[depth];
  coordinate_t *ship_position = problem->ship_position;
  block_t **yard_block = state->yard_block;

#ifdef LOWER_BOUND_COMPUTATION_TIME_CHECK
  if(tlimit > 0 && ++lb_count == 100000) {
    lb_count = 0;
    if(get_time(problem) >= (double) tlimit) {
      return(-1);
    }
  }
#endif /* LOWER_BOUND_COMPUTATION_TIME_CHECK */

  for(i = 0; i < problem->yard_n_stack; ++i) {
    int lb = current_lb, c = 0;

#ifdef RESTRICTED
    if(target_stack >= 0 && i != target_stack) {
      continue;
    }
#endif /* !RESTRICTED */

#ifdef LB_BB_DOMINANCE
    if(pystate[i].n_stacked == problem->yard_s_height || pdominated[i]) {
      continue;
    }
#else /* !LB_BB_DOMINANCE */
    if(pystate[i].n_stacked == problem->yard_s_height) {
      continue;
    }
#endif /* !LB_BB_DOMINANCE */

    lb += pystate[i].n_stacked - yard_block[i][pystate[i].n_tier - 1].n_blocking
      + yard_block[i][pystate[i].n_tier - pystate[i].n_stacked - 1].n_blocking;
    if(lb >= *min_lb) {
      continue;
    }

    state->ship_n_block = ship_n_block_backup + pystate[i].n_stacked;
    if(state->ship_n_block + 1 == problem->ship_n_block) {
      *min_lb = lb;
      continue;
    }

    memcpy((void *) cship_n_tier, (void *) lb_ship_n_tier[depth - 1],
	   (size_t) (problem->ship_n_stack + problem->yard_n_block)
	   *sizeof(int));
    memcpy((void *) cship_block, (void *) lb_ship_block[depth - 1],
	   (size_t) problem->yard_n_block*sizeof(int));

    for(j = 0; j <= bucket_size;
	cblocking4_bucket[j] = pblocking4_bucket[j], ++j);

    nn_cycle4 = n_cycle4;
    for(j = pystate[i].n_tier - 1;
	j >= pystate[i].n_tier - pystate[i].n_stacked; --j) {
      int no = yard_block[i][j].no, s = ship_position[no].s;

      if(!yard_block[i][j].blocking) {
	--cblocking4_bucket[cn_blocking4[no]];
	cn_blocking4[no] = 0;

	for(k = 0; k < problem->yard_n_block; ++k) {
	  if(state->blocking4_matrix[no][k] && cn_blocking4[k] > 0) {
	    --nn_cycle4;
	    --cblocking4_bucket[cn_blocking4[k]--];
	    ++cblocking4_bucket[cn_blocking4[k]];
	  }
	}
      }

      if(s >= 0) {
	int h;

	cship_block[problem->ship_block[s][ship_position[no].t]] = -1;
	for(h = ship_position[no].t + 1;
	    h < problem->ship_n_tier[s]
	      && cship_block[problem->ship_block[s][h]] == -1; ++h);
	for(k = ship_position[no].t - 1;
	    k >= 0 && cship_block[problem->ship_block[s][k]] == -1; --k);
	cship_block[problem->ship_block[s][k]] = h;
      }
    }

    if(nn_cycle4 > 0) {
      for(j = bucket_size, cn_cycle4 = nn_cycle4; j > 0; --j) {
	if(cn_cycle4 > j*cblocking4_bucket[j]) {
	  cn_cycle4 -= j*cblocking4_bucket[j];
	  c += cblocking4_bucket[j];
	} else {
	  c += (cn_cycle4 + j - 1)/j;
	  break;
	}
      }
    }

    if(lb + c >= *min_lb) {
      continue;
    }

    memcpy((void *) cystate, (void *) pystate,
	   (size_t) problem->yard_n_stack*sizeof(yard_stack_state_t));

    cystate[i].n_tier -= cystate[i].n_stacked;
    cystate[i].n_stacked = 0;
    state->ship_n_tier = cship_n_tier;
    state->n_blocking4 = cn_blocking4;
    state->ystate = cystate;

    move_all_blocks3(problem, state, cship_block, i);

    if(state->ship_n_block < problem->ship_n_block) {
#if 0
      printf("depth=%d(remaining %d) lb=%d+%d ub=%d\n", depth,
	     problem->ship_n_block -  state->ship_n_block,
	     lb, c, *min_lb);
#endif
#ifdef RESTRICTED
#ifdef LB_BB_DOMINANCE
      if(target_stack == -1) {
	for(j = 0; j < i; ++j) {
	  if(cystate[j].n_tier != pystate[j].n_tier
	     || cystate[j].n_stacked != pystate[j].n_stacked) {
	    cdominated[j] = 0;
	  } else {
	    cdominated[j] = 1;
	  }
	}
	for(j = i; j < problem->yard_n_stack; ++j) {
	  if(cystate[j].n_tier != pystate[j].n_tier
	     || cystate[j].n_stacked != pystate[j].n_stacked) {
	    cdominated[j] = 0;
	  } else {
	    cdominated[j] = pdominated[j];
	  }
	}
      }
#endif /* LB_BB_DOMINANCE */

#ifdef LOWER_BOUND_COMPUTATION_TIME_CHECK
      if(lower_bound3_sub(problem, state, depth + 1, lb, nn_cycle4, -1, min_lb)
	 < 0) {
	*min_lb = -1;
	return(-1);
      }
#else /* !LOWER_BOUND_COMPUTATION_TIME_CHECK */
      lower_bound3_sub(problem, state, depth + 1, lb, nn_cycle4, -1, min_lb);
#endif /* !LOWER_BOUND_COMPUTATION_TIME_CHECK */
#else /* !RESTRICTED */
#ifdef LB_BB_DOMINANCE
      for(j = 0; j < i; ++j) {
	if(cystate[j].n_tier != pystate[j].n_tier
	   || cystate[j].n_stacked != pystate[j].n_stacked) {
	  cdominated[j] = 0;
	} else {
	  cdominated[j] = 1;
	}
      }
      for(j = i; j < problem->yard_n_stack; ++j) {
	if(cystate[j].n_tier != pystate[j].n_tier
	   || cystate[j].n_stacked != pystate[j].n_stacked) {
	  cdominated[j] = 0;
	} else {
	  cdominated[j] = pdominated[j];
	}
      }
#endif /* LB_BB_DOMINANCE */

#ifdef LOWER_BOUND_COMPUTATION_TIME_CHECK
      if(lower_bound3_sub(problem, state, depth + 1, lb, nn_cycle4, min_lb)
	 < 0) {
	*min_lb = -1;
	return(-1);
      }
#else /* !LOWER_BOUND_COMPUTATION_TIME_CHECK */
      lower_bound3_sub(problem, state, depth + 1, lb, nn_cycle4, min_lb);
#endif /* !LOWER_BOUND_COMPUTATION_TIME_CHECK */
#endif /* !RESTRICTED */
    } else {
      *min_lb = lb;
    }
  }

  return(*min_lb);
}
#endif /* LOWER_BOUND == 3 */
