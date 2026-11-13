/*
 * Copyright 2016-2017 Shunji Tanaka.  All rights reserved.
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
 *  $Id: solve.c,v 1.14 2017/03/24 11:17:10 tanaka Exp $
 *  $Revision: 1.14 $
 *  $Date: 2017/03/24 11:17:10 $
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

typedef struct {
  int src_stack;
  int dst_stack;
  int index;
  int min_priority;
  int n_block;
  int lb;
#if LOWER_BOUND != 1
  int lb1;
#endif /* LOWER_BOUND != 1 */
} lbtable_t;

static int ***bbn_tier;
static state_t *state;
static stack_info_t ***bbinfo;
#if LOWER_BOUND == 3
static stack_info_t *lbinfo;
static int *lbpriority;
#endif /* LOWER_BOUND == 3 */
static solution_t *partial_solution;
static lbtable_t **lbtable;
static int initial_lb;

#ifdef REDUCTION_RULE2
static int *dominance_check;
#ifdef RULE2A
static int dominance_table[4][4] =
  { { 0, 1, 0, 0 },
    { 1, 1, 2, 1 },
    { 0, 2, 0, 0 },
    { 0, 1, 0, 0 } };

static struct {
  int priority;
  int src;
  int dst;
} preloc[3];
#endif /* RULE2A */
#endif /* REDUCTION_RULE2 */

static ulint n_node;
static uint count;

static uchar bb(problem_t *, solution_t *, int *, int);
#if LOWER_BOUND == 3
#define lower_bound(x, y) lower_bound3(x, y)
static int lower_bound3(problem_t *, state_t *);
#elif LOWER_BOUND == 2
#define lower_bound(x, y) lower_bound2(x, y)
static int lower_bound2(problem_t *, state_t *);
#endif /* LOWER_BOUND == 2 */

uchar solve(problem_t *problem, solution_t *solution)
{
  int i, j;
  int n_relocation;
  int max_n_child = problem->n_stack*(problem->n_stack - 1) + 1;
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE3)
  int size = 2*problem->n_stack + problem->n_block;
#elif defined(REDUCTION_RULE2)
  int size = 2*problem->n_stack;
#else /* !REDUCTION_RULE1 && !REDUCTION_RULE2 && !REDUCTION_RULE3 */
  int size = problem->n_stack;
#endif /* !REDUCTION_RULE1 && !REDUCTION_RULE2 && !REDUCTION_RULE3 */
  uchar ret;
  state_t *cstate = initialize_state(problem, NULL);

  /* initial upper bound */
  heuristics(problem, cstate, solution, MAX_N_RELOCATION);

#if LOWER_BOUND == 3
#if 1
  lbpriority = (int *) malloc((size_t) 2*problem->n_stack*sizeof(int));
#else
  lbpriority = (int *) malloc((size_t) problem->n_stack*sizeof(int));
#endif

  lbinfo
    = (stack_info_t *) malloc((size_t) problem->n_stack*sizeof(stack_info_t));
#endif /* LOWER_BOUND == 3 */

  n_relocation = solution->n_relocation + 1;
  state = (state_t *) calloc((size_t) n_relocation, sizeof(state_t));
  state[0].n_tier = (int *) calloc((size_t) n_relocation*size, sizeof(int));
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE2) \
  || defined(REDUCTION_RULE3)
  state[0].last_modified = state[0].n_tier + problem->n_stack;
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE3)
  state[0].last_relocation = state[0].last_modified + problem->n_stack;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE3 */
#endif /* REDUCTION_RULE1 || REDUCTION_RULE2 || REDUCTION_RULE3 */
  state[0].block = (block_t **) malloc((size_t) n_relocation*problem->n_stack
				       *sizeof(block_t *));
  state[0].block[0] = (block_t *) malloc((size_t) n_relocation*problem->n_stack
					 *problem->s_height*sizeof(block_t));
  state[0].bi = (block_info_t **) malloc((size_t) n_relocation*problem->n_stack
					 *sizeof(block_info_t *));
  state[0].bi[0]
    = (block_info_t *) malloc((size_t) n_relocation*problem->n_stack
			      *(problem->s_height + 1)*sizeof(block_info_t));
  state[0].info = (stack_info_t *) malloc((size_t) n_relocation*problem->n_stack
					  *sizeof(stack_info_t));
  for(j = 1; j < problem->n_stack; ++j) {
    state[0].block[j] = state[0].block[j - 1] + problem->s_height;
    state[0].bi[j] = state[0].bi[j - 1] + (problem->s_height + 1);
  }

  for(i = 1; i < n_relocation; ++i) {
    state[i].n_tier = state[i - 1].n_tier + size;
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE2) \
  || defined(REDUCTION_RULE3)
    state[i].last_modified = state[i].n_tier + problem->n_stack;
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE3)
    state[i].last_relocation = state[i].last_modified + problem->n_stack;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE3 */
#endif /* REDUCTION_RULE1 || REDUCTION_RULE2 || REDUCTION_RULE3 */
    state[i].block = state[i - 1].block + problem->n_stack;
    state[i].bi = state[i - 1].bi + problem->n_stack;
    state[i].info = state[i - 1].info + problem->n_stack;
    state[i].block[0]
      = state[i - 1].block[0] + problem->n_stack*problem->s_height;
    state[i].bi[0]
      = state[i - 1].bi[0] + problem->n_stack*(problem->s_height + 1);
    for(j = 1; j < problem->n_stack; ++j) {
      state[i].block[j] = state[i].block[j - 1] + problem->s_height;
      state[i].bi[j] = state[i].bi[j - 1] + (problem->s_height + 1);
    }
  }

  copy_state(problem, state, cstate);
  free_state(cstate);

  bbinfo = (stack_info_t ***) malloc((size_t) n_relocation
				     *sizeof(stack_info_t **));
  bbinfo[0] = (stack_info_t **) malloc((size_t) n_relocation
				       *(max_n_child + problem->n_stack)
				       *sizeof(stack_info_t *));
  bbinfo[0][0]
    = (stack_info_t *) malloc((size_t) n_relocation
			      *(max_n_child + problem->n_stack)
			      *problem->n_stack*sizeof(stack_info_t));

  bbn_tier = (int ***) malloc((size_t) n_relocation*sizeof(int **));
  bbn_tier[0] = (int **) malloc((size_t) n_relocation*max_n_child
				*sizeof(int *));
  bbn_tier[0][0] = (int *) malloc((size_t) n_relocation*max_n_child
				  *size*sizeof(int));

  for(i = 0; i < n_relocation; ++i) {
    if(i > 0) {
      bbinfo[i] = bbinfo[i - 1] + (max_n_child + problem->n_stack);
      bbinfo[i][0] = bbinfo[i - 1][0]
	+ (max_n_child + problem->n_stack)*problem->n_stack;
      bbn_tier[i] = bbn_tier[i - 1] + max_n_child;
      bbn_tier[i][0] = bbn_tier[i - 1][0] + max_n_child*size;
    }
    for(j = 1; j < max_n_child; ++j) {
      bbinfo[i][j] = bbinfo[i][j - 1] + problem->n_stack;
      bbn_tier[i][j] = bbn_tier[i][j - 1] + size;
    }
    for(j = max_n_child; j < max_n_child + problem->n_stack; ++j) {
      bbinfo[i][j] = bbinfo[i][j - 1] + problem->n_stack;
    }
  }

  lbtable = (lbtable_t **) malloc((size_t) n_relocation*sizeof(lbtable_t *));
  lbtable[0] = (lbtable_t *) malloc((size_t) n_relocation
				    *max_n_child*sizeof(lbtable_t));

  for(i = 1; i < n_relocation; ++i) {
    lbtable[i] = lbtable[i - 1] + max_n_child;
  }

#ifdef REDUCTION_RULE2
#ifdef RULE2A
  dominance_check = (int *) malloc((size_t) problem->n_stack*sizeof(int));
#else /* !RULE2A */
  dominance_check = (int *) malloc((size_t) n_relocation*sizeof(int));
#endif /* !RULE2A */
#endif /* REDUCTION_RULE2 */

  partial_solution = create_solution();

  /* all the retrievable blocks are retrieved */
  retrieve_all_blocks(problem, state, NULL);

  n_node = 1;
  count = 0;
  ret = True;
  if(state->n_block > 0) {
#if LOWER_BOUND == 3
    initial_lb = state->lb3 = lower_bound(problem, state);
#elif LOWER_BOUND == 2
    initial_lb = lower_bound(problem, state);
#else /* LOWER_BOUND == 1 */
    initial_lb = state[0].lb1;
#endif /* LOWER_BOUND == 1 */

    fprintf(stderr, "initial lb=%d ub=%d\n", initial_lb,
	    solution->n_relocation);

    if(initial_lb < solution->n_relocation) {
      int ub = initial_lb;

      /* main loop */
      for(; ub < solution->n_relocation; ++ub) {
	fprintf(stderr, "cub=%d ", ub);
	print_time(problem, stderr);
	if((ret = bb(problem, solution, &ub, 1)) == TimeLimit) {
	  break;
	}
	initial_lb = 0;
      }
    } else {
      printf("Initial upper bound is optimal.\n");
    }
  } else {
    printf("Trivial optimal solution (0 relocation).\n");
  }

  fprintf(stderr, "nodes=%llu\n", n_node);

  free_solution(partial_solution);
#ifdef REDUCTION_RULE2
  free(dominance_check);
#endif /* REDUCTION_RULE2 */
  free(lbtable[0]);
  free(lbtable);
  free(bbn_tier[0][0]);
  free(bbn_tier[0]);
  free(bbn_tier);
  free(bbinfo[0][0]);
  free(bbinfo[0]);
  free(bbinfo);
#if LOWER_BOUND == 3
  free(lbinfo);
  free(lbpriority);
#endif /* LOWER_BOUND == 3 */
  free(state[0].info);
  free(state[0].bi[0]);
  free(state[0].bi);
  free(state[0].block[0]);
  free(state[0].block);
  free(state[0].n_tier);
  free(state);
  heuristics(NULL, NULL, NULL, 0);

  return((ret == TimeLimit)?False:True);
}

uchar bb(problem_t *problem, solution_t *solution, int *ub, int level)
{
  int i, j, k;
  int max_n_child = problem->n_stack*(problem->n_stack - 1) + 1;
  int n_child = 0;
  int src_stack, dst_stack, max_index;
  int min_priority;
#if LOWER_BOUND != 1
  int lb;
#endif /* LOWER_BOUND != 1 */
  uchar ret;
  block_t reloc_block;
  stack_info_t current, *info, *cinfo, *ninfo;
  int *n_tier, *nn_tier;
  state_t *pstate = &(state[level - 1]);
#ifdef REDUCTION_RULE2
#ifdef RULE2A
  int *pdominance_table;
#else /* !RULE2A */
  int src_level, dst_level;
#endif /* !RULE2A */
#endif /* REDUCTION_RULE2 */
#ifdef REDUCTION_RULE1
  int *last_relocation;
#endif /* REDUCTION_RULE1 */
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE2)
  int lv;
  int *last_modified;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE2 */
  state_t *cstate = &(state[level]);
  lbtable_t *lbt = lbtable[level];
  
  if(level > *ub) {
    return(False);
  }

  if(tlimit > 0 && ++count == 200000) {
    count = 0;
    if(get_time(problem) >= (double) tlimit) {
      return(TimeLimit);
    }
  }

#if 0
  printf("level=%d\n", level);
  print_state(problem, pstate, stdout);
  for(i = 0; i < problem->n_stack; ++i) {
    printf("[%d:%d:%d:%d]", pstate->info[i].stack, pstate->info[i].n_space,
	   pstate->info[i].min_priority, pstate->info[i].n_stacked);
  }
  printf("\n");
#endif

  /* copy the state from the parent node */
  copy_state(problem, cstate, pstate);

  /* backup the state */
  info = cstate->info;
  n_tier = cstate->n_tier;

#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE2)
  last_modified = cstate->last_modified;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE2 */
#ifdef REDUCTION_RULE1
  last_relocation = cstate->last_relocation;
#endif /* REDUCTION_RULE1 */

  for(i = 0; i < problem->n_stack && n_tier[info[i].stack] > 0; ++i);
  if(i < problem->n_stack) {
    max_index = i + 1;
  } else {
    max_index = problem->n_stack;
  }

#ifdef REDUCTION_RULE2
#ifdef RULE2A
  memset((void *) dominance_check, 0, problem->n_stack*sizeof(int));
  preloc[0].priority = preloc[1].priority = preloc[2].priority = -1;
#else /* !RULE2A */
  dominance_check[level - 1] = 0;
#endif /* !RULE2A */

  if(level >= 2) {
#ifdef RULE2A
    int priority, dst;

    for(i = 0; i < problem->n_stack; ++i) {
      if(cstate->last_modified[i] > MAX_N_RELOCATION) {
	lv = last_modified[i] - MAX_N_RELOCATION;
	dst = partial_solution->relocation[lv - 1].dst;
	if(last_modified[dst] == lv) {
	  priority = state[lv - 1].bi[i][state[lv - 1].n_tier[i]].min_priority;
	  if(priority > preloc[0].priority) {
	    preloc[2] = preloc[1];
	    preloc[1] = preloc[0];
	    preloc[0].priority = priority;
	    preloc[0].src = i;
	    preloc[0].dst = dst;
	  } else if(priority > preloc[1].priority) {
	    preloc[2] = preloc[1];
	    preloc[1].priority = priority;
	    preloc[1].src = i;
	    preloc[1].dst = dst;
	  } else if(priority > preloc[2].priority) {
	    preloc[2].priority = priority;
	    preloc[2].src = i;
	    preloc[2].dst = dst;
	  }
	}
      }
    }

    if(preloc[0].priority >= 0) {
      dominance_check[preloc[0].src] = dominance_check[preloc[0].dst] = 1;
      if(preloc[1].priority >= 0) {
	dominance_check[preloc[1].src] = dominance_check[preloc[1].dst] = 2;
	if(preloc[2].priority >= 0) {
	  dominance_check[preloc[2].src] = dominance_check[preloc[2].dst] = 3;
	}
      }
    }
#else /* !RULE2A */
    int priority, src;

    for(lv = level - 2; lv >= 0; --lv) {
      src = partial_solution->relocation[lv].src;
      priority = state[lv].bi[src][state[lv].n_tier[src]].min_priority;
      dominance_check[lv] = max(dominance_check[lv + 1], priority);
    }
#endif /* !RULE2A */
  }
#endif /* REDUCTION_RULE2 */

  for(i = 0; i < max_index; ++i) {
    /* src_stack: source stack of the relocation */
    src_stack = info[i].stack;

    /* no block */
    if(n_tier[src_stack] == 0) {
      break;
    }

    /* reloc_block: the block to be relocated */
    reloc_block = cstate->block[src_stack][n_tier[src_stack] - 1];

    min_priority = cstate->bi[src_stack][n_tier[src_stack] - 1].min_priority;

#ifdef REDUCTION_RULE1
    lv = last_relocation[reloc_block.no];
    if(lv > 0) {
      uchar flag = False;

      if(last_modified[partial_solution->relocation[lv - 1].src]
	 == MAX_N_RELOCATION + lv) {
	continue;
      }

      if(last_modified[src_stack] != lv) {
	for(j = problem->n_stack - 1; j >= 0; --j) {
	  if(info[j].n_space > 0
	     && (last_modified[info[j].stack] + MAX_N_RELOCATION)
	     %MAX_N_RELOCATION < lv) {
	    flag = True;
	    break;
	  }
	}
      } else if(n_tier[src_stack] > 1) {
	for(j = problem->n_stack - 1; info[j].min_priority > min_priority;
	    --j) {
	  if(info[j].n_space > 0
	     && (last_modified[info[j].stack] + MAX_N_RELOCATION)
	     %MAX_N_RELOCATION < lv) {
	    flag = True;
	    break;
	  }
	}
      }

      if(flag == True) {
	continue;
      }
    }
#endif /* REDUCTION_RULE1 */

    cinfo = bbinfo[level][i];
    memcpy((void *) cinfo, (void *) info,
	   (size_t) problem->n_stack*sizeof(stack_info_t));

    /* remove the block from the source stack */
    ++cinfo[i].n_space;
    --n_tier[src_stack];
    cinfo[i].min_priority = min_priority;
    cinfo[i].n_stacked = cstate->bi[src_stack][n_tier[src_stack]].n_stacked;

    /* insertion sort of stacks */
    current = cinfo[i];
    for(k = i; k < problem->n_stack - 1
	  && stack_info_comp((void *) &(cinfo[k + 1]),
			     (void *) &current) < 0; ++k) {
      cinfo[k] = cinfo[k + 1];
    }
    cinfo[k] = current;

#ifdef REDUCTION_RULE2
#ifdef RULE2A
    pdominance_table = dominance_table[dominance_check[src_stack]];
#else /* !RULE2A */
    src_level = (last_modified[src_stack] + MAX_N_RELOCATION)%MAX_N_RELOCATION;
#endif /* !RULE2A */
#endif /* REDUCTION_RULE2 */

    /* enumerate the candidates for the destination stack */
    for(j = 0; j < max_index; ++j) {
      /* destination stack */
      dst_stack = cinfo[j].stack;

      /* destination = source or no space in the destination stack */
      if(dst_stack == src_stack || cinfo[j].n_space == 0) {
	/* destination == source or no space */
	continue;
      }

#ifdef REDUCTION_RULE1
      if((last_modified[dst_stack] + MAX_N_RELOCATION)%MAX_N_RELOCATION < lv) {
	continue;
      }
#endif /* REDUCTION_RULE1 */

#ifdef REDUCTION_RULE2
#ifdef RULE2A
      if(info[i].min_priority
	 < preloc[pdominance_table[dominance_check[dst_stack]]].priority) {
	continue;
      }
#else /* !RULE2A */
      dst_level
	= (last_modified[dst_stack] + MAX_N_RELOCATION)%MAX_N_RELOCATION;
      if(info[i].min_priority < dominance_check[max(dst_level, src_level)]) {
	continue;
      }
#endif /* !RULE2A */
#endif /* REDUCTION_RULE2 */

      ++n_node;

#if 0
      if(n_node > 1U<<30) {
	print_time(problem, stderr);
	exit(1);
      }
#endif

      /* LB1 after relocation */
      cstate->lb1 = pstate->lb1
	- ((info[i].min_priority < reloc_block.priority)?1:0)
        + ((cinfo[j].min_priority < reloc_block.priority)?1:0);

      if(level + cstate->lb1 > *ub) {
	continue;
      }

      /* create and copy the state for a child node */
      cstate->n_block = pstate->n_block;
      cstate->info = ninfo = bbinfo[level][problem->n_stack + n_child];
      cstate->n_tier = nn_tier = bbn_tier[level][n_child];
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE2) \
  || defined(REDUCTION_RULE3)
      cstate->last_modified = nn_tier + problem->n_stack;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE2 || REDUCTION_RULE3 */
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE3)
      cstate->last_relocation = nn_tier + 2*problem->n_stack;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE3 */

      memcpy((void *) ninfo, (void *) cinfo,
	     (size_t) problem->n_stack*sizeof(stack_info_t));
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE3)
      memcpy((void *) nn_tier, (void *) n_tier,
	     (size_t) (2*problem->n_stack + problem->n_block)*sizeof(int));
#elif defined(REDUCTION_RULE2)
      memcpy((void *) nn_tier, (void *) n_tier,
	     (size_t) 2*problem->n_stack*sizeof(int));
#else /* !REDUCTION_RULE1 && !REDUCTION_RULE2 && !REDUCTION_RULE3 */
      memcpy((void *) nn_tier, (void *) n_tier,
	     (size_t) problem->n_stack*sizeof(int));
#endif /* !REDUCTION_RULE1 && !REDUCTION_RULE2 && !REDUCTION_RULE3 */

#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE2) \
  || defined(REDUCTION_RULE3)
      cstate->last_modified[src_stack] = MAX_N_RELOCATION + level;
      cstate->last_modified[dst_stack] = level;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE2 || REDUCTION_RULE3 */
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE3)
      cstate->last_relocation[reloc_block.no] = level;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE3 */

      /* update the state of the destination stack */
      --ninfo[j].n_space;

      /* does not become a blocking block after relocation */
      if(ninfo[j].min_priority >= reloc_block.priority) {
	ninfo[j].min_priority = reloc_block.priority;
	ninfo[j].n_stacked = 0;

	/* insertion sort of stacks */
	current = ninfo[j];
	for(k = j; k > 1 && stack_info_comp((void *) &(ninfo[k - 1]),
					    (void *) &current) > 0; --k) {
	  ninfo[k] = ninfo[k - 1];
	}
	ninfo[k] = current;
      } else {
	/* becomes a blocking block after relocation */

	/* increases the number of blocks stacked above the highest */
	/* (smallest) priority block in the dstination stack */
	++ninfo[j].n_stacked;
	current = ninfo[j];
      }
#if 0
      printf("===\n");
      for(k = 0; k < problem->n_stack; ++k) {
	printf("[%d:%d:%d:%d]", ninfo[k].stack, ninfo[k].n_space,
	       ninfo[k].min_priority, ninfo[k].n_stacked);
      }
      printf("\n");
#endif

      /* the block is relocated to the destination stack*/
      cstate->block[dst_stack][nn_tier[dst_stack]++] = reloc_block;

      /* update the table for incremental computation of */
      /* min_priority and n_stacked.                     */
      cstate->bi[dst_stack][nn_tier[dst_stack]].min_priority
	= current.min_priority;
      cstate->bi[dst_stack][nn_tier[dst_stack]].n_stacked = current.n_stacked;
      
      /* partial solution up to the current node */
      partial_solution->n_relocation = level - 1;
      add_relocation(partial_solution, src_stack, dst_stack, &reloc_block);

      if(cinfo[0].n_stacked == 0) {
	/* the target block is now retrievable */
#ifdef REDUCTION_RULE3
	if(retrieve_all_blocks(problem, cstate, partial_solution) == True) {
	  continue;
	}
#else /* !REDUCTION_RULE3 */
	retrieve_all_blocks(problem, cstate, partial_solution);
#endif /* !REDUCTION_RULE3 */

	/* all the retrievable blocks are retrieved */
	if(cstate->n_block == 0) {
	  copy_solution(solution, partial_solution);
	  fprintf(stderr, "ub=%d ", solution->n_relocation);
	  print_time(problem, stderr);

	  cstate->info = info;
	  cstate->n_tier = n_tier;
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE2) \
  || defined(REDUCTION_RULE3)
	  cstate->last_modified = n_tier + problem->n_stack;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE2 || REDUCTION_RULE3 */
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE3)
	  cstate->last_relocation = n_tier + 2*problem->n_stack;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE3 */

	  return(True);
	}
      }

      /* lower bound */
#if LOWER_BOUND != 1
#if LOWER_BOUND == 3
      if(pstate->lb3 - cstate->lb1 == 2) {
	lb = cstate->lb3 = cstate->lb1 + 1;
      } else {
	lb = cstate->lb3 = lower_bound(problem, cstate);
      }
#else /* LOWER_BOUND == 2 */
      lb = lower_bound(problem, cstate);
#endif /* LOWER_BOUND == 2 */

      /* bounding */
      if(lb + level > *ub) {
	continue;
      }
#if 0
      fprintf(stdout, "lb=%d cub=%d ub=%d\n",
	      lb + level, *ub, solution->n_relocation);
#endif
#endif /* LOWER_BOUND != 1 */

#if 1
#if LOWER_BOUND == 1
      if(cstate->lb1 + level == *ub - 1) {
#else /* LOWER_BOUND != 1 */
      if(lb + level == *ub - 1 || cstate->lb1 + level < initial_lb) {
#endif /* LOWER_BOUND != 1 */
	/* upper bound computation */
	heuristics(problem, cstate, partial_solution, solution->n_relocation);

	if(partial_solution->n_relocation < solution->n_relocation) {
	  /* better upper bound is found */
	  copy_solution(solution, partial_solution);
	  fprintf(stderr, "ub=%d ", solution->n_relocation);
	  print_time(problem, stderr);

	  if(solution->n_relocation <= *ub) {
	    /* When a solution as good as *ub is found, */
	    /* the search is terminated */

	    /* recover from the backup */
	    cstate->info = info;
	    cstate->n_tier = n_tier;
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE2) \
  || defined(REDUCTION_RULE3)
	    cstate->last_modified = n_tier + problem->n_stack;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE2 || REDUCTION_RULE3 */
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE3)
	    cstate->last_relocation = n_tier + 2*problem->n_stack;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE3 */
	    return(True);
	  }
	}
#if LOWER_BOUND == 1
      }
#else /* LOWER_BOUND != 1 */
      }
#endif /* LOWER_BOUND != 1 */
#endif

      /* list of the child nodes */
      lbt[max_n_child - 1].src_stack = src_stack;
      lbt[max_n_child - 1].dst_stack = dst_stack;
      lbt[max_n_child - 1].index = n_child;
      lbt[max_n_child - 1].min_priority = cinfo[j].min_priority;
#if LOWER_BOUND == 1
      lbt[max_n_child - 1].lb = cstate->lb1;
#else /* LOWER_BOUND != 1 */
      lbt[max_n_child - 1].lb1 = cstate->lb1;
      lbt[max_n_child - 1].lb = lb;
#endif /* LOWER_BOUND != 1 */

      lbt[max_n_child - 1].n_block = cstate->n_block;

      /* the node with the largest lower bound is branched first */

      /* insertion sort */
      for(k = n_child - 1; k >= 0
	    && (lbt[k].lb > lbt[max_n_child - 1].lb
		|| (lbt[k].lb == lbt[max_n_child - 1].lb
		    && lbt[k].min_priority
		    < lbt[max_n_child - 1].min_priority)); --k) {
	lbt[k + 1] = lbt[k];
      }
      lbt[k + 1] = lbt[max_n_child - 1];

      ++n_child;
    }

    ++n_tier[src_stack];
  }

#if 0
  for(k = 0; k < n_child; ++k) {
    if(lbt[k].lb + level <= *ub) {
      printf("[%d] lb=%d, src_stack=%d, dst_stack=%d, "
	     "index=%d, min_priority=%d\n",
	     level, lbt[k].lb, lbt[k].src_stack, lbt[k].dst_stack, lbt[k].index,
	     lbt[k].min_priority);
    }
  }
#endif

  /* branching */
  ret = False;
  for(j = 0; j < n_child; ++j) {

    /* bounding */
    if(lbt[j].lb + level > *ub) {
      continue;
    }

    /* update the information for the child node */
    src_stack = lbt[j].src_stack;
    dst_stack = lbt[j].dst_stack;
    reloc_block = cstate->block[src_stack][n_tier[src_stack] - 1];

    cstate->block[dst_stack][n_tier[dst_stack]] = reloc_block;
#if LOWER_BOUND == 1
    cstate->lb1 = lbt[j].lb;
#else /* LOWER_BOUND != 1 */
    cstate->lb1 = lbt[j].lb1;
#if LOWER_BOUND == 3
    cstate->lb3 = lbt[j].lb;
#endif /* LOWER_BOUND == 3 */
#endif /* LOWER_BOUND != 1 */
    cstate->n_block = lbt[j].n_block;
    cstate->info = bbinfo[level][problem->n_stack + lbt[j].index];
    cstate->n_tier = bbn_tier[level][lbt[j].index];

#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE2) \
  || defined(REDUCTION_RULE3)
    cstate->last_modified = cstate->n_tier + problem->n_stack;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE2 || REDUCTION_RULE3 */
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE3)
    cstate->last_relocation = cstate->n_tier + 2*problem->n_stack;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE3 */
    if(cstate->bi[dst_stack][n_tier[dst_stack]].min_priority
       >= reloc_block.priority) {
      cstate->bi[dst_stack][n_tier[dst_stack] + 1].min_priority
	= reloc_block.priority;
      cstate->bi[dst_stack][n_tier[dst_stack] + 1].n_stacked = 0;
    } else {
      cstate->bi[dst_stack][n_tier[dst_stack] + 1].min_priority
	= cstate->bi[dst_stack][n_tier[dst_stack]].min_priority;
      cstate->bi[dst_stack][n_tier[dst_stack] + 1].n_stacked
	= cstate->bi[dst_stack][n_tier[dst_stack]].n_stacked + 1;
    }

    /* update the partial solution */
    partial_solution->n_relocation = level - 1;
    add_relocation(partial_solution, src_stack, dst_stack, &reloc_block);

#if 0
    print_state(problem, cstate, stdout);
    for(k = 0; k < problem->n_stack; ++k) {
      printf("[%d:%d:%d:%d]", cstate->info[k].min_priority,
	     cstate->info[k].stack,
	     cstate->info[k].n_stacked,
	     cstate->info[k].n_space);
    }
    printf("\n");
#endif

    /* recursive call for branching */
    if((ret = bb(problem, solution, ub, level + 1)) != False) {
      /* an optimal solution is found, or time limit is reached */
      break;
    }
  }

  /* recover from the backup */
  cstate->info = info;
  cstate->n_tier = n_tier;
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE2) \
  || defined(REDUCTION_RULE3)
  cstate->last_modified = n_tier + problem->n_stack;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE2 || REDUCTION_RULE3 */
#if defined(REDUCTION_RULE1) || defined(REDUCTION_RULE3)
  cstate->last_relocation = n_tier + 2*problem->n_stack;
#endif /* REDUCTION_RULE1 || REDUCTION_RULE3 */

  return(ret);
}

#if LOWER_BOUND == 2
/* Forster and Bortfeldt, 2012 */
int lower_bound2(problem_t *problem, state_t *state)
{
  int i;
  int priority;
  int *n_tier = state->n_tier;
  stack_info_t *info = state->info;
  block_t **block = state->block;

  if(info[problem->n_stack - 1].n_space == problem->s_height) {
    /* empty stack found */
    return(state->lb1);
  }

  priority = info[problem->n_stack - 1].min_priority;
  for(i = 0; i < problem->n_stack; ++i) {
    if(info[i].n_stacked > 0
       && block[info[i].stack][n_tier[info[i].stack] - 1].priority
       <= priority) {
      return(state->lb1);
    }
  }

  return(state->lb1 + 1);
}
#elif LOWER_BOUND == 3
int lower_bound3(problem_t *problem, state_t *state)
{
  int i, j, n;
  int src_stack, n_stacked, priority;
  int n_block = state->n_block;
  int *n_tier = state->n_tier, *cn_tier = lbpriority + problem->n_stack;
  block_t **block = state->block;
  stack_info_t *info = state->info, *cinfo = lbinfo;

  for(i = problem->n_stack - 1; i > 0 && info[i].n_space == 0; --i);
  lbpriority[0] = info[i].min_priority;
  src_stack = info[0].stack;
  n_stacked = info[0].n_stacked;
  priority = block[src_stack][n_tier[src_stack] - 1].priority;

  if(lbpriority[0] < priority) {
    return(state->lb1 + 1);
  }

  if(n_stacked > 1) {
    for(--i, n = 1; i > 0; --i) {
      if(info[i].n_space > 0
	 || (info[i].n_stacked > 0
	     && block[info[i].stack][n_tier[info[i].stack] - 1].priority
	     <= lbpriority[0])) {
	lbpriority[n++] = info[i].min_priority;
      }
    }

    for(i = 1; i < n_stacked; ++i) {
      for(j = n - 1; lbpriority[j] < priority; --j);
      lbpriority[j] = priority;
      priority = block[src_stack][n_tier[src_stack] - i - 1].priority;
      if(priority > lbpriority[0]) {
	return(state->lb1 + 1);
      }
    }
  }

  memcpy((void *) cinfo, (void *) info,
	 (size_t) problem->n_stack*sizeof(stack_info_t));
  memcpy((void *) cn_tier, (void *) n_tier,
	 (size_t) problem->n_stack*sizeof(int));

  state->info = cinfo;
  state->n_tier = cn_tier;

  cn_tier[src_stack] -= n_stacked;

  while(state->n_block > 0) {
    state->n_block -= n_stacked;
    cinfo[0].n_space += n_stacked;
    cinfo[0].n_stacked = 0;
    retrieve_all_blocks(problem, state, NULL);
    if(state->n_block == 0) {
      break;
    }

    for(i = problem->n_stack - 1; i > 0 && cinfo[i].n_space == 0; --i);

    lbpriority[0] = cinfo[i].min_priority;
    src_stack = cinfo[0].stack;
    n_stacked = cinfo[0].n_stacked;
    priority = block[src_stack][cn_tier[src_stack] - 1].priority;

    if(lbpriority[0] < priority) {
      state->n_block = n_block;
      state->info = info;
      state->n_tier = n_tier;
      return(state->lb1 + 1);
    }

    if(n_stacked > 1) {
      --cn_tier[src_stack];
      for(--i, n = 1; i > 0; --i) {
	if(cinfo[i].n_space > 0
	   || (cinfo[i].n_stacked > 0
	       && block[cinfo[i].stack][cn_tier[cinfo[i].stack] - 1].priority
	       <= lbpriority[0])) {
	  lbpriority[n++] = cinfo[i].min_priority;
	}
      }

      for(i = 1; i < n_stacked; ++i) {
	for(j = n - 1; lbpriority[j] < priority; --j);
	lbpriority[j] = priority;
	priority = block[src_stack][--cn_tier[src_stack]].priority;
	if(priority > lbpriority[0]) {
	  state->n_block = n_block;
	  state->info = info;
	  state->n_tier = n_tier;
	  return(state->lb1 + 1);
	}
      }
    }
  }

  state->n_block = n_block;
  state->info = info;
  state->n_tier = n_tier;

  return(state->lb1);
}
#endif /* LOWER_BOUND == 3 */
