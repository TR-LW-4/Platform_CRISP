/*
 * Copyright 2014-2017 Shunji Tanaka.  All rights reserved.
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
 *  $Id: solve.c,v 1.32 2017/03/24 11:17:21 tanaka Exp $
 *  $Revision: 1.32 $
 *  $Date: 2017/03/24 11:17:21 $
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

#ifndef LOWER_BOUND
/* #define LOWER_BOUND (1) */
/* #define LOWER_BOUND (2) */
/* #define LOWER_BOUND (3) */
#define LOWER_BOUND (4)
#endif /* !LOWER_BOUND */

typedef struct {
  int index;
  int n_block;
  int lb;
  int lb1;
} lbtable_t;

static int ***bbn_tier;
static state_t *state;
static stack_info_t ***bbinfo;
#if LOWER_BOUND == 3 || LOWER_BOUND == 4
static stack_info_t *lbinfo;
static int *lbn_tier;
#endif /* LOWER_BOUND == 3 || LOWER_BOUND == 4*/
static solution_t *partial_solution;
static lbtable_t **lbtable;
static int initial_lb;

static ulint n_node;
static uint count;

static uchar bb(problem_t *, solution_t *, int *, int);
#if LOWER_BOUND == 2
#define lower_bound(x, y) lower_bound2(x, y)
static int lower_bound2(problem_t *, state_t *);
#elif LOWER_BOUND == 3
#define lower_bound(x, y) lower_bound3(x, y)
static int lower_bound3(problem_t *, state_t *);
#elif LOWER_BOUND == 4
#define lower_bound(x, y) lower_bound4(x, y)
static int lower_bound4(problem_t *, state_t *);
#if 1
static void lb_sub(problem_t *, int, int, int *, int, int, stack_info_t *, int,
		   int *);
#else
static void lb_sub(problem_t *, int, int, int *, int, int *, int, int *);
#endif
#endif /* LOWER_BOUND */

uchar solve(problem_t *problem, solution_t *solution)
{
  int i, j;
  int n_relocation;
  uchar ret;
  state_t *cstate = initialize_state(problem, NULL);

  heuristics(problem, cstate, solution, MAX_N_RELOCATION);

#if LOWER_BOUND == 3 || LOWER_BOUND == 4
#if 1
  lbn_tier = (int *) malloc((size_t) (problem->n_stack + problem->s_height)
			    *sizeof(int));
#else
  lbn_tier
    = (int *) malloc((size_t) (2*problem->n_stack + problem->s_height + 1)
		     *sizeof(int));
#endif
  lbinfo
    = (stack_info_t *) malloc((size_t) problem->n_stack*sizeof(stack_info_t));
#endif /* LOWER_BOUND == 3 || LOWER_BOUND == 4 */

  n_relocation = solution->n_relocation + 1;
  state = (state_t *) calloc((size_t) n_relocation, sizeof(state_t));
  state[0].n_tier
    = (int *) malloc((size_t) n_relocation
		     *(2*problem->n_stack + problem->n_block)*sizeof(int));
  state[0].last_modified = state[0].n_tier + problem->n_stack;
  state[0].last_relocation = state[0].last_modified + problem->n_stack;
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
    state[i].n_tier
      = state[i - 1].n_tier + (2*problem->n_stack + problem->n_block);
    state[i].last_modified = state[i].n_tier + problem->n_stack;
    state[i].last_relocation = state[i].last_modified + problem->n_stack;
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
  bbinfo[0] = (stack_info_t **) malloc((size_t) n_relocation*problem->n_stack
				       *sizeof(stack_info_t *));
  bbinfo[0][0]
    = (stack_info_t *) malloc((size_t) n_relocation*problem->n_stack
			      *problem->n_stack*sizeof(stack_info_t));

  bbn_tier = (int ***) malloc((size_t) n_relocation*sizeof(int **));
  bbn_tier[0] = (int **) malloc((size_t) n_relocation*problem->n_stack
				*sizeof(int *));
  bbn_tier[0][0] = (int *) malloc((size_t) n_relocation*problem->n_stack
				  *(2*problem->n_stack + problem->n_block)
				  *sizeof(int));

  for(i = 0; i < n_relocation; ++i) {
    if(i > 0) {
      bbinfo[i] = bbinfo[i - 1] + problem->n_stack;
      bbinfo[i][0] = bbinfo[i - 1][0] + problem->n_stack*problem->n_stack;
      bbn_tier[i] = bbn_tier[i - 1] + problem->n_stack;
      bbn_tier[i][0] = bbn_tier[i - 1][0]
	+ problem->n_stack*(2*problem->n_stack + problem->n_block);
    }
    for(j = 1; j < problem->n_stack; ++j) {
      bbinfo[i][j] = bbinfo[i][j - 1] + problem->n_stack;
      bbn_tier[i][j] = bbn_tier[i][j - 1]
	+ (2*problem->n_stack + problem->n_block);
    }
  }

  lbtable = (lbtable_t **) malloc((size_t) n_relocation*sizeof(lbtable_t *));
  lbtable[0] = (lbtable_t *) malloc((size_t) n_relocation*problem->n_stack
				    *sizeof(lbtable_t));

  for(i = 1; i < n_relocation; ++i) {
    lbtable[i] = lbtable[i - 1] + problem->n_stack;
  }

  partial_solution = create_solution();

  retrieve_all_blocks(problem, state, NULL);

  n_node = 1;
  count = 0;
  ret = True;
  if(state->n_block > 0) {
#if LOWER_BOUND == 1
    initial_lb = state->lb1;
#else /* LOWER_BOUND != 1 */
    initial_lb = lower_bound(problem, state);
#endif /* LOWER_BOUND != 1 */

    fprintf(stderr, "initial lb=%d ub=%d\n", initial_lb,
	    solution->n_relocation);

    if(initial_lb < solution->n_relocation) {
      int ub = initial_lb;

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
  free(lbtable[0]);
  free(lbtable);
  free(bbn_tier[0][0]);
  free(bbn_tier[0]);
  free(bbn_tier);
  free(bbinfo[0][0]);
  free(bbinfo[0]);
  free(bbinfo);
#if LOWER_BOUND == 3 || LOWER_BOUND == 4
  free(lbn_tier);
  free(lbinfo);
#endif /* LOWER_BOUND == 3 || LOWER_BOUND == 4 */
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
  int j, k;
  int n_child = 0;
  int last_relocation;
  int src_stack, dst_stack, max_index;
#if LOWER_BOUND != 1
  int lb;
#endif /* LOWER_BOUND != 1 */
  uchar ret;
  block_t reloc_block;
  stack_info_t current, *info, *ninfo;
  int *n_tier, *nn_tier;
  state_t *pstate = &(state[level - 1]), *cstate = &(state[level]);
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
  for(j = 0; j < problem->n_stack; ++j) {
    printf("[%d:%d:%d:%d]", pstate->info[j].stack, pstate->info[j].n_space,
	   pstate->info[j].min_priority, pstate->info[j].n_stacked);
  }
  printf("\n");
#endif

  copy_state(problem, cstate, pstate);

  info = cstate->info;
  n_tier = cstate->n_tier;
  src_stack = info[0].stack;
  ++info[0].n_space;
  --info[0].n_stacked;

  reloc_block = cstate->block[src_stack][--n_tier[src_stack]];

  last_relocation = cstate->last_relocation[reloc_block.no];

  for(j = 1; j < problem->n_stack && n_tier[info[j].stack] > 0; ++j);
  if(j < problem->n_stack) {
    max_index = j + 1;
  } else {
    max_index = problem->n_stack;
  }

  for(j = 1; j < max_index; ++j) {
    if(info[j].n_space == 0
       || pstate->last_modified[info[j].stack] < last_relocation) {
      continue;
    }

    ++n_node;
#if 0
    if(n_node > 1U<<30) {
      print_time(problem, stderr);
      exit(1);
    }
#endif

    if(info[j].min_priority < reloc_block.priority
       && level + pstate->lb1 > *ub) {
      continue;
    }

    cstate->lb1 = pstate->lb1 - 1;
    cstate->n_block = pstate->n_block;
    cstate->info = ninfo = bbinfo[level][j];
    cstate->n_tier = nn_tier = bbn_tier[level][j];
    cstate->last_modified = nn_tier + problem->n_stack;
    cstate->last_relocation = nn_tier + 2*problem->n_stack;
    memcpy((void *) ninfo, (void *) info,
	   (size_t) problem->n_stack*sizeof(stack_info_t));
    memcpy((void *) nn_tier, (void *) n_tier,
	   (size_t) (2*problem->n_stack + problem->n_block)*sizeof(int));
    dst_stack = info[j].stack;

    --ninfo[j].n_space;
    if(ninfo[j].min_priority >= reloc_block.priority) {
      ninfo[j].min_priority = reloc_block.priority;
      ninfo[j].n_stacked = 0;
      current = ninfo[j];
      for(k = j; k > 1 && stack_info_comp((void *) &(ninfo[k - 1]),
					  (void *) &current) > 0; --k) {
	ninfo[k] = ninfo[k - 1];
      }
      ninfo[k] = current;
    } else {
      ++cstate->lb1;
      ++ninfo[j].n_stacked;
      current = ninfo[j];
      for(k = j; k < problem->n_stack - 1
	    && stack_info_comp((void *) &(ninfo[k + 1]),
			       (void *) &current) < 0; ++k) {
	ninfo[k] = ninfo[k + 1];
      }
      ninfo[k] = current;
    }
#if 0
    printf("===\n");
    for(k = 0; k < problem->n_stack; ++k) {
      printf("[%d:%d:%d:%d]", ninfo[k].stack, ninfo[k].n_space,
	     ninfo[k].min_priority, ninfo[k].n_stacked);
    }
    printf("\n");
#endif
    cstate->block[dst_stack][nn_tier[dst_stack]++] = reloc_block;
    cstate->last_modified[src_stack] = level;
    cstate->last_modified[dst_stack] = level;
    cstate->last_relocation[reloc_block.no] = level;
    cstate->bi[dst_stack][nn_tier[dst_stack]].min_priority
      = current.min_priority;
    cstate->bi[dst_stack][nn_tier[dst_stack]].n_stacked = current.n_stacked;

    partial_solution->n_relocation = level - 1;
    add_relocation(partial_solution, src_stack, dst_stack, &reloc_block);

    if(info[0].n_stacked == 0) {
      if(retrieve_all_blocks(problem, cstate, partial_solution) == True) {
	continue;
      }
      if(cstate->n_block == 0) {
	copy_solution(solution, partial_solution);
	fprintf(stderr, "ub=%d ", solution->n_relocation);
	print_time(problem, stderr);

	cstate->info = info;
	cstate->n_tier = n_tier;
	cstate->last_modified = n_tier + problem->n_stack;
	cstate->last_relocation = n_tier + 2*problem->n_stack;
	return(True);
      }
    }

#if LOWER_BOUND != 1
    lb = lower_bound(problem, cstate);

    if(lb + level > *ub) {
      continue;
    }
#endif /* LOWER_BOUND == 1 */

#if 1
#if LOWER_BOUND == 1
    if(cstate->lb1 + level == *ub - 1) {
#else /* LOWER_BOUND != 1 */
    if(lb + level == *ub - 1 || cstate->lb1 + level < initial_lb) {
#endif /* LOWER_BOUND != 1 */
      heuristics(problem, cstate, partial_solution, solution->n_relocation);
      if(partial_solution->n_relocation < solution->n_relocation) {
	copy_solution(solution, partial_solution);
	fprintf(stderr, "ub=%d ", solution->n_relocation);
	print_time(problem, stderr);

	if(solution->n_relocation <= *ub) {
	  cstate->info = info;
	  cstate->n_tier = n_tier;
	  cstate->last_modified = n_tier + problem->n_stack;
	  cstate->last_relocation = n_tier + 2*problem->n_stack;
	  return(True);
	}
      }
#if LOWER_BOUND == 1
    }
#else /* LOWER_BOUND != 1 */
    }
#endif /* LOWER_BOUND != 1 */
#endif
    lbt[problem->n_stack - 1].index = j;
    lbt[problem->n_stack - 1].lb = lb;
    lbt[problem->n_stack - 1].lb1 = cstate->lb1;
    lbt[problem->n_stack - 1].n_block = cstate->n_block;
    for(k = n_child - 1; k >= 0
	  && (lbt[k].lb > lbt[problem->n_stack - 1].lb
	      || (lbt[k].lb == lbt[problem->n_stack - 1].lb
		  && info[lbt[k].index].min_priority
		  < info[lbt[problem->n_stack - 1].index].min_priority)); --k) {
      lbt[k + 1] = lbt[k];
    }
    lbt[k + 1] = lbt[problem->n_stack - 1];
    ++n_child;
  }

#if 0
  for(k = 0; k < n_child; ++k) {
    if(lbt[k].lb + level <= *ub) {
      printf("[%d] lb=%d, dst_stack=%d\n", level, lbt[k].lb,
	     info[lbt[k].index].stack);
    }
  }
#endif

  ret = False;
  for(j = 0; j < n_child; ++j) {
    if(lbt[j].lb + level > *ub) {
      continue;
    }

    dst_stack = info[lbt[j].index].stack;
    cstate->block[dst_stack][n_tier[dst_stack]] = reloc_block;
    cstate->lb1 = lbt[j].lb1;
    cstate->n_block = lbt[j].n_block;
    cstate->info = bbinfo[level][lbt[j].index];
    cstate->n_tier = bbn_tier[level][lbt[j].index];
    cstate->last_modified = cstate->n_tier + problem->n_stack;
    cstate->last_relocation = cstate->n_tier + 2*problem->n_stack;
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
    if((ret = bb(problem, solution, ub, level + 1)) != False) {
      break;
    }
  }

  cstate->info = info;
  cstate->n_tier = n_tier;
  cstate->last_modified = n_tier + problem->n_stack;
  cstate->last_relocation = n_tier + 2*problem->n_stack;

  return(ret);
}

#if LOWER_BOUND == 2
int lower_bound2(problem_t *problem, state_t *state)
{
  int i;
  int lb = state->lb1;
  int src_stack = state->info[0].stack;
  int max_priority = state->info[problem->n_stack - 1].min_priority;

  for(i = 0; i < state->info[0].n_stacked; ++i) {
    if(max_priority
       < state->block[src_stack][state->n_tier[src_stack] - i - 1].priority) {
      ++lb;
    }
  }

  return(lb);
}
#endif /* LOWER_BOUND == 2 */

#if LOWER_BOUND == 3
int lower_bound3(problem_t *problem, state_t *state)
{
  int i;
  int lb = 0;
  int n_block = state->n_block;
  int *n_tier = state->n_tier, *cn_tier = lbn_tier;
  stack_info_t *info = state->info, *cinfo = lbinfo;

  memcpy((void *) cinfo, (void *) info,
	 (size_t) problem->n_stack*sizeof(stack_info_t));
  memcpy((void *) cn_tier, (void *) n_tier,
	 (size_t) problem->n_stack*sizeof(int));

  state->info = cinfo;
  state->n_tier = cn_tier;

  while(state->n_block > 0) {
    int src_stack = cinfo[0].stack;
    int n_stacked = cinfo[0].n_stacked;
    int max_priority;

    for(i = problem->n_stack - 1; i > 0 && cinfo[i].n_space == 0; --i);
    if(i == 0) {
      fprintf(stderr, "No space.\n");
      exit(1);
    }
    max_priority = cinfo[i].min_priority;
    lb += n_stacked;

    for(i = 0; i < n_stacked; ++i) {
      if(max_priority
	 < state->block[src_stack][--cn_tier[src_stack]].priority) {
	++lb;
      }
    }
    
    state->n_block -= n_stacked;
    cinfo[0].n_space += n_stacked;
    cinfo[0].n_stacked = 0;

    retrieve_all_blocks(problem, state, NULL);
  }

  state->n_block = n_block;
  state->info = info;
  state->n_tier = n_tier;

  return(lb);
}
#endif /* LOWER_BOUND == 3 */
#if LOWER_BOUND == 4
#if 1
int lower_bound4(problem_t *problem, state_t *state)
{
  int i;
  int lb = 0;
  int n_block = state->n_block;
  int *n_tier = state->n_tier, *cn_tier = lbn_tier;
  int *pr = lbn_tier + problem->n_stack;
  stack_info_t *info = state->info, *cinfo = lbinfo;

  memcpy((void *) cinfo, (void *) info,
	 (size_t) problem->n_stack*sizeof(stack_info_t));
  memcpy((void *) cn_tier, (void *) n_tier,
	 (size_t) problem->n_stack*sizeof(int));

  state->info = cinfo;
  state->n_tier = cn_tier;

  while(state->n_block > 0) {
    int n_pr = 0, max_i;
    int src_stack = cinfo[0].stack;
    int n_stacked = cinfo[0].n_stacked;
    int max_priority, min_priority = problem->max_priority;
    block_t *src_stackp = state->block[src_stack];

    for(max_i = problem->n_stack - 1; max_i > 0 && cinfo[max_i].n_space == 0;
	--max_i);
    if(max_i == 0) {
      fprintf(stderr, "No space.\n");
      exit(1);
    }
    max_priority = cinfo[max_i].min_priority;
    lb += n_stacked;

    for(i = 0; i < n_stacked; ++i) {
      int priority = src_stackp[--cn_tier[src_stack]].priority;

      if(max_priority < priority) {
	++lb;
      } else {
	pr[n_pr++] = priority;
	min_priority = min(min_priority, priority);
      }
    }
    
    if(n_pr > 1) {
      int j, k;
      int f = n_pr - 1;
      for(j = 1; j < problem->n_stack && cinfo[j].n_space == 0; ++j);
      for(k = j; k < problem->n_stack
	    && (cinfo[k].n_space == 0 || cinfo[k].min_priority < min_priority);
	  ++k);
      if(max_i - k >= 0) {
	lb_sub(problem, 0, n_pr, pr, k - j, max_i - j + 1,  cinfo + j, 0, &f);
	lb += f;
      }
    }

    state->n_block -= n_stacked;
    cinfo[0].n_space += n_stacked;
    cinfo[0].n_stacked = 0;

    retrieve_all_blocks(problem, state, NULL);
  }

  state->n_block = n_block;
  state->info = info;
  state->n_tier = n_tier;

  return(lb);
}

void lb_sub(problem_t *problem, int level, int n, int *pr, int s,
	    int w, stack_info_t *info, int f, int *ub)
{
  int i;

  if(level == n) {
    *ub = f;
    return;
  }

  for(i = s; i < w
	&& (info[i].min_priority < pr[level] || info[i].n_space == 0); ++i);
  if(i < w) {
    int prev_priority = info[i].min_priority;
    info[i].min_priority = pr[level];
    lb_sub(problem, level + 1, n, pr, s, w, info, f, ub);
    info[i].min_priority = prev_priority;
    if(*ub == 0) {
      return;
    }
  }

  if(i > 0) {
    if(++f < *ub) {
      lb_sub(problem, level + 1, n, pr, s, w, info, f, ub);
    }
  }
}
#else
int lower_bound4(problem_t *problem, state_t *state)
{
  int i;
  int lb = 0;
  int n_block = state->n_block;
  int *n_tier = state->n_tier, *cn_tier = lbn_tier;
  int *pr = lbn_tier + problem->n_stack;
  int *stackp = lbn_tier + problem->n_stack + problem->s_height;
  stack_info_t *info = state->info, *cinfo = lbinfo;

  memcpy((void *) cinfo, (void *) info,
	 (size_t) problem->n_stack*sizeof(stack_info_t));
  memcpy((void *) cn_tier, (void *) n_tier,
	 (size_t) problem->n_stack*sizeof(int));

  state->info = cinfo;
  state->n_tier = cn_tier;

  while(state->n_block > 0) {
    int n_pr = 0, max_i;
    int src_stack = cinfo[0].stack;
    int n_stacked = cinfo[0].n_stacked;
    int max_priority, min_priority = problem->max_priority;
    block_t *src_stackp = state->block[src_stack];

    for(max_i = problem->n_stack - 1; max_i > 0 && cinfo[max_i].n_space == 0;
	--max_i);
    if(max_i == 0) {
      fprintf(stderr, "No space.\n");
      exit(1);
    }
    max_priority = cinfo[max_i].min_priority;
    lb += n_stacked;

    for(i = 0; i < n_stacked; ++i) {
      int priority = src_stackp[--cn_tier[src_stack]].priority;

      if(max_priority < priority) {
	++lb;
      } else {
	pr[n_pr++] = priority;
	min_priority = min(min_priority, priority);
      }
    }
    
    if(n_pr > 1) {
      int j, ns = 1;
      int f = n_pr - 1;

      stackp[0] = problem->min_priority - 1;
      for(j = 1; j <= max_i; ++j) {
	if(cinfo[j].n_space > 0) {
	  stackp[ns++] = cinfo[j].min_priority;
	}
      }
      stackp[ns] = problem->max_priority + 1;

      lb_sub(problem, 0, n_pr, pr, ns, stackp, 0, &f);
      lb += f;
    }

    state->n_block -= n_stacked;
    cinfo[0].n_space += n_stacked;
    cinfo[0].n_stacked = 0;

    retrieve_all_blocks(problem, state, NULL);
  }

  state->n_block = n_block;
  state->info = info;
  state->n_tier = n_tier;

  return(lb);
}

void lb_sub(problem_t *problem, int level, int n, int *pr, int w,
	    int *info, int f, int *ub)
{
  int s = 0, e = w, c;

  if(level == n) {
    *ub = f;
    return;
  }

  while(e - s > 1) {
    c = (s + e)/2;
    if(info[c] >= pr[level]) {
      e = c;
    } else {
      s = c;
    }
  }

  if(e < w) {
    int prev_priority = info[e];
    info[e] = pr[level];
    lb_sub(problem, level + 1, n, pr, w, info, f, ub);
    info[e] = prev_priority;
    if(*ub == 0) {
      return;
    }
  }

  if(s > 0) {
    if(++f < *ub) {
      lb_sub(problem, level + 1, n, pr, w, info, f, ub);
    }
  }
}
#endif
#endif /* LOWER_BOUND == 4 */
