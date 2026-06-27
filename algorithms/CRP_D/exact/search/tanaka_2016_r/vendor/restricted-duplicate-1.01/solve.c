/*
 * Copyright 2014-2015 Shunji Tanaka.  All rights reserved.
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
 *  $Id: solve.c,v 1.5 2016/04/27 10:33:29 tanaka Exp tanaka $
 *  $Revision: 1.5 $
 *  $Date: 2016/04/27 10:33:29 $
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
/* #define LOWER_BOUND 1 */
/* #define LOWER_BOUND 2 */
/* #define LOWER_BOUND 3 */
#define LOWER_BOUND 4
#endif /* !LOWER_BOUND */

typedef struct {
  int index;
  int n_block;
  int lb;
  int lb1;
} lbtable_t;

static int ***bbn_tier;
static state_t *state;
static stack_info_t *lbinfo, ***bbinfo;
static int *lbn_tier;
static solution_t *partial_solution;
static lbtable_t **lbtable;

static unsigned long long n_node;
static int count;

static uchar bb(problem_t *, solution_t *, int *, int, int);
#if LOWER_BOUND == 2
#define lower_bound(x, y, z) lower_bound2(x, y, z)
static int lower_bound2(problem_t *, state_t *, uchar);
#elif LOWER_BOUND == 3
#define lower_bound(x, y, z) lower_bound3(x, y, z)
static int lower_bound3(problem_t *, state_t *, uchar);
#elif LOWER_BOUND == 4
#define lower_bound(x, y, z) lower_bound4(x, y, z)
static int lower_bound4(problem_t *, state_t *, uchar);
static void lb_sub(problem_t *, int, int, int *, int, int *, int, int *, char);
static int int_comp(const void *, const void *);
#endif /* LOWER_BOUND */

uchar solve(problem_t *problem, solution_t *solution)
{
  int i, j;
  int n_relocation;
  int ret;
  state_t *cstate = initialize_state(problem, NULL);

  heuristics(problem, cstate, solution, FALSE);

  lbn_tier
    = (int *) malloc((size_t) (2*problem->n_stack + problem->s_height + 1)
		     *sizeof(int));
  lbinfo
    = (stack_info_t *) malloc((size_t) problem->n_stack*sizeof(stack_info_t));

  n_relocation = solution->n_relocation + 1;
  state = (state_t *) calloc((size_t) n_relocation, sizeof(state_t));
  state[0].n_tier
    = (int *) malloc((size_t) n_relocation*problem->n_stack*sizeof(int));
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
  for(j = 1; j < problem->n_stack; j++) {
    state[0].block[j] = state[0].block[j - 1] + problem->s_height;
    state[0].bi[j] = state[0].bi[j - 1] + (problem->s_height + 1);
  }

  for(i = 1; i < n_relocation; i++) {
    state[i].n_tier = state[i - 1].n_tier + problem->n_stack;
    state[i].block = state[i - 1].block + problem->n_stack;
    state[i].bi = state[i - 1].bi + problem->n_stack;
    state[i].info = state[i - 1].info + problem->n_stack;
    state[i].block[0]
      = state[i - 1].block[0] + problem->n_stack*problem->s_height;
    state[i].bi[0]
      = state[i - 1].bi[0] + problem->n_stack*(problem->s_height + 1);
    for(j = 1; j < problem->n_stack; j++) {
      state[i].block[j] = state[i].block[j - 1] + problem->s_height;
      state[i].bi[j] = state[i].bi[j - 1] + (problem->s_height + 1);
    }
  }

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
				 *problem->n_stack*sizeof(int));

  for(i = 0; i < n_relocation; i++) {
    if(i > 0) {
      bbinfo[i] = bbinfo[i - 1] + problem->n_stack;
      bbinfo[i][0] = bbinfo[i - 1][0] + problem->n_stack*problem->n_stack;
      bbn_tier[i] = bbn_tier[i - 1] + problem->n_stack;
      bbn_tier[i][0] = bbn_tier[i - 1][0] + problem->n_stack*problem->n_stack;
    }
    for(j = 1; j < problem->n_stack; j++) {
      bbinfo[i][j] = bbinfo[i][j - 1] + problem->n_stack;
      bbn_tier[i][j] = bbn_tier[i][j - 1] + problem->n_stack;
    }
  }

  lbtable = (lbtable_t **) malloc((size_t) n_relocation*sizeof(lbtable_t *));
  lbtable[0] = (lbtable_t *) malloc((size_t) n_relocation*problem->n_stack
				    *sizeof(lbtable_t));

  for(i = 1; i < n_relocation; i++) {
    lbtable[i] = lbtable[i - 1] + problem->n_stack;
  }

  partial_solution = create_solution();

  copy_state(problem, state, cstate);

  free_state(cstate);

  retrieve_all_blocks(problem, state);

  n_node = 1;
  count = 0;
  ret = TRUE;
  if(state->n_block > 0) {
#if LOWER_BOUND == 1
    int lb = state->lb1;
#else /* LOWER_BOUND != 1 */
    int lb = lower_bound(problem, state, FALSE);
#endif /* LOWER_BOUND != 1 */

    fprintf(stderr, "initial lb=%d ub=%d\n", lb, solution->n_relocation);

    if(lb < solution->n_relocation) {
      int ub = lb;

      for(; ub < solution->n_relocation; ub++) {
	fprintf(stderr, "cub=%d ", ub);
	print_time(problem);
	if((ret = bb(problem, solution, &ub, 1, -1)) == TLIMIT) {
	  break;
	}
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
  free(lbn_tier);
  free(lbinfo);
  free(state[0].info);
  free(state[0].bi[0]);
  free(state[0].bi);
  free(state[0].block[0]);
  free(state[0].block);
  free(state[0].n_tier);
  free(state);

  return((ret == TLIMIT)?FALSE:TRUE);
}

uchar bb(problem_t *problem, solution_t *solution, int *ub, int level,
	 int target_w)
{
  int i, j, k;
  int n_child;
  int src_stack, dst_stack, max_w, lb;
  int priority = state[level - 1].info[0].min_priority;
  uchar flag = FALSE, ret;
  block_t reloc_block;
  stack_info_t current, *info, *ninfo;
  int *n_tier, *nn_tier;
  state_t *pstate = &(state[level - 1]), *cstate = &(state[level]);
  lbtable_t *lbt = lbtable[level];
  
  if(level > *ub) {
    return(FALSE);
  }

  if(tlimit > 0 && ++count == 200000) {
    count = 0;
    if(get_time(problem) >= (double) tlimit) {
      return(TLIMIT);
    }
  }

#if 0
  printf("level=%d, target=%d\n", level, target_w);
  print_state(problem, pstate, stdout);
  for(i = 0; i < problem->n_stack; i++) {
    printf("[%d:%d:%d:%d]", pstate->info[i].stack, pstate->info[i].n_space,
	   pstate->info[i].min_priority, pstate->info[i].n_stacked);
  }
  printf("\n");
  print_solution(problem, partial_solution, stdout);
#endif

  if(target_w < 0) {
    /* choose the next target */
    max_w = problem->n_stack;
  } else {
    max_w = 1;
  }

  copy_state(problem, cstate, pstate);
  info = cstate->info;
  n_tier = cstate->n_tier;

  for(i = 0; i < max_w && pstate->info[i].min_priority == priority; i++) {
    uchar status = TRUE;

    if(problem->n_stack*problem->s_height
       < info[0].n_stacked + info[0].n_space) {
      continue;
    }

    /* source stack */

    info[0] = pstate->info[i];
    for(j = 1; j <= i; j++) {
      info[j] = pstate->info[j - 1];
    }

#if 0
#if LOWER_BOUND != 1
    lb = lower_bound(problem, cstate, TRUE);
    n_node++;
    if(lb + level > *ub + 1) {
      continue;
    }
#endif /* LOWER_BOUND != 1 */
#endif

    info[0].n_space++;
    if(--info[0].n_stacked == 0) {
      status = FALSE;
    }

    src_stack = info[0].stack;
    reloc_block = cstate->block[src_stack][--n_tier[src_stack]];

    flag = FALSE;
    n_child = 0;
    for(j = 1; j < problem->n_stack; j++) { /* destination stack */
      if(info[j].n_space == 0) {
	continue;
      }

      if(n_tier[info[j].stack] == 0) { /* empty stack */
	if(flag == TRUE) { /* should be checked only once */
	  continue;
	}
	flag = TRUE;
      }

      n_node++;
#if 0
      if(n_node > 1U<<30) {
	print_time(problem);
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
      memcpy((void *) ninfo, (void *) info,
	     (size_t) problem->n_stack*sizeof(stack_info_t));
      memcpy((void *) nn_tier, (void *) n_tier,
	     (size_t) problem->n_stack*sizeof(int));
      dst_stack = info[j].stack;

      ninfo[j].n_space--;
      if(ninfo[j].min_priority >= reloc_block.priority) {
	if(ninfo[j].min_priority > reloc_block.priority) {
	  ninfo[j].min_priority = reloc_block.priority;
	  ninfo[j].min_priority2
	    = cstate->bi[dst_stack][nn_tier[dst_stack]].min_priority;
	}
	ninfo[j].n_stacked = 0;
	current = ninfo[j];
	for(k = j; k > 1 && stack_info_comp((void *) &(ninfo[k - 1]),
					    (void *) &current) > 0; k--) {
	  ninfo[k] = ninfo[k - 1];
	}
	ninfo[k] = current;
      } else {
	cstate->lb1++;
	ninfo[j].n_stacked++;
	current = ninfo[j];
	for(k = j; k < problem->n_stack - 1
	      && stack_info_comp((void *) &(ninfo[k + 1]),
				 (void *) &current) < 0; k++) {
	  ninfo[k] = ninfo[k + 1];
	}
	ninfo[k] = current;
      }
#if 0
      printf("===\n");
      for(k = 0; k < problem->n_stack; k++) {
	printf("[%d:%d:%d:%d]", ninfo[k].stack, ninfo[k].n_space,
	       ninfo[k].min_priority, ninfo[k].n_stacked);
      }
      printf("\n");
#endif
      cstate->block[dst_stack][nn_tier[dst_stack]++] = reloc_block;
      cstate->bi[dst_stack][nn_tier[dst_stack]].min_priority
	= current.min_priority;
      cstate->bi[dst_stack][nn_tier[dst_stack]].min_priority2
	= current.min_priority2;
      cstate->bi[dst_stack][nn_tier[dst_stack]].n_stacked = current.n_stacked;

      partial_solution->n_relocation = level - 1;
      add_relocation(partial_solution, src_stack, dst_stack, reloc_block.no);

      if(status == FALSE) {
	retrieve_all_blocks(problem, cstate);
	if(cstate->n_block == 0) {
	  if(partial_solution->n_relocation < solution->n_relocation) {
	    copy_solution(solution, partial_solution);
	    fprintf(stderr, "ub=%d ", solution->n_relocation);
	    print_time(problem);

	    if(solution->n_relocation <= *ub) {
	      return(TRUE);
	    }
	  }
	  continue;
	}
      }

#if LOWER_BOUND == 1
      lb = cstate->lb1;
#else /* LOWER_BOUND != 1 */
      lb = lower_bound(problem, cstate, status);

#if 0
      fprintf(stdout, "lb=%d cub=%d ub=%d\n",
	      lb + level, *ub, solution->n_relocation);
#endif
      if(lb + level > *ub) {
	continue;
      }
#endif /* LOWER_BOUND != 1 */

      if(lb + level == *ub - 1) {
	heuristics(problem, cstate, partial_solution, status);
	if(partial_solution->n_relocation < solution->n_relocation) {
	  copy_solution(solution, partial_solution);
	  fprintf(stderr, "ub=%d ", solution->n_relocation);
	  print_time(problem);

	  if(solution->n_relocation <= *ub) {
	    return(TRUE);
	  }
	}
      }

      lbt[problem->n_stack - 1].index = j;
      lbt[problem->n_stack - 1].lb = lb;
      lbt[problem->n_stack - 1].lb1 = cstate->lb1;
      lbt[problem->n_stack - 1].n_block = cstate->n_block;
      for(k = n_child - 1; k >= 0
	    && (lbt[k].lb > lbt[problem->n_stack - 1].lb
		|| (lbt[k].lb == lbt[problem->n_stack - 1].lb
		    && info[lbt[k].index].min_priority
		    < info[lbt[problem->n_stack - 1].index].min_priority));
	  k--) {
	lbt[k + 1] = lbt[k];
      }
      lbt[k + 1] = lbt[problem->n_stack - 1];
      n_child++;
    }

#if 0
    for(k = 0; k < n_child; k++) {
      if(lbt[k].lb + level <= *ub) {
	printf("[%d] lb=%d, dst_stack=%d\n", level, lbt[k].lb,
	       info[lbt[k].index].stack);
      }
    }
#endif
    for(j = 0; j < n_child; j++) {
      if(lbt[j].lb + level > *ub) {
	continue;
      }
      dst_stack = info[lbt[j].index].stack;
      cstate->block[dst_stack][n_tier[dst_stack]] = reloc_block;
      cstate->lb1 = lbt[j].lb1;
      cstate->n_block = lbt[j].n_block;
      cstate->info = bbinfo[level][lbt[j].index];
      cstate->n_tier = bbn_tier[level][lbt[j].index];
      partial_solution->n_relocation = level - 1;
      add_relocation(partial_solution, src_stack, dst_stack, reloc_block.no);
#if 0
      print_state(problem, cstate, stdout);
      for(k = 0; k < problem->n_stack; k++) {
	printf("[%d:%d:%d:%d]", cstate->info[k].stack, cstate->info[k].n_space,
	       cstate->info[k].min_priority, cstate->info[k].n_stacked);
      }
      printf("\n");
#endif
      if((ret = bb(problem, solution, ub, level + 1,
		   (status == FALSE)?(-1):src_stack)) != FALSE) {
	cstate->info = info;
	cstate->n_tier = n_tier;
	return(ret);
      }
    }

    n_tier[src_stack]++;
  }

  cstate->info = info;
  cstate->n_tier = n_tier;

  return(FALSE);
}

#if LOWER_BOUND == 2
int lower_bound2(problem_t *problem, state_t *state, uchar status)
{
  int i, j;
  int lb = state->lb1;
  int src_stack = state->info[0].stack;
  int max_priority = -1;
  stack_info_t *info = state->info;

  if(status == TRUE) {
    max_priority = state->info[problem->n_stack - 1].min_priority;
  } else {
    for(i = 1; i < problem->n_stack
	  && info[i].min_priority == info[0].min_priority; i++) {
      max_priority = max(max_priority, info[i].min_priority2);
    }
    for(j = problem->n_stack - 1; j >= i; j--) {
      if(info[j].n_space > 0) {
	max_priority = max(max_priority, info[j].min_priority);
	break;
      }
    }
  }

  for(i = 0; i < state->info[0].n_stacked; i++) {
    if(max_priority
       < state->block[src_stack][state->n_tier[src_stack] - i - 1].priority) {
      lb++;
    }
  }

  return(lb);
}
#endif /* LOWER_BOUND == 2 */

#if LOWER_BOUND == 3
int lower_bound3(problem_t *problem, state_t *state, uchar status)
{
  int i, j;
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
    int max_priority = -1;

    if(status == TRUE) {
      i = 1;
      status = FALSE;
    } else {
      for(i = 1; i < problem->n_stack
	    && cinfo[i].min_priority == cinfo[0].min_priority; i++) {
	max_priority = max(max_priority, cinfo[i].min_priority2);
      }
    }
    for(j = problem->n_stack - 1; j >= i; j--) {
      if(cinfo[j].n_space > 0) {
	max_priority = max(max_priority, cinfo[j].min_priority);
	break;
      }
    }

    lb += n_stacked;
    for(i = 0; i < n_stacked; i++) {
      if(max_priority
	 < state->block[src_stack][--cn_tier[src_stack]].priority) {
	lb++;
      }
    }

    state->n_block -= n_stacked;
    cinfo[0].n_space += n_stacked;
    cinfo[0].n_stacked = 0;

    retrieve_all_blocks(problem, state);
  }

  state->n_block = n_block;
  state->info = info;
  state->n_tier = n_tier;

  return(lb);
}
#endif /* LOWER_BOUND == 3 */

#if LOWER_BOUND == 4
#if 1
int lower_bound4(problem_t *problem, state_t *state, uchar status)
{
  int i;
  int lb = 0;
  int n_block = state->n_block;
  int *n_tier = state->n_tier, *cn_tier = lbn_tier;
  int *stack_pr = lbn_tier + problem->n_stack;
  int *pr = lbn_tier + 2*problem->n_stack;
  stack_info_t *info = state->info, *cinfo = lbinfo;

  memcpy((void *) cinfo, (void *) info,
	 (size_t) problem->n_stack*sizeof(stack_info_t));
  memcpy((void *) cn_tier, (void *) n_tier,
	 (size_t) problem->n_stack*sizeof(int));

  state->info = cinfo;
  state->n_tier = cn_tier;

  while(state->n_block > 0) {
    int n_pr = 0;
    int src_stack = cinfo[0].stack;
    int n_stacked = cinfo[0].n_stacked;
    int max_priority = -1, min_priority = problem->max_priority + 1;
    int same_i, max_i;

    if(status == TRUE) {
      same_i = 1;
      status = FALSE;
    } else {
      for(same_i = 1; same_i < problem->n_stack
	    && cinfo[same_i].min_priority == cinfo[0].min_priority; same_i++) {
	  max_priority = max(max_priority, cinfo[same_i].min_priority2);
      }
    }

    for(max_i = problem->n_stack - 1; max_i >= same_i; max_i--) {
      if(cinfo[max_i].n_space > 0) {
	max_priority = max(max_priority, cinfo[max_i].min_priority);
	break;
      }
    }

    for(i = 0; i < n_stacked; i++) {
      int priority = state->block[src_stack][--cn_tier[src_stack]].priority;

      if(max_priority < priority) {
	lb++;
      } else {
	pr[n_pr++] = priority;
	min_priority = min(min_priority, priority);
      }
    }

    if(n_pr > 1) {
      int j, w = 0;
      int f = n_pr - 1;
      uchar flag = FALSE, flag2 = FALSE;

      for(j = 1; j < same_i; j++) {
	if(cinfo[j].min_priority2 >= min_priority) {
	  stack_pr[w++] = cinfo[j].min_priority2;
	  flag2 = TRUE;
	} else {
	  flag = TRUE;
	}
      }
      for(j = same_i; j <= max_i; j++) {
	if(cinfo[j].n_space > 0) {
	  if(cinfo[j].min_priority >= min_priority) {
	    stack_pr[w++] = cinfo[j].min_priority;
	  } else {
	    flag = TRUE;
	  }
	}
      }

      if(flag2 == TRUE) {
	qsort((void *) stack_pr, w, sizeof(int), int_comp);
      }
      lb_sub(problem, 0, n_pr, pr, w, stack_pr, 0, &f, flag);
      lb += f;
    }

    lb += n_stacked;

    state->n_block -= n_stacked;
    cinfo[0].n_space += n_stacked;
    cinfo[0].n_stacked = 0;

    retrieve_all_blocks(problem, state);
  }

  state->n_block = n_block;
  state->info = info;
  state->n_tier = n_tier;

  return(lb);
}

void lb_sub(problem_t *problem, int level, int n, int *pr,
	    int w, int *stack_pr, int f, int *ub, char flag)
{
  int i;

  if(level == n) {
    *ub = f;
    return;
  }

  for(i = 0; i < w && stack_pr[i] < pr[level]; i++);

  if(i < w) {
    int prev_priority = stack_pr[i];
    stack_pr[i] = pr[level];
    lb_sub(problem, level + 1, n, pr, w, stack_pr, f, ub, flag);
    stack_pr[i] = prev_priority;
    if(*ub == 0) {
      return;
    }
  }

  if(flag == TRUE || i > 0) {
    if(++f < *ub) {
      lb_sub(problem, level + 1, n, pr, w, stack_pr, f, ub, flag);
    }
  }
}
#else

static void lb_sub2(problem_t *, int, int, int *, int, int *, int, int *);

int lower_bound4(problem_t *problem, state_t *state, uchar status)
{
  int i;
  int lb = 0;
  int n_block = state->n_block;
  int *n_tier = state->n_tier, *cn_tier = lbn_tier;
  int *stack_pr = lbn_tier + problem->n_stack;
  int *pr = lbn_tier + 2*problem->n_stack + 1;
  stack_info_t *info = state->info, *cinfo = lbinfo;

  memcpy((void *) cinfo, (void *) info,
	 (size_t) problem->n_stack*sizeof(stack_info_t));
  memcpy((void *) cn_tier, (void *) n_tier,
	 (size_t) problem->n_stack*sizeof(int));

  state->info = cinfo;
  state->n_tier = cn_tier;

  while(state->n_block > 0) {
    int n_pr = 0;
    int src_stack = cinfo[0].stack;
    int n_stacked = cinfo[0].n_stacked;
    int max_priority = -1, min_priority = problem->max_priority + 1;
    int same_i, max_i;

    if(status == TRUE) {
      same_i = 1;
      status = FALSE;
    } else {
      for(same_i = 1; same_i < problem->n_stack
	    && cinfo[same_i].min_priority == cinfo[0].min_priority; same_i++) {
	  max_priority = max(max_priority, cinfo[same_i].min_priority2);
      }
    }

    for(max_i = problem->n_stack - 1; max_i >= same_i; max_i--) {
      if(cinfo[max_i].n_space > 0) {
	max_priority = max(max_priority, cinfo[max_i].min_priority);
	break;
      }
    }

    for(i = 0; i < n_stacked; i++) {
      int priority = state->block[src_stack][--cn_tier[src_stack]].priority;

      if(max_priority < priority) {
	lb++;
      } else {
	pr[n_pr++] = priority;
	min_priority = min(min_priority, priority);
      }
    }

    if(n_pr > 1) {
      int j, w = 1;
      int f = n_pr - 1;

      stack_pr[0] = problem->min_priority - 1;
      for(j = 1; j < same_i; j++) {
	stack_pr[w++] = cinfo[j].min_priority2;
      }

      for(j = same_i; j <= max_i; j++) {
	if(cinfo[j].n_space > 0) {
	  stack_pr[w++] = cinfo[j].min_priority;
	}
      }
      stack_pr[w] = problem->max_priority + 1;

      if(same_i > 1) {
	qsort((void *) (stack_pr + 1), w, sizeof(int), int_comp);
      }

      lb_sub2(problem, 0, n_pr, pr, w, stack_pr, 0, &f);
      lb += f;
    }

    lb += n_stacked;

    state->n_block -= n_stacked;
    cinfo[0].n_space += n_stacked;
    cinfo[0].n_stacked = 0;

    retrieve_all_blocks(problem, state);
  }

  state->n_block = n_block;
  state->info = info;
  state->n_tier = n_tier;

  return(lb);
}

void lb_sub2(problem_t *problem, int level, int n, int *pr,
	     int w, int *stack_pr, int f, int *ub)
{
  int s = 0, e = w, c;

  if(level == n) {
    *ub = f;
    return;
  }

  while(e - s > 1) {
    c = (s + e)/2;
    if(stack_pr[c] >= pr[level]) {
      e = c;
    } else {
      s = c;
    }
  }

  printf("lv=%d, pr=%d, [%d %d %d]\n", level, pr[level], s, c, e);

  if(e < w) {
    int prev_priority = stack_pr[e];
    stack_pr[e] = pr[level];
    lb_sub2(problem, level + 1, n, pr, w, stack_pr, f, ub);
    stack_pr[e] = prev_priority;
    if(*ub == 0) {
      return;
    }
  }

  if(s > 0) {
    if(++f < *ub) {
      lb_sub2(problem, level + 1, n, pr, w, stack_pr, f, ub);
    }
  }
}
#endif

int int_comp(const void *a, const void *b)
{
  int x = *((int *) a), y = *((int *) b);

  if(x > y) {
    return(1);
  } else if(x < y) {
    return(-1);
  }

  return(0);
}
#endif /* LOWER_BOUND == 4 */
