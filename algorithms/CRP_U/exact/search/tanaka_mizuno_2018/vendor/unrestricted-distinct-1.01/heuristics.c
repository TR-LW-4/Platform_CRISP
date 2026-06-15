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
 *  $Id: heuristics.c,v 1.7 2017/03/24 11:17:13 tanaka Exp $
 *  $Revision: 1.7 $
 *  $Date: 2017/03/24 11:17:13 $
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
#define PU2

solution_t *heuristics(problem_t *problem, state_t *state,
		       solution_t *solution, int ub)
{
  int i, j, k;
  int n_block;
  int lb;
  solution_t *csolution = solution;
  static state_t *cstate = NULL;
  int *n_tier;
  block_t **block;
  block_info_t **bi;
  stack_info_t *info;

#if 0
  printf("heuristics\n");
#endif

  if(problem == NULL) {
    if(cstate != NULL) {
      free_state(cstate);
      cstate = NULL;
    }
    return(NULL);
  }

  if(cstate == NULL) {
    cstate = duplicate_state(problem, state);
  } else {
    copy_state(problem, cstate, state);
  }
  n_tier = cstate->n_tier;
  block = cstate->block;
  bi = cstate->bi;
  info = cstate->info;

  if(csolution == NULL) {
    csolution = create_solution();
  }

  lb = state->lb1 + csolution->n_relocation;

  for(n_block = state->n_block; n_block > 0; --n_block) {
    int src_stack, dst_stack;
    stack_info_t current;
    block_t cblock;

    /* each block above the target block is relocated to another stack */
    /* one by one from the top to the bottom */
    while(info[0].n_stacked > 0) {
#if 0
      print_state(problem, cstate, stdout);
      for(i = 0; i < problem->n_stack; ++i) {
	printf("[%d:%d:%d:%d]", info[i].stack, info[i].min_priority,
	       info[i].n_space, info[i].n_stacked);
      }
      printf("\n");
#endif

      /* default source stack is info[0] */
      k = 0;
      src_stack = info[0].stack;

      /* the block to be relocated */
      cblock = block[src_stack][n_tier[src_stack] - 1];

      /* skip the stack without space */
      for(i = problem->n_stack - 1; i > 0 && info[i].n_space == 0; --i);
      if(i == 0) {
	fprintf(stderr, "No stack found.\n");
	exit(1);
      }

      /* determine the destination stack */
      /* info[i].min_priority: highest (smallest) priority of the stacks */
      if(info[i].min_priority >= cblock.priority) {
	/* the block can be relocated without making it a blocking block */

	/* the stack with the highest priority satisfying */
	/*   (the stack priority)>=(the block priority)   */
	/* is chosen as the destination stack             */
	for(i = 1;
	    info[i].n_space == 0 || info[i].min_priority < cblock.priority;
	    ++i);

	/* update min_priority of the destination stack */
	info[i].min_priority = cblock.priority;

	/* the highest (smallest) priority block is on the top of the stack */
	info[i].n_stacked = 0;
      } else if(++lb >= ub) {
	csolution->n_relocation = ub;
	return(csolution);
      } else {
	/* the block becomes a blocking block after relocation             */
#ifdef PU2
	/* Scan stacks such that                                           */
        /*  (1) the block with the highest priority is the topmost block,  */
        /*  (2) it is relocatable without making it a blocking block       */
        /*      after relocation,                                          */
        /*  (3) the topmost block on the stack with the highest priority   */
        /*      is relocatable without making it a blocking block after    */
        /*      relocation.                                                */
        /* Ties are broken by:                                             */
	/*   (1) the stack with a greater number of blocks is chosen first,*/
	/*   (2) the block with the lowest priority is chosen first.       */

	for(j = i - 1; j >= 1; --j) {
	  if(info[j].n_stacked == 0
	     && bi[info[j].stack][n_tier[info[j].stack] - 1].min_priority
	     >= cblock.priority) {
	    /* candidate found */
	    if(k == 0 || info[j].n_space < info[k].n_space) {
	      k = j;
	    }
	  }
	}
#endif
	if(k > 0) {
	  /* the block to be relocated */
	  src_stack = info[k].stack;
	  cblock = block[src_stack][n_tier[src_stack] - 1];

	  /* destination stack */
	  for(i = k + 1; info[i].n_space == 0; ++i);

	  info[i].n_stacked = 0;
	  info[i].min_priority = cblock.priority;
	} else if(info[i].n_space == 1) {
	  /* only the space for one block is left */
	  /* in this case, the second best stack is chosen */
	  for(j = i - 1; j >= 1; --j) {
	    if(info[j].n_space > 0) {
	      i = j;
	      break;
	    }
	  }
	  ++info[i].n_stacked;
	} else {
	  /* the block always becomes a blocking block after relocation */
	  /* the stack with the lowest (largest) priority is chosen */

	  /* the number of blocks stacked above the highest (smallest) */
	  /* priority block increases by 1 */

	  ++info[i].n_stacked;
	}
      }

      /* decrease the number of blocks in the source stack */
      ++info[k].n_space;
      info[k].min_priority = bi[src_stack][--n_tier[src_stack]].min_priority;
      info[k].n_stacked = bi[src_stack][n_tier[src_stack]].n_stacked;

      /* destination stack */
      dst_stack = info[i].stack;

      /* update the partial solution */
      add_relocation(csolution, src_stack, dst_stack, &cblock);

      /* decrease the free space of the destination stack */
      --info[i].n_space;

      /* update the table for incremental computation of */
      /* min_priority and n_stacked */
      block[dst_stack][n_tier[dst_stack]++] = cblock;
      bi[dst_stack][n_tier[dst_stack]].min_priority = info[i].min_priority;
      bi[dst_stack][n_tier[dst_stack]].n_stacked = info[i].n_stacked;

      if(k != 0) {
	/* first, swap the two info's */
	/* it is because the priority of the destination stack is */
	/* equal to the source stack */
	current = info[k];
	info[k] = info[i];
	info[i] = current;

	if(info[i - 1].min_priority > info[i].min_priority) {
	  for(; i > 1 && stack_info_comp((void *) &(info[i - 1]),
					 (void *) &current) > 0; --i) {
	    info[i] = info[i - 1];
	  }
	} else {
	  for(; i < problem->n_stack - 1
		&& stack_info_comp((void *) &(info[i + 1]),
				   (void *) &current) < 0; ++i) {
	    info[i] = info[i + 1];
	  }
	}
	info[i] = current;
      } else {
	/* insertion sort of stacks */
	current = info[i];
	if(current.n_stacked == 0) {
	  /* sort is necessary only when the block does not become a */
	  /* blocking block after relocation because the stack priority */
	  /* does not change */
	  for(; i > 1 && stack_info_comp((void *) &(info[i - 1]),
					 (void *) &current) > 0; --i) {
	    info[i] = info[i - 1];
	  }
	  info[i] = current;
	}
      }
    }

    /* remove the target block and increase the free space of the target */
    /* stack */
    ++info[0].n_space;

    /* update min_priority and n_stacked by the table */
    src_stack = info[0].stack;
    info[0].min_priority = bi[src_stack][--n_tier[src_stack]].min_priority;
    info[0].n_stacked = bi[src_stack][n_tier[src_stack]].n_stacked;

    /* insertion sort so that the target stack is always numbered 0 */
    current = info[0];
    for(i = 0; i < problem->n_stack - 1
	  && stack_info_comp((void *) &(info[i + 1]), (void *) &current) < 0;
	++i) {
      info[i] = info[i + 1];
    }
    info[i] = current;
#if 0
    print_state(problem, cstate, stdout);

    for(i = 0; i < problem->n_stack; ++i) {
      printf("[%d:%d:%d:%d]", info[i].stack, info[i].n_space,
	     info[i].min_priority, info[i].n_stacked);
    }
    printf("\n");
#endif
  }

  return(csolution);
}
