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
 *  $Id: heuristics.c,v 1.12 2017/03/24 11:17:24 tanaka Exp $
 *  $Revision: 1.12 $
 *  $Date: 2017/03/24 11:17:24 $
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

solution_t *heuristics(problem_t *problem, state_t *state,
		       solution_t *solution, int ub)
{
  int i, j;
  int n_block, lb;
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
    int src_stack = info[0].stack, dst_stack;
    stack_info_t current;
    block_t cblock;

    for(; info[0].n_stacked > 0; ++info[0].n_space, --info[0].n_stacked) {
#if 0
      print_state(problem, cstate, stdout);
      for(i = 0; i < problem->n_stack; ++i) {
	printf("[%d:%d:%d:%d]", info[i].stack, info[i].n_space,
	       info[i].min_priority, info[i].n_stacked);
      }
      printf("\n");
#endif
      cblock = block[src_stack][--n_tier[src_stack]];

      for(i = problem->n_stack - 1; i > 0 && info[i].n_space == 0; --i);
      if(i == 0) {
	fprintf(stderr, "No stack found.\n");
	exit(1);
      }

      if(info[i].min_priority >= cblock.priority) {
	for(j = 1; j <= i; ++j) {
	  if(info[j].n_space > 0 && info[j].min_priority >= cblock.priority) {
	    break;
	  }
	}
	i = j;

	info[i].min_priority = cblock.priority;
	info[i].n_stacked = 0;
      } else if(++lb >= ub) {
	csolution->n_relocation = ub;
	return(csolution);
      } else if(info[i].n_space == 1) {
	for(j = i - 1; j >= 1 && info[j].n_space == 0; --j);
	if(j >= 1) {
	  i = j;
	}
	++info[i].n_stacked;
      } else {
	++info[i].n_stacked;
      }

      dst_stack = info[i].stack;
      add_relocation(csolution, src_stack, dst_stack, &cblock);

      --info[i].n_space;
      block[dst_stack][n_tier[dst_stack]++] = cblock;
      bi[dst_stack][n_tier[dst_stack]].min_priority = info[i].min_priority;
      bi[dst_stack][n_tier[dst_stack]].n_stacked = info[i].n_stacked;

      current = info[i];
      if(current.n_stacked == 0) {
	for(; i > 1 && stack_info_comp((void *) &(info[i - 1]),
				       (void *) &current) > 0; --i) {
	  info[i] = info[i - 1];
	}
	info[i] = current;
      }
    }

    ++info[0].n_space;
    info[0].min_priority = bi[src_stack][--n_tier[src_stack]].min_priority;
    info[0].n_stacked = bi[src_stack][n_tier[src_stack]].n_stacked;

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
