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
 *  $Id: heuristics.c,v 1.3 2015/06/11 03:26:15 tanaka Exp tanaka $
 *  $Revision: 1.3 $
 *  $Date: 2015/06/11 03:26:15 $
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
		       solution_t *solution, uchar status)
{
  int i, j;
  int n_block, total_space;
  state_t *nstate;
  solution_t *csolution = (solution == NULL)?create_solution():solution;
  int *n_tier;
  block_t **block;
  block_info_t **bi;
  stack_info_t *info;

#if 0
  printf("heuristics\n");
#endif

  total_space = problem->s_height*problem->n_stack - state->n_block;
  if(status == TRUE) {
    if(total_space < state->info[0].n_space + state->info[0].n_stacked) {
      csolution->n_relocation = 10000;
      return(csolution);
    }
  }

  nstate = duplicate_state(problem, state);
  n_tier = nstate->n_tier;
  block = nstate->block;
  bi = nstate->bi;
  info = nstate->info;

  for(n_block = nstate->n_block; n_block > 0; n_block--) {
    int src_stack, dst_stack;
    stack_info_t current;
    block_t cblock;

    if(total_space < info[0].n_space + info[0].n_stacked) {
      if(status == TRUE) {
	csolution->n_relocation = 10000;
	free(nstate);
	return(csolution);
      }

      for(i = 1; i < problem->n_stack
	    && info[i].min_priority == info[0].min_priority; i++) {
	if(total_space >= info[i].n_space + info[i].n_stacked) {
	  break;
	}
      }
      if(i == problem->n_stack
	 || info[i].min_priority != info[0].min_priority) {
	csolution->n_relocation = 10000;
	free(nstate);
	return(csolution);
      }

      current = info[i];
      for(; i > 0; i--) {
	info[i] = info[i - 1];
      }
      info[0] = current;
    }

    status = FALSE;
    src_stack = info[0].stack;
    for(; info[0].n_stacked > 0; info[0].n_space++, info[0].n_stacked--) {
#if 0
      print_state(problem, nstate, stdout);
      for(i = 0; i < problem->n_stack; i++) {
	printf("[%d:%d:%d:%d]", info[i].stack, info[i].n_space,
	       info[i].min_priority, info[i].n_stacked);
      }
      printf("\n");
#endif
      cblock = block[src_stack][--n_tier[src_stack]];

      for(i = problem->n_stack - 1; i > 0 && info[i].n_space == 0; i--);
      if(i == 0) {
	fprintf(stderr, "No stack found.\n");
	exit(1);
      }

      if(info[i].min_priority >= cblock.priority) {
	for(j = 1; j <= i; j++) {
	  if(info[j].n_space > 0 && info[j].min_priority >= cblock.priority) {
	    break;
	  }
	}
	i = j;

	if(info[i].min_priority > cblock.priority) {
	  info[i].min_priority2 = info[i].min_priority;
	  info[i].min_priority = cblock.priority;
	}
	info[i].n_stacked = 0;
#if 1
      } else if(info[i].n_space == 1) {
	for(j = i - 1; j >= 1 && info[j].n_space == 0; j--);
	if(j >= 1) {
	  i = j;
	}
	info[i].n_stacked++;
#endif
      } else {
	info[i].n_stacked++;
      }

      dst_stack = info[i].stack;
      add_relocation(csolution, src_stack, dst_stack, cblock.no);

      info[i].n_space--;
      block[dst_stack][n_tier[dst_stack]++] = cblock;
      bi[dst_stack][n_tier[dst_stack]].min_priority = info[i].min_priority;
      bi[dst_stack][n_tier[dst_stack]].min_priority2 = info[i].min_priority2;
      bi[dst_stack][n_tier[dst_stack]].n_stacked = info[i].n_stacked;

      current = info[i];
      if(current.n_stacked == 0) {
	for(; i > 1 && stack_info_comp((void *) &(info[i - 1]),
				       (void *) &current) > 0; i--) {
	  info[i] = info[i - 1];
	}
      } else {
	for(; i < problem->n_stack - 1
	      && stack_info_comp((void *) &(info[i + 1]),
				 (void *) &current) < 0; i++) {
	  info[i] = info[i + 1];
	}
      }
      info[i] = current;
    }

    total_space++;
    info[0].n_space++;
    info[0].min_priority = bi[src_stack][--n_tier[src_stack]].min_priority;
    info[0].min_priority2 = bi[src_stack][n_tier[src_stack]].min_priority2;
    info[0].n_stacked = bi[src_stack][n_tier[src_stack]].n_stacked;

    current = info[0];
    for(i = 0; i < problem->n_stack - 1
	  && stack_info_comp((void *) &(info[i + 1]), (void *) &current) < 0;
	i++) {
      info[i] = info[i + 1];
    }
    info[i] = current;
#if 0
    print_state(problem, nstate, stdout);

    for(i = 0; i < problem->n_stack; i++) {
      printf("[%d:%d:%d:%d]", info[i].stack, info[i].n_space,
	     info[i].min_priority, info[i].n_stacked);
    }
    printf("\n");
#endif
  }

  free_state(nstate);

  return(csolution);
}

#if 0
solution_t *heuristics(problem_t *problem, state_t *state,
		       solution_t *solution, uchar status)
{
  int i, j;
  int n_block, total_space;
  state_t *nstate;
  solution_t *csolution = (solution == NULL)?create_solution():solution;
  int *n_tier;
  block_t **block;
  block_info_t **bi;
  stack_info_t *info;

#if 0
  printf("heuristics\n");
#endif

  total_space = problem->s_height*problem->n_stack - state->n_block;
  if(status == TRUE) {
    if(total_space < state->info[0].n_space + state->info[0].n_stacked) {
      csolution->n_relocation = 10000;
      return(csolution);
    }
  }

  nstate = duplicate_state(problem, state);
  n_tier = nstate->n_tier;
  block = nstate->block;
  bi = nstate->bi;
  info = nstate->info;

  for(n_block = state->n_block; n_block > 0;) {
    int src_stack, dst_stack;
    stack_info_t current;
    block_t cblock;
    int mm = problem->max_priority + 2, mi = -1;
    int max_priority;

    if(status == FALSE) {
      for(i = 0; i < problem->n_stack
	    && info[i].min_priority == info[0].min_priority; i++) {
	if(total_space < info[i].n_space + info[i].n_stacked) {
	  continue;
	}
	max_priority = 0;
	for(j = 0; j < info[i].n_stacked; j++) {
	  max_priority
	    = max(max_priority,
		  block[info[i].stack][n_tier[info[i].stack] - 1 - j].priority);
	}
	if(mm > max_priority) {
	  mm = max_priority;
	  mi = i;
	}
      }

      if(mi < 0) {
	csolution->n_relocation = 10000;
	free(nstate);
	return(csolution);
      }

      current = info[mi];
      for(i = mi; i > 0; i--) {
	info[i] = info[i - 1];
      }
      info[0] = current;
    } else if(total_space < info[0].n_space + info[0].n_stacked) {
      csolution->n_relocation = 10000;
      free(nstate);
      return(csolution);
    }

    status = FALSE;
    src_stack = info[0].stack;
    for(; info[0].n_stacked > 0; info[0].n_space++, info[0].n_stacked--) {
#if 0
      print_state(problem, nstate, stdout);
      for(i = 0; i < problem->n_stack; i++) {
	printf("[%d:%d:%d:%d]", info[i].stack, info[i].n_space,
	       info[i].min_priority, info[i].n_stacked);
      }
      printf("\n");
#endif
      cblock = block[src_stack][--n_tier[src_stack]];

      for(i = problem->n_stack - 1; i > 0 && info[i].n_space == 0; i--);
      if(i == 0) {
	fprintf(stderr, "No space found\n");
	exit(1);
      }

      if(info[i].min_priority >= cblock.priority) {
	for(j = i - 1; j > 0 && info[j].min_priority >= cblock.priority; j--);
	for(i = j + 1; info[i].n_space == 0; i++);

	info[i].min_priority = cblock.priority;
	info[i].n_stacked = 0;
#if 1
      } else if(info[i].n_space == 1) {
	for(j = i - 1; j >= 1 && info[j].n_space == 0; j--);
	if(j >= 1) {
	  i = j;
	}
	info[i].n_stacked++;
#endif
      } else {
	info[i].n_stacked++;
      }

      dst_stack = info[i].stack;
      add_relocation(csolution, src_stack, dst_stack, cblock.no);

      info[i].n_space--;
      block[dst_stack][n_tier[dst_stack]++] = cblock;
      bi[dst_stack][n_tier[dst_stack]].min_priority = info[i].min_priority;
      bi[dst_stack][n_tier[dst_stack]].n_stacked = info[i].n_stacked;

      current = info[i];
      if(current.n_stacked == 0) {
	for(; i > 1 && stack_info_comp((void *) &(info[i - 1]),
				       (void *) &current) > 0; i--) {
	  info[i] = info[i - 1];
	}
      } else {
	for(; i < problem->n_stack - 1
	      && stack_info_comp((void *) &(info[i + 1]),
				 (void *) &current) < 0; i++) {
	  info[i] = info[i + 1];
	}
      }
      info[i] = current;
    }

    while(info[0].n_stacked == 0) {
      src_stack = info[0].stack;
      n_block--;
      total_space++;
      info[0].n_space++;
      info[0].min_priority = bi[src_stack][--n_tier[src_stack]].min_priority;
      info[0].n_stacked = bi[src_stack][n_tier[src_stack]].n_stacked;

      current = info[0];
      for(i = 0; i < problem->n_stack - 1
	    && stack_info_comp((void *) &(info[i + 1]), (void *) &current) < 0;
	  i++) {
	info[i] = info[i + 1];
      }
      info[i] = current;
    }

#if 0
    print_state(problem, nstate, stdout);

    for(i = 0; i < problem->n_stack; i++) {
      printf("[%d:%d:%d:%d]", info[i].stack, info[i].n_space,
	     info[i].min_priority, info[i].n_stacked);
    }
    printf("\n");
#endif
  }

  free_state(nstate);

  return(csolution);
}
#endif
