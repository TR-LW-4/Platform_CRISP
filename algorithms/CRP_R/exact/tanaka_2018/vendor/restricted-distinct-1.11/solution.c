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
 *  $Id: solution.c,v 1.12 2017/02/11 04:45:35 tanaka Exp $
 *  $Revision: 1.12 $
 *  $Date: 2017/02/11 04:45:35 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "define.h"
#include "solution.h"

solution_t *create_solution(void)
{
  return((solution_t *) calloc(1, sizeof(solution_t)));
}

void free_solution(solution_t *solution)
{
  if(solution != NULL) {
    free(solution->relocation);
    free(solution);
  }
}

void copy_solution(solution_t *dst, solution_t *src)
{
  if(dst != NULL && src != NULL) {
    if(dst->n_block < src->n_relocation) {
      dst->n_block = src->n_block;
      dst->relocation
	= (relocation_t *) realloc((void *) dst->relocation,
				   (size_t) dst->n_block
				   *sizeof(relocation_t));
    }
    dst->n_relocation = src->n_relocation;
    memcpy((void *) dst->relocation, (void *) src->relocation,
	   (size_t) src->n_relocation*sizeof(relocation_t));
  }
}

void add_relocation(solution_t *solution, int src, int dst, block_t *block)
{
  if(solution != NULL) {
    if(solution->n_block <= solution->n_relocation) {
      solution->n_block += 100;
      solution->relocation
	= (relocation_t *) realloc((void *) solution->relocation,
				   (size_t) solution->n_block
				   *sizeof(relocation_t));
    }
    solution->relocation[solution->n_relocation].src = src;
    solution->relocation[solution->n_relocation].dst = dst;
    solution->relocation[solution->n_relocation++].block = *block;
  }
}

state_t *create_state(problem_t *problem)
{
  int i;
  state_t *state = (state_t *) malloc(sizeof(state_t));
  
  state->n_block = 0;
  state->lb1 = 0;
  state->n_tier
    = (int *) calloc((size_t) (2*problem->n_stack + problem->n_block),
		     sizeof(int));
  state->last_modified = state->n_tier + problem->n_stack;
  state->last_relocation = state->last_modified + problem->n_stack;

  state->block
    = (block_t **) malloc((size_t) problem->n_stack*sizeof(block_t *));
  state->block[0]
    = (block_t *) calloc((size_t) problem->n_stack*problem->s_height,
			 sizeof(block_t));
  state->bi
    = (block_info_t **) malloc((size_t) problem->n_stack
			       *sizeof(block_info_t *));
  state->bi[0]
    = (block_info_t *) malloc((size_t) problem->n_stack*(problem->s_height + 1)
			      *sizeof(block_info_t));
  for(i = 1; i < problem->n_stack; ++i) {
    state->block[i] = state->block[i - 1] + problem->s_height;
    state->bi[i] = state->bi[i - 1] + (problem->s_height + 1);
  }
  state->info
    = (stack_info_t *) calloc((size_t) problem->n_stack, sizeof(stack_info_t));

  return(state);
}

state_t *initialize_state(problem_t *problem, state_t *state)
{
  int i, j;
  state_t *nstate = (state == NULL)?create_state(problem):state;

  nstate->n_block = problem->n_block;
  nstate->lb1 = 0;

  memcpy((void *) nstate->n_tier, (void *) problem->n_tier,
	 (size_t) problem->n_stack*sizeof(int));
  memcpy((void *) nstate->block[0], (void *) problem->block[0],
	 (size_t) problem->n_stack*problem->s_height*sizeof(block_t));

  for(i = 0; i < problem->n_stack; ++i) {
    stack_info_t *cinfo = nstate->info + i;
    block_info_t *bi = nstate->bi[i];
    cinfo->stack = i;
    cinfo->min_priority = problem->max_priority + 1;
    cinfo->n_space = problem->s_height - problem->n_tier[i];
    bi[0].min_priority = problem->max_priority + 1;
    bi[0].n_stacked = 0;
    for(j = 0; j < problem->n_tier[i]; ++j) {
      if(cinfo->min_priority >= problem->block[i][j].priority) {
	cinfo->min_priority = problem->block[i][j].priority;
	bi[j + 1].min_priority = cinfo->min_priority;
	bi[j + 1].n_stacked = 0;
      } else {
	bi[j + 1].min_priority = bi[j].min_priority;
	bi[j + 1].n_stacked = bi[j].n_stacked + 1;
	++nstate->lb1;
      }
    }
    cinfo->n_stacked = bi[nstate->n_tier[i]].n_stacked;
  }

  qsort((void *) nstate->info, problem->n_stack, sizeof(stack_info_t),
	stack_info_comp);

  return(nstate);
}

void copy_state(problem_t *problem, state_t *dst, state_t *src)
{
  dst->n_block = src->n_block;
  dst->lb1 = src->lb1;

  memcpy((void *) dst->n_tier, (void *) src->n_tier,
	 (size_t) (2*problem->n_stack + problem->n_block)*sizeof(int));
  memcpy((void *) dst->block[0], (void *) src->block[0],
	 (size_t) problem->n_stack*problem->s_height*sizeof(block_t));
  memcpy((void *) dst->bi[0], (void *) src->bi[0],
	 (size_t) problem->n_stack*(problem->s_height + 1)
	 *sizeof(block_info_t));
  memcpy((void *) dst->info, (void *) src->info,
	 (size_t) problem->n_stack*sizeof(stack_info_t));
}

state_t *duplicate_state(problem_t *problem, state_t *state)
{
  state_t *nstate = create_state(problem);
  copy_state(problem, nstate, state);
  return(nstate);
}

void free_state(state_t *state)
{
  if(state != NULL) {
    free(state->info);
    free(state->bi[0]);
    free(state->bi);
    free(state->block[0]);
    free(state->block);
    free(state->n_tier);
    free(state);
  }
}

uchar retrieve_all_blocks(problem_t *problem, state_t *state,
			  solution_t *solution)
{
  int i;
  int *n_tier = state->n_tier;
  stack_info_t *info = state->info, current;

  while(state->n_block > 0 && info[0].n_stacked == 0) {
    int src_stack = info[0].stack;
    int min_priority
      = state->bi[src_stack][--n_tier[src_stack]].min_priority;

    if(solution != NULL) {
      int lv
	= state->last_relocation[state->block[src_stack][n_tier[src_stack]].no];

      if(lv > 0) {
	int j;

	for(j = problem->n_stack - 1; 
	    j > 0 && info[j].min_priority > min_priority; --j) {
	  if(info[j].n_space > info[0].n_space
	     && state->last_modified[info[j].stack] < lv) {
	    return(True);
	  }
	}
	for(; j > 0; --j) {
	  if(info[j].n_space > info[0].n_space + 1
	     && state->last_modified[info[j].stack] < lv) {
	    return(True);
	  }
	}
      }

      state->last_modified[src_stack] = solution->n_relocation;
    }

    --state->n_block;
    ++info[0].n_space;
    info[0].min_priority = min_priority;
    info[0].n_stacked = state->bi[src_stack][n_tier[src_stack]].n_stacked;

    current = info[0];
    for(i = 0; i < problem->n_stack - 1
	  && stack_info_comp((void *) &(info[i + 1]), (void *) &current) < 0;
	++i) {
      info[i] = info[i + 1];
    }
    info[i] = current;
  }

  return(False);
}

int stack_info_comp(const void *a, const void *b)
{
  stack_info_t *x = (stack_info_t *) a;
  stack_info_t *y = (stack_info_t *) b;

  if(x->min_priority > y->min_priority) {
    return(1);
  } else if(x->min_priority < y->min_priority) {
    return(-1);
  }

  return(0);
}
