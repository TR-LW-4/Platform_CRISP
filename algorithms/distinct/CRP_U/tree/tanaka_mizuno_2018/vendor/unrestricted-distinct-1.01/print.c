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
 *  $Id: print.c,v 1.15 2018/06/19 07:22:43 tanaka Exp tanaka $
 *  $Revision: 1.15 $
 *  $Date: 2018/06/19 07:22:43 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "define.h"
#include "print.h"
#include "problem.h"
#include "solution.h"

void print_problem(problem_t *problem, FILE *fp)
{
  int i, j;
  int max_tier;

  fprintf(fp, "stacks=%d, s_height=%d, blocks=%d\n", problem->n_stack, 
	  problem->s_height, problem->n_block);

  max_tier = 0;
  for(i = 0; i < problem->n_stack; ++i) {
    max_tier = max(max_tier, problem->n_tier[i]);
  }

  for(j = max_tier - 1; j >= 0; --j) {
    fprintf(fp, "%3d:", j + 1);
    for(i = 0; i < problem->n_stack; ++i) {
      if(j < problem->n_tier[i]) {
	fprintf(fp, "[%3d]", problem->priority[problem->block[i][j].no]);
      } else {
	fprintf(fp, "     ");
      }
    }
    fprintf(fp, "\n");
  }
}

void print_state(problem_t *problem, state_t *state, FILE *fp)
{
  int i, j;
  int max_tier = 0;

  for(i = 0; i < problem->n_stack; ++i) {
    max_tier = max(max_tier, state->n_tier[i]);
  }
  for(j = max_tier - 1; j >= 0; --j) {
    fprintf(fp, "%3d:", j + 1);
    for(i = 0; i < problem->n_stack; ++i) {
      if(j < state->n_tier[i]) {
	fprintf(fp, "[%3d]", problem->priority[state->block[i][j].no]);
      } else {
	fprintf(fp, "     ");
      }
    }
    fprintf(fp, "\n");
  }
}

void print_solution(problem_t *problem, solution_t *solution, FILE *fp)
{
  int iter, i;
  int src_stack, dst_stack;
  int n_block;
  block_t reloc_block;
  state_t *state = initialize_state(problem, NULL);
  stack_info_t *info = state->info, current;

  fprintf(fp, "========\nInitial configuration\n");
  print_state(problem, state, fp);

  n_block = state->n_block;
  retrieve_all_blocks(problem, state, NULL);
  if(state->n_block < n_block) {
    if(n_block - state->n_block > 1) {
      fprintf(fp, "++++++++\nRetrieve %d blocks\n", n_block - state->n_block);
    } else {
      fprintf(fp, "++++++++\nRetrieve 1 block\n");
    }
    print_state(problem, state, fp);
    n_block = state->n_block;
  }

  for(iter = 0; iter < solution->n_relocation; ++iter) {
    src_stack = solution->relocation[iter].src;
    dst_stack = solution->relocation[iter].dst;
    reloc_block = state->block[src_stack][--state->n_tier[src_stack]];
    state->block[dst_stack][state->n_tier[dst_stack]++] = reloc_block;

    if(solution->relocation[iter].block.no != reloc_block.no) {
      fprintf(stderr, "Item mismatch in solution %d!=%d.\n,",
	      solution->relocation[iter].block.no, reloc_block.no);
      exit(1);
    }

    fprintf(fp, "--------\n");
    fprintf(fp, "Relocation %d: [%2d] %d->%d\n", iter + 1,
	    problem->priority[reloc_block.no], src_stack + 1, dst_stack + 1);

    for(i = 0; i < problem->n_stack && info[i].stack != src_stack; ++i);

    if(info[i].min_priority < reloc_block.priority) {
      --state->lb1;
    }
    info[i].min_priority
      = state->bi[src_stack][state->n_tier[src_stack]].min_priority;
    info[i].n_stacked
      = state->bi[src_stack][state->n_tier[src_stack]].n_stacked;
    ++info[i].n_space;

    current = info[i];
    for(; i < problem->n_stack - 1
	  && stack_info_comp((void *) &(info[i + 1]),
			     (void *) &current) < 0; ++i) {
      info[i] = info[i + 1];
    }
    info[i] = current;

    for(i = 0; i < problem->n_stack && info[i].stack != dst_stack; ++i);

    --info[i].n_space;
    if(info[i].min_priority >= reloc_block.priority) {
      info[i].min_priority = reloc_block.priority;
      info[i].n_stacked = 0;
      state->bi[dst_stack][state->n_tier[dst_stack]].min_priority
	= reloc_block.priority;
      state->bi[dst_stack][state->n_tier[dst_stack]].n_stacked = 0;

      current = info[i];
      for(; i > 1 && stack_info_comp((void *) &(info[i - 1]),
				     (void *) &current) > 0; --i) {
	info[i] = info[i - 1];
      }
      info[i] = current;
    } else {
      ++state->lb1;
      ++info[i].n_stacked;
      state->bi[dst_stack][state->n_tier[dst_stack]].min_priority
	= info[i].min_priority;
      state->bi[dst_stack][state->n_tier[dst_stack]].n_stacked
	= info[i].n_stacked;
      current = info[i];
      for(; i < problem->n_stack - 1
	    && stack_info_comp((void *) &(info[i + 1]),
			       (void *) &current) < 0; ++i) {
	info[i] = info[i + 1];
      }
      info[i] = current;
    }
    print_state(problem, state, fp);

    if(info[0].n_stacked == 0) {
      retrieve_all_blocks(problem, state, NULL);
      if(n_block - state->n_block > 1) {
	fprintf(fp, "++++++++\nRetrieve %d blocks\n",
		n_block - state->n_block);
      } else {
	fprintf(fp, "++++++++\nRetrieve 1 block\n");
      }
      print_state(problem, state, fp);
      n_block = state->n_block;
    }
  }
  fprintf(fp, "--------\n");

  if(state->n_block != 0) {
    fprintf(stderr, "Invalid solution.\n");
    exit(1);
  }

  fprintf(fp, "relocations=%d\n", solution->n_relocation);

  free_state(state);
}

void print_solution_relocation(problem_t *problem, solution_t *solution,
			       FILE *fp)
{
  int i;

  for(i = 0; i < solution->n_relocation; ++i) {
    fprintf(fp, "[%d=>%d(%d)]", solution->relocation[i].src + 1,
	    solution->relocation[i].dst + 1,
	    problem->priority[solution->relocation[i].block.no]);
  }
  fprintf(fp, "\n");
}

void print_time(problem_t *problem, FILE *fp)
{
  set_time(problem);
  fprintf(fp, "time=%.2f\n", problem->time);
}
