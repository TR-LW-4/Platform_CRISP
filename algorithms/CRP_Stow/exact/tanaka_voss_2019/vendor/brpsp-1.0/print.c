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
 *  $Id: print.c,v 1.2 2019/06/11 12:13:41 tanaka Exp tanaka $
 *  $Revision: 1.2 $
 *  $Date: 2019/06/11 12:13:41 $
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
  int max_yard_tier = 0, max_ship_tier = 0;
  char empty_slot[] = "                               ";
  char stack_label_format[32];

  empty_slot[problem->name_length] = '\0';
#ifdef _MSC_VER
  _snprintf_s(stack_label_format, 32, _TRUNCATE, "%%%dd ",
	      problem->name_length + 1);
#else /* _MSC_VER */
  snprintf(stack_label_format, 32, "%%%dd ", problem->name_length + 1);
#endif /* _MSC_VER */

  fprintf(fp, "yard_stacks=%d, yard_s_height=%d, yard_blocks=%d\n",
	  problem->yard_n_stack, problem->yard_s_height,
	  problem->yard_n_block);
  fprintf(fp, "ship_stacks=%d, ship_blocks=%d\n",
	  problem->ship_n_stack, problem->ship_n_block);

  if(problem->yard_s_height < problem->yard_n_block) {
    max_yard_tier = problem->yard_s_height;
  } else {
    for(i = 0; i < problem->yard_n_stack; ++i) {
      max_yard_tier = max(max_yard_tier, problem->yard_n_tier[i]);
    }
  }
  for(i = 0; i < problem->ship_n_stack; ++i) {
    max_ship_tier = max(max_ship_tier, problem->ship_n_tier[i]);
  }

  for(j = max(max_yard_tier, max_ship_tier) - 1; j >= 0; --j) {
    if(j < problem->yard_s_height) {
      fprintf(fp, "%2d:", j + 1);
    } else {
      fprintf(fp, "   ");
    }

    for(i = 0; i < problem->yard_n_stack; ++i) {
      if(j < problem->yard_n_tier[i]) {
	fprintf(fp, "[%s]", problem->name[problem->yard_block[i][j]]);
      } else if(j < problem->yard_s_height) {
	fprintf(fp, "[%s]", empty_slot);
      } else {
	fprintf(fp, " %s ", empty_slot);
      }
    }
    fprintf(fp, "  ");

    for(i = 0; i < problem->ship_n_stack; ++i) {
#if 1
      if(j >= max_ship_tier) {
	fprintf(fp, " %s ", empty_slot);
      } else if(j >= max_ship_tier - problem->ship_n_tier[i]) {
	fprintf(fp, "[%s]",
		problem->name[problem->ship_block[i]
			      [j + problem->ship_n_tier[i] - max_ship_tier]]);
      } else {
	fprintf(fp, "[%s]", empty_slot);
      }
#else
      if(j < problem->ship_n_tier[i]) {
	fprintf(fp, "[%s]", problem->name[problem->ship_block[i][j]]);
      } else {
	fprintf(fp, "[%s]", empty_slot);
      }
#endif
    }

    if(j < max_ship_tier) {
      fprintf(fp, ":%2d\n", j + 1);
    } else {
      fprintf(fp, "\n");
    }
  }

  fprintf(fp, "   ");
  for(i = 0; i < problem->yard_n_stack; ++i) {
    fprintf(fp, (const char *) stack_label_format, i + 1);
  }
  fprintf(fp, "  ");
  for(i = 0; i < problem->ship_n_stack; ++i) {
    fprintf(fp, (const char *) stack_label_format, i + 1);
  }
  fprintf(fp, "\n");
}

void print_state(problem_t *problem, state_t *state, FILE *fp)
{
  int i, j;
  int max_yard_tier = 0, max_ship_tier = 0;
  char empty_slot[] = "                               ";
  char stack_label_format[32];

  empty_slot[problem->name_length] = '\0';
#ifdef _MSC_VER
  _snprintf_s(stack_label_format, 32,
	      _TRUNCATE, "%%%dd ", problem->name_length + 1);
#else /* _MSC_VER */
  snprintf(stack_label_format, 32, "%%%dd ", problem->name_length + 1);
#endif /* _MSC_VER */

  for(i = 0; i < problem->yard_n_stack; ++i) {
    max_yard_tier = max(max_yard_tier, state->ystate[i].n_tier);
  }
  for(i = 0; i < problem->ship_n_stack; ++i) {
    max_ship_tier = max(max_ship_tier, problem->ship_n_tier[i]);
  }

  for(j = max(max_yard_tier, max_ship_tier) - 1; j >= 0; --j) {
    if(j < problem->yard_s_height) {
      fprintf(fp, "%2d:", j + 1);
    } else {
      fprintf(fp, "   ");
    }

    for(i = 0; i < problem->yard_n_stack; ++i) {
      if(j < state->ystate[i].n_tier) {
	if(state->ship_n_tier[problem->ship_position
			      [state->yard_block[i][j].no].s]
	   == problem->ship_position[state->yard_block[i][j].no].t) {
	  fprintf(fp, "<%s>", problem->name[state->yard_block[i][j].no]);
	} else {
	  fprintf(fp, "[%s]", problem->name[state->yard_block[i][j].no]);
	}
      } else {
	fprintf(fp, " %s ", empty_slot);
      }
    }
    fprintf(fp, "  ");

    for(i = 0; i < problem->ship_n_stack; ++i) {
#if 1
      if(j >= max_ship_tier - problem->ship_n_tier[i] + state->ship_n_tier[i]) {
	fprintf(fp, " %s ", empty_slot);
      } else if(j >= max_ship_tier - problem->ship_n_tier[i]) {
	fprintf(fp, "[%s]",
		problem->name[problem->ship_block[i]
			      [j - max_ship_tier + problem->ship_n_tier[i]]]);
      } else {
	fprintf(fp, "[%s]", empty_slot);
      }
#else
      if(j < state->ship_n_tier[i]) {
	fprintf(fp, "[%s]", problem->name[problem->ship_block[i][j]]);
      } else {
	fprintf(fp, " %s ", empty_slot);
      }
#endif
    }

    if(j < max_ship_tier) {
      fprintf(fp, ":%2d\n", j + 1);
    } else {
      fprintf(fp, "\n");
    }
  }

  fprintf(fp, "   ");
  for(i = 0; i < problem->yard_n_stack; ++i) {
    fprintf(fp, (const char *) stack_label_format, i + 1);
  }
  fprintf(fp, "  ");
  for(i = 0; i < problem->ship_n_stack; ++i) {
    fprintf(fp, (const char *) stack_label_format, i + 1);
  }
  fprintf(fp, "\n");

#if 0
  for(i = 0; i < problem->yard_n_stack; ++i) {
    printf("[%d,%d]", state->ystate[i].n_target, state->ystate[i].n_stacked);
  }
  printf("\n");
#endif
}

void print_solution(problem_t *problem, solution_t *solution, FILE *fp)
{
  int iter, k;
  int ship_n_block;
  int src_stack, dst_stack;
  state_t *state = initialize_state(problem, NULL);

  fprintf(fp, "========\nInitial layout\n");
  print_state(problem, state, fp);

  ship_n_block = state->ship_n_block;
  move_all_blocks(problem, state);

  if(state->ship_n_block > ship_n_block) {
    if(state->ship_n_block - ship_n_block > 1) {
      fprintf(fp, "++++++++\nMove %d blocks\n",
	      state->ship_n_block - ship_n_block);
    } else {
      fprintf(fp, "++++++++\nMove 1 block\n");
    }
    print_state(problem, state, fp);
    ship_n_block = state->ship_n_block;
  }

  for(iter = 0; iter < solution->n_relocation; ++iter) {
    src_stack = solution->relocation[iter].src;
    dst_stack = solution->relocation[iter].dst;
    k = state->yard_block[src_stack][--state->ystate[src_stack].n_tier].no;

    state->yard_position[k].s = dst_stack;
    state->yard_position[k].t = state->ystate[dst_stack].n_tier;

    state->yard_block[dst_stack][state->ystate[dst_stack].n_tier++].no = k;

    fprintf(fp, "--------\n");
    fprintf(fp, "Relocation %d: [%s] %d->%d\n", iter + 1,
	    problem->name[k], src_stack + 1, dst_stack + 1);

    --state->ystate[src_stack].n_stacked;
    ++state->ystate[dst_stack].n_stacked;

    print_state(problem, state, fp);

    if(state->ystate[src_stack].n_stacked == 0) {
      ship_n_block = state->ship_n_block;
      move_all_blocks(problem, state);
      if(state->ship_n_block - ship_n_block > 1) {
	fprintf(fp, "++++++++\nMove %d blocks\n",
		state->ship_n_block - ship_n_block);
      } else {
	fprintf(fp, "++++++++\nMove 1 block\n");
      }
      print_state(problem, state, fp);
    }
  }
  fprintf(fp, "--------\n");

  if(state->ship_n_block != problem->ship_n_block) {
    fprintf(stderr, "Invalid solution.\n");
    printf("%d!=%d\n",
	   state->ship_n_block, problem->ship_n_block);
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
    fprintf(fp, "[%s:%d=>%d]", 
	    problem->name[solution->relocation[i].no],
	    solution->relocation[i].src + 1,
	    solution->relocation[i].dst + 1);
  }
  fprintf(fp, "\n");
}

void print_time(problem_t *problem)
{
  set_time(problem);
  fprintf(stderr, "time=%.2f\n", problem->time);
}
