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
 *  $Id: solution.c,v 1.5 2019/06/11 12:13:41 tanaka Exp tanaka $
 *  $Revision: 1.5 $
 *  $Date: 2019/06/11 12:13:41 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "define.h"
#include "solution.h"

static void check_blocking4(problem_t *, state_t *);
static void move_all_blocks_sub(problem_t *, state_t *, int);

solution_t *create_solution(void)
{
  return((solution_t *) calloc(1, sizeof(solution_t)));
}

void free_solution(solution_t *solution)
{
  if(solution != NULL) {
#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
    free(solution->target);
#endif /* DOMINANCE_CHECK */
#endif /* RESTRICTED */
    free(solution->relocation);
    free(solution);
  }
}

void copy_solution(solution_t *dest, solution_t *src)
{
  if(dest != NULL && src != NULL) {
    if(dest->n_block < src->n_block) {
      dest->n_block = src->n_block;
      dest->relocation
	= (relocation_t *) realloc((void *) dest->relocation,
				   (size_t) dest->n_block
				   *sizeof(relocation_t));
#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
      dest->n_target = src->n_target;
      dest->target = (int *) realloc((void *) dest->target,
				     (size_t) dest->n_block*sizeof(int));
#endif /* DOMINANCE_CHECK */
#endif /* RESTRICTED */
    }
    dest->n_relocation = src->n_relocation;
    memcpy((void *) dest->relocation, (void *) src->relocation,
	   (size_t) src->n_relocation*sizeof(relocation_t));
#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
    dest->n_target = src->n_target;
    memcpy((void *) dest->target, (void *) src->target,
	   (size_t) src->n_target*sizeof(int));
#endif /* DOMINANCE_CHECK */
#endif /* RESTRICTED */
  }
}

void add_relocation(solution_t *solution, int src, int dst, int no)
{
  if(solution != NULL) {
    if(solution->n_block <= solution->n_relocation) {
      solution->n_block += 100;
      solution->relocation
	= (relocation_t *) realloc((void *) solution->relocation,
				   (size_t) solution->n_block
				   *sizeof(relocation_t));
#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
      solution->target
	= (int *) realloc((void *) solution->target,
			  (size_t) solution->n_block*sizeof(int));
#endif /* DOMINANCE_CHECK */
#endif /* RESTRICTED */
    }
    solution->relocation[solution->n_relocation].src = src;
    solution->relocation[solution->n_relocation].dst = dst;
    solution->relocation[solution->n_relocation++].no = no;
  }
}

#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
void add_target(solution_t *solution, int depth)
{
  if(solution != NULL) {
    if(solution->n_block <= solution->n_target) {
      solution->n_block += 100;
      solution->relocation
	= (relocation_t *) realloc((void *) solution->relocation,
				   (size_t) solution->n_block
				   *sizeof(relocation_t));
      solution->target
	= (int *) realloc((void *) solution->target,
			  (size_t) solution->n_block*sizeof(int));
    }
    solution->target[solution->n_target++] = depth;
  }
}
#endif /* DOMINANCE_CHECK */
#endif /* RESTRICTED */

state_t *create_state(problem_t *problem)
{
  int i;
  state_t *state = (state_t *) malloc(sizeof(state_t));
  
  state->ship_n_block = 0;
  state->n_blocking = 0;

  state->ystate = (yard_stack_state_t *) malloc((size_t) problem->yard_n_stack
						*sizeof(yard_stack_state_t));
  state->ship_n_tier
    = (int *) malloc((size_t) (problem->ship_n_stack + problem->yard_n_block)
		     *sizeof(int));
  state->n_blocking4 = state->ship_n_tier + problem->ship_n_stack;
#ifdef DOMINANCE_CHECK
  state->last_relocation = (int *) malloc((size_t) problem->yard_n_block
					  *sizeof(int));
#endif /* !DOMINANCE_CHECK */

  state->yard_block = (block_t **) malloc((size_t) problem->yard_n_stack
					  *sizeof(block_t *));
  state->yard_block[0] = (block_t *) malloc((size_t) problem->yard_n_stack
					    *problem->yard_s_height
					    *sizeof(block_t));

  for(i = 1; i < problem->yard_n_stack; ++i) {
    state->yard_block[i] = state->yard_block[i - 1] + problem->yard_s_height;
  }

  state->blocking4_matrix = (uchar **) malloc((size_t) problem->yard_n_block
					      *sizeof(uchar *));
  state->blocking4_matrix[0] = (uchar *) malloc((size_t) problem->yard_n_block
						*problem->yard_n_block);

  for(i = 1; i < problem->yard_n_block; ++i) {
    state->blocking4_matrix[i]
      = state->blocking4_matrix[i - 1] + problem->yard_n_block;
  }

  state->yard_position = (coordinate_t *) malloc((size_t) problem->yard_n_block
						 *sizeof(coordinate_t));

  return(state);
}

state_t *initialize_state(problem_t *problem, state_t *state)
{
  int i, j, k, k2, l;
  int yard_stack, ship_stack;
  state_t *nstate = (state == NULL)?create_state(problem):state;

  memcpy((void *) nstate->yard_position, (void *) problem->yard_position,
	 (size_t) problem->yard_n_block*sizeof(coordinate_t));
  memset((void *) nstate->ship_n_tier, 0,
	 (size_t) (problem->ship_n_stack + problem->yard_n_block)*sizeof(int));
#ifdef DOMINANCE_CHECK
  memset((void *) nstate->last_relocation, 0,
	 (size_t) problem->yard_n_block*sizeof(int));
#endif /* !DOMINANCE_CHECK */
  memset((void *) nstate->blocking4_matrix[0], 0,
	 (size_t) problem->yard_n_block*problem->yard_n_block);

  nstate->ship_n_block = 0;
  nstate->n_blocking = 0;
  for(i = 0; i < problem->yard_n_stack; ++i) {
    nstate->ystate[i].n_tier = problem->yard_n_tier[i];
    nstate->ystate[i].n_stacked = problem->yard_s_height;
    nstate->ystate[i].n_target = 0;
#ifdef DOMINANCE_CHECK
    nstate->ystate[i].last_modified = 0;
#endif /* DOMINANCE_CHECK */

    nstate->yard_block[i][0].n_blocking = 0;

    for(j = 0; j < problem->yard_n_tier[i]; ++j) {
      k = problem->yard_block[i][j];
      nstate->yard_block[i][j].no = k;
      nstate->yard_block[i][j].blocking = 0;

      ship_stack = problem->ship_position[k].s;
      if(ship_stack == -1) {
	if(j > 0) {
	  nstate->yard_block[i][j].blocking = 1;
	  ++nstate->n_blocking;
	}
      } else {
	for(l = 0; l < problem->ship_position[k].t; ++l) {
	  k2 = problem->ship_block[ship_stack][l];
	  if(problem->yard_position[k2].s == i
	     && problem->yard_position[k2].t < j) {
	    nstate->yard_block[i][j].blocking = 1;
	    ++nstate->n_blocking;
	    break;
	  }
	}
      }

      if(j > 0) {
	nstate->yard_block[i][j].n_blocking
	  = nstate->yard_block[i][j - 1].n_blocking
	  + nstate->yard_block[i][j].blocking;
      }
    }
  }

  for(i = 0; i < problem->ship_n_stack; ++i) {
    j = problem->ship_block[i][0];
    yard_stack = problem->yard_position[j].s;
    ++(nstate->ystate[yard_stack].n_target);
    nstate->ystate[yard_stack].n_stacked
      = min(nstate->ystate[yard_stack].n_stacked,
	    problem->yard_n_tier[yard_stack] - problem->yard_position[j].t - 1);
  }

  check_blocking4(problem, nstate);

  return(nstate);
}

void copy_state(problem_t *problem, state_t *dest, state_t *src)
{
  dest->ship_n_block = src->ship_n_block;
  dest->n_blocking = src->n_blocking;

  memcpy((void *) dest->ystate, (void *) src->ystate,
	 (size_t) problem->yard_n_stack*sizeof(yard_stack_state_t));
  memcpy((void *) dest->ship_n_tier, (void *) src->ship_n_tier,
	 (size_t) (problem->ship_n_stack + problem->yard_n_block)*sizeof(int));
#ifdef DOMINANCE_CHECK
  memcpy((void *) dest->last_relocation, (void *) src->last_relocation,
	 (size_t) problem->yard_n_block*sizeof(int));
#endif /* !DOMINANCE_CHECK */

  memcpy((void *) dest->yard_position, (void *) src->yard_position,
	 (size_t) problem->yard_n_block*sizeof(coordinate_t));
  memcpy((void *) dest->yard_block[0], (void *) src->yard_block[0],
	 (size_t) problem->yard_n_stack*problem->yard_s_height
	 *sizeof(block_t));
  memcpy((void *) dest->blocking4_matrix[0], (void *) src->blocking4_matrix[0],
	 (size_t) problem->yard_n_block*problem->yard_n_block);
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
    free(state->yard_position);
    free(state->blocking4_matrix[0]);
    free(state->blocking4_matrix);
    free(state->yard_block[0]);
    free(state->yard_block);
#ifdef DOMINANCE_CHECK
    free(state->last_relocation);
#endif /* DOMINANCE_CHECK */
    free(state->ship_n_tier);
    free(state->ystate);
    free(state);
  }
}

void move_all_blocks(problem_t *problem, state_t *state)
{
  int i;

  for(i = 0; i < problem->yard_n_stack; ++i) {
    if(state->ystate[i].n_stacked == 0) {
      move_all_blocks_sub(problem, state, i);
    }
  }
}

void move_all_blocks_sub(problem_t *problem, state_t *state, int s)
{
  int i;
  int no = state->yard_block[s][--state->ystate[s].n_tier].no;
  yard_stack_state_t *ystate = state->ystate;
  int ship_stack = problem->ship_position[no].s;
  int no2, s2, ship_stack2;

  ++state->ship_n_block;
  --ystate[s].n_target;
  ++state->ship_n_tier[ship_stack];

  for(i = ystate[s].n_tier - 1; i >= 0; --i) {
    no2 = state->yard_block[s][i].no;
    ship_stack2 = problem->ship_position[no2].s;

    if(ship_stack2 >= 0
       && state->ship_n_tier[ship_stack2] == problem->ship_position[no2].t) {
      break;
    }
  }

  if(i >= 0) {
    ystate[s].n_stacked = ystate[s].n_tier - i - 1;
  } else {
    ystate[s].n_stacked = problem->yard_s_height;
  }

  s2 = s;
  if(state->ship_n_tier[ship_stack] < problem->ship_n_tier[ship_stack]) {
    no2 = problem->ship_block[ship_stack][state->ship_n_tier[ship_stack]];
    s2 = state->yard_position[no2].s;
    ++ystate[state->yard_position[no2].s].n_target;
    ystate[s2].n_stacked
      = min(ystate[s2].n_stacked,
	    ystate[s2].n_tier - state->yard_position[no2].t - 1);
  }

  if(ystate[s].n_stacked == 0) {
    move_all_blocks_sub(problem, state, s);
  }

  if(s2 != s && ystate[s2].n_stacked == 0) {
    move_all_blocks_sub(problem, state, s2);
  }
}

uchar move_all_blocks2(problem_t *problem, state_t *state,
		       solution_t *solution)
{
  int i, j, no;
#ifdef DOMINANCE_CHECK
  int lv;
#endif /* DOMINANCE_CHECK */
  int ship_stack;
  yard_stack_state_t *ystate = state->ystate;
  int *ship_n_tier = state->ship_n_tier;
  int **ship_block = problem->ship_block;
  block_t **yard_block = state->yard_block;
  coordinate_t *yard_position = state->yard_position;

  for(i = problem->yard_n_stack - 1;
      i >= 0 && state->ship_n_block < problem->ship_n_block; --i) {
    if(ystate[i].n_stacked > 0 || ystate[i].n_tier == 0) {
      continue;
    }

#ifdef DOMINANCE_CHECK
#ifdef RESTRICTED
      ystate[i].last_modified = solution->n_relocation;
#else /* !RESTRICTED */
      ystate[i].last_modified = solution->n_relocation - MAX_N_RELOCATION;
#endif /* !RESTRICTED */
#endif /* DOMINANCE_CHECK */

    for(; ystate[i].n_tier > 0; --ystate[i].n_tier) {
      no = yard_block[i][ystate[i].n_tier - 1].no;
      ship_stack = problem->ship_position[no].s;

      if(ship_stack < 0
	 || ship_n_tier[ship_stack] < problem->ship_position[no].t) {
	break;
      }

#ifdef DOMINANCE_CHECK
      lv = state->last_relocation[no];

      if(lv > 0) {
#ifdef RESTRICTED
	for(j = 0; j < i; ++j) {
	  if(ystate[j].n_tier <= ystate[i].n_tier - 1
	     && ystate[j].last_modified%MAX_N_RELOCATION < lv) {
	      return(TRUE);
	  }
	}

	for(j = i + 1; j < problem->yard_n_stack; ++j) {
	  if(ystate[j].n_tier < ystate[i].n_tier - 1
	     && ystate[j].last_modified%MAX_N_RELOCATION < lv) {
	    return(TRUE);
	  }
	}
#else /* !RESTRICTED */
	if(ystate[solution->relocation[lv - 1].src].last_modified
	   == MAX_N_RELOCATION + lv) {
	  return(TRUE);
	}

	for(j = 0; j < i; ++j) {
	  if(ystate[j].n_tier <= ystate[i].n_tier - 1
	     && (ystate[j].last_modified + MAX_N_RELOCATION)%MAX_N_RELOCATION
	     < lv) {
	    return(TRUE);
	  }
	}

	for(j = i + 1; j < problem->yard_n_stack; ++j) {
	  if(ystate[j].n_tier < ystate[i].n_tier - 1
	     && (ystate[j].last_modified + MAX_N_RELOCATION)%MAX_N_RELOCATION
	     < lv) {
	    return(TRUE);
	  }
	}
#endif /* !RESTRICTED */
      }
#endif /* DOMINANCE_CHECK */

      ++ship_n_tier[ship_stack];
      ++state->ship_n_block;
      --ystate[i].n_target;

      if(ship_n_tier[ship_stack] < problem->ship_n_tier[ship_stack]) {
	no = ship_block[ship_stack][ship_n_tier[ship_stack]];
	++ystate[yard_position[no].s].n_target;
	if(yard_position[no].s != i) {
	  if(ystate[yard_position[no].s].n_tier - yard_position[no].t - 1
	     < ystate[yard_position[no].s].n_stacked) {
	    ystate[yard_position[no].s].n_stacked
	      = ystate[yard_position[no].s].n_tier - yard_position[no].t - 1;
#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
	    ystate[yard_position[no].s].last_modified
	      = solution->target[solution->n_target - 1]*MAX_N_RELOCATION
	      + ystate[yard_position[no].s].last_modified%MAX_N_RELOCATION;
#endif /* DOMINANCE_CHECK */
#endif /* RESTRICTED */
	  }
	}
      }
    }

    for(j = ystate[i].n_tier - 1; j >= 0; --j) {
      no = yard_block[i][j].no;
      ship_stack = problem->ship_position[no].s;
      if(ship_stack >= 0
	 && ship_n_tier[ship_stack] == problem->ship_position[no].t) {
	break;
      }
    }

    if(j >= 0) {
      ystate[i].n_stacked = ystate[i].n_tier - j - 1;
#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
      ystate[i].last_modified
	+= solution->target[solution->n_target - 1]*MAX_N_RELOCATION;
#endif /* DOMINANCE_CHECK */
#endif /* RESTRICTED */
    } else {
      ystate[i].n_stacked = problem->yard_s_height;
    }

    if(i < problem->yard_n_stack - 1) {
      i = problem->yard_n_stack;
    }

#if 0
    print_state(problem, state, stdout);
    for(j = 0; j < problem->yard_n_stack; ++j) {
      printf("[%d,%d]", ystate[j].n_target, ystate[j].n_stacked);
    }
    printf("\n");
#endif
  }

  return(FALSE);
}

#if 0
uchar move_all_blocks2(problem_t *problem, state_t *state,
		       int *lb_ship_block, int lv)
{
  int i, j, no;
  int ship_stack;
  yard_stack_state_t *ystate = state->ystate;
  int *ship_n_tier = state->ship_n_tier;
  int **ship_block = problem->ship_block;
  block_t **yard_block = state->yard_block;
  coordinate_t *yard_position = state->yard_position;

  for(i = problem->yard_n_stack - 1; i >= 0; --i) {
    int n_removed = 0;

    if(ystate[i].n_stacked > 0 || ystate[i].n_tier == 0) {
      continue;
    }

    if(lv > 0) {
      ystate[i].last_modified = lv - MAX_N_RELOCATION;
    }

    for(; ystate[i].n_tier > 0; --ystate[i].n_tier) {
      no = yard_block[i][ystate[i].n_tier - 1].no;
      ship_stack = problem->ship_position[no].s;

      if(ship_stack < 0
	 || ship_n_tier[ship_stack] < problem->ship_position[no].t) {
	break;
      }

      if(lb_ship_block != NULL) {
	ship_n_tier[ship_stack] = lb_ship_block[no];
      } else {
	++ship_n_tier[ship_stack];
      }
      ++n_removed;
      //      --ystate[i].n_target;

      if(ship_n_tier[ship_stack] < problem->ship_n_tier[ship_stack]) {
	no = ship_block[ship_stack][ship_n_tier[ship_stack]];
	//	++ystate[yard_position[no].s].n_target;
	if(yard_position[no].s != i) {
	  ystate[yard_position[no].s].n_stacked
	    = min(ystate[yard_position[no].s].n_stacked,
		  ystate[yard_position[no].s].n_tier - yard_position[no].t - 1);
	}
      }
    }

    state->ship_n_block += n_removed;
    if(state->ship_n_block == problem->ship_n_block) {
      break;
    }

    for(j = ystate[i].n_tier - 1; j >= 0; --j) {
      no = yard_block[i][j].no;
      ship_stack = problem->ship_position[no].s;
      if(ship_stack >= 0
	 && ship_n_tier[ship_stack] == problem->ship_position[no].t) {
	break;
      }
    }
    if(j >= 0) {
      ystate[i].n_stacked = ystate[i].n_tier - j - 1;
    } else {
      ystate[i].n_stacked = problem->yard_s_height;
    }

    if(i < problem->yard_n_stack - 1) {
      i = problem->yard_n_stack;
    }

#if 0
    print_state(problem, state, stdout);
    for(j = 0; j < problem->yard_n_stack; ++j) {
      printf("[%d,%d]", ystate[j].n_target, ystate[j].n_stacked);
    }
    printf("\n");
#endif
  }

  return(FALSE);
}
#endif

void move_all_blocks3(problem_t *problem, state_t *state,
		      int *lb_ship_block, int s)
{
  int i;
  int no = state->yard_block[s][state->ystate[s].n_tier - 1].no;
  int ship_stack = problem->ship_position[no].s;
  int no2, s2, ship_stack2;

#if 0
  printf("name=%s, stack=%d, ship_stack=%d\n",
	 problem->name[no], s + 1, ship_stack + 1);
  print_state(problem, state, stdout);
  for(i = 0; i < problem->yard_n_stack; ++i) {
    printf("[%d,%d]", state->ystate[i].n_target, state->ystate[i].n_stacked);
  }
  printf("\n");
  printf("# ship blocks=%d\n", state->ship_n_block);
#endif

  if(++state->ship_n_block == problem->ship_n_block) {
    return;
  }

  --state->ystate[s].n_tier;
  state->ship_n_tier[ship_stack] = lb_ship_block[no];

  for(i = state->ystate[s].n_tier - 1; i >= 0; --i) {
    no2 = state->yard_block[s][i].no;
    ship_stack2 = problem->ship_position[no2].s;

    if(ship_stack2 >= 0
       && state->ship_n_tier[ship_stack2] == problem->ship_position[no2].t) {
      break;
    }
  }

  if(i >= 0) {
    state->ystate[s].n_stacked = state->ystate[s].n_tier - i - 1;
  } else {
    state->ystate[s].n_stacked = problem->yard_s_height;
  }

  s2 = s;
  if(lb_ship_block[no] < problem->ship_n_tier[ship_stack]) {
    no2 = problem->ship_block[ship_stack][lb_ship_block[no]];
    s2 = state->yard_position[no2].s;

    state->ystate[s2].n_stacked
      = min(state->ystate[s2].n_stacked,
	    state->ystate[s2].n_tier - state->yard_position[no2].t - 1);
  }

  if(state->ystate[s].n_stacked == 0) {
    move_all_blocks3(problem, state, lb_ship_block, s);
  }
  if(s2 != s && state->ystate[s2].n_stacked == 0) {
    move_all_blocks3(problem, state, lb_ship_block, s2);
  }
}

void check_blocking4(problem_t *problem, state_t *state)
{
  int i, j, k, l;
  int no1, no2;
  int yard_stack, ship_stack, ship_stack2;
  int **ship_block = problem->ship_block;
  block_t **yard_block = state->yard_block;
  uchar **blocking4_matrix = state->blocking4_matrix;
  int *n_blocking4 = state->n_blocking4;
  coordinate_t *yard_position = state->yard_position;
  coordinate_t *ship_position = problem->ship_position;
  int *vposition_list
    = (int *) malloc((size_t) problem->yard_n_stack*sizeof(int));

  for(i = 0; i < problem->yard_n_stack; ++i) {
    for(j = state->ystate[i].n_tier - 1; j > 0; --j) {
      if(yard_block[i][j].blocking) {
	continue;
      }

      no1 = yard_block[i][j].no;
      ship_stack = ship_position[no1].s;

      for(k = 0; k < problem->yard_n_stack;
	  vposition_list[k++] = problem->yard_s_height);
      for(k = state->ship_n_tier[ship_stack]; k < ship_position[no1].t; ++k) {
	no2 = ship_block[ship_stack][k];
	vposition_list[yard_position[no2].s]
	  = min(vposition_list[yard_position[no2].s], yard_position[no2].t);
      }

      for(k = j - 1; k >= 0; --k) {
	ship_stack2 = ship_position[yard_block[i][k].no].s;
	if(ship_stack2 == ship_stack) {
	  continue;
	}

	for(l = ship_position[yard_block[i][k].no].t + 1;
	    l < problem->ship_n_tier[ship_stack2]; ++l) {
	  no2 = ship_block[ship_stack2][l];
	  yard_stack = yard_position[no2].s;
	  if(!blocking4_matrix[no1][no2]
	     && yard_stack != i
	     && !yard_block[yard_stack][yard_position[no2].t].blocking
	     && yard_position[no2].t > vposition_list[yard_stack]) {
	    blocking4_matrix[no1][no2] = blocking4_matrix[no2][no1] = 1;
	    ++n_blocking4[no1];
	    ++n_blocking4[no2];
	  }
	}
      }
    }
  }

  free(vposition_list);
}
