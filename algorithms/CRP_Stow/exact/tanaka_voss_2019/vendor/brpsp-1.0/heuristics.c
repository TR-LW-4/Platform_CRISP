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
 *  $Id: heuristics.c,v 1.3 2019/06/11 12:13:41 tanaka Exp tanaka $
 *  $Revision: 1.3 $
 *  $Date: 2019/06/11 12:13:41 $
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

#ifdef RESTRICTED
solution_t *heuristics(problem_t *problem, state_t *state,
		       solution_t *solution, int target_stack, int ub)
#else /* RESTRICTED */
solution_t *heuristics(problem_t *problem, state_t *state,
		       solution_t *solution, int ub)
#endif /* !RESTRICTED */
{
  int i, j, k;
  int no, no2;
  int ship_stack, ship_stack2;
  int yard_stack, yard_stack2;
  uchar cblocking, cblocking4;
  state_t *nstate = duplicate_state(problem, state);
  solution_t *csolution = (solution == NULL)?create_solution():solution;
  yard_stack_state_t *ystate = nstate->ystate;
  int *ship_n_tier = nstate->ship_n_tier;
  int **ship_block = problem->ship_block;
  block_t **yard_block = nstate->yard_block;
  coordinate_t *yard_position = nstate->yard_position;
  coordinate_t *ship_position = problem->ship_position;
  int *n_blocking4 = nstate->n_blocking4;
  uchar **blocking4_matrix = nstate->blocking4_matrix;
  int *vposition_list = (int *) malloc((size_t) 2*problem->yard_n_stack
				   *sizeof(int));
  int *stack_priority = vposition_list + problem->yard_n_stack;

#if 0
  printf("heuristics\n");
  print_state(problem, state, stdout);
#endif

  while(nstate->ship_n_block < problem->ship_n_block) {
    int src_stack = -1;
    int min_score = 2*problem->max_ship_tier*problem->max_ship_tier
      *problem->max_ship_tier*problem->yard_s_height
      *problem->yard_s_height*problem->yard_s_height;
    int yard_space = problem->yard_s_height*(problem->yard_n_stack - 1)
      - problem->yard_n_block + nstate->ship_n_block;

#if 0
    printf("----\n");
    print_state(problem, nstate, stdout);
    for(i = 0; i < problem->yard_n_stack; ++i) {
      printf("[%d,%d]", ystate[i].n_target, ystate[i].n_stacked);
    }
    printf("\n");
    for(i = 0; i < problem->yard_n_stack; ++i) {
      for(j = 0; j < ystate[i].n_tier; ++j) {
	if(yard_block[i][j].blocking || n_blocking4[yard_block[i][j].no]) {
	  printf("%s(%u,%u) ",
		 problem->name[yard_block[i][j].no],
		 yard_block[i][j].blocking, n_blocking4[yard_block[i][j].no]);
	}
      }
    }
    printf("\n");
    for(i = 0; i < problem->yard_n_block; ++i) {
      for(j = i + 1; j < problem->yard_n_block; ++j) {
	if(blocking4_matrix[i][j]) {
	  printf("<%s,%s> ", problem->name[i], problem->name[j]);
	}
      }
    }
    printf("\n");
    printf("----\n");
#endif

#ifdef RESTRICTED
    if(target_stack >= 0) {
      src_stack = target_stack;
      target_stack = -1;
      goto src_stack_skip;
    }
#endif /* RESTRICTED */

    for(i = 0; i < problem->yard_n_stack; ++i) {
      if(ystate[i].n_stacked < problem->yard_s_height
	 && yard_space + ystate[i].n_tier >= ystate[i].n_stacked) {
	int no = yard_block[i][ystate[i].n_tier - ystate[i].n_stacked - 1].no;
	int n_stacked_blocking = 0;
	int n_stacked_blocking4 = 0;
	int score, spr, spr2;

	spr = problem->max_ship_tier;
	for(j = ystate[i].n_tier - 1;
	    j >= ystate[i].n_tier - ystate[i].n_stacked; --j) {
	  int pr = ship_position[yard_block[i][j].no].t
	    - ship_n_tier[ship_position[yard_block[i][j].no].s];
	  spr = min(spr, pr);

	  if(yard_block[i][j].blocking) {
	    ++n_stacked_blocking;
	  } else if(n_blocking4[yard_block[i][j].no] > 0) {
	    ++n_stacked_blocking4;
	  }
	}

	spr2 = problem->max_ship_tier;
	for(--j; j >= 0; --j) {
	  int pr = ship_position[yard_block[i][j].no].t
	    - ship_n_tier[ship_position[yard_block[i][j].no].s];
	  spr2 = min(spr, pr);
	}

#if 0
	printf("%d: stacked=%d, blocking2=%d, blocking4=%d\n",
	       i + 1, ystate[i].n_stacked, n_stacked_blocking,
	       n_stacked_blocking4);
#endif

	score = problem->max_ship_tier
	  *(problem->max_ship_tier
	    *(problem->max_ship_tier
	      *(problem->yard_s_height
		*(problem->yard_s_height
		  *(ystate[i].n_stacked - n_stacked_blocking
		    - n_stacked_blocking4)
		  + n_stacked_blocking4)
		+ n_stacked_blocking)
	      + (problem->max_ship_tier - spr))
	    + (problem->max_ship_tier
	       - problem->ship_n_tier[ship_position[no].s]
	       + ship_position[no].t))
	  + spr2;

	if(score < min_score) {
	  src_stack = i;
	  min_score = score;
	}
      }
    }

#ifdef RESTRICTED
  src_stack_skip:
#endif /* RESTRICTED */

#if 0
    printf("src_stack=%d\n", src_stack + 1);
#endif

    while(ystate[src_stack].n_stacked-- > 0) {
      int dst_stack = -1;
      int min_score
	= 4*(problem->max_ship_tier + 1)*(problem->max_ship_tier + 1);
      int score, spr, spr2;

      no = yard_block[src_stack][--ystate[src_stack].n_tier].no;
      ship_stack = ship_position[no].s;

      if(n_blocking4[no] > 0) {
	for(i = 0; i < problem->yard_n_block; ++i) {
	  n_blocking4[i] -= blocking4_matrix[no][i];
	  blocking4_matrix[no][i] = blocking4_matrix[i][no] = 0;
	}
	n_blocking4[no] = 0;
      }

      for(i = 0; i < problem->yard_n_stack;
	  vposition_list[i] = problem->yard_s_height,
	  stack_priority[i] = problem->yard_n_block, ++i);

      for(i = ship_n_tier[ship_stack]; i < ship_position[no].t; ++i) {
	no2 = ship_block[ship_stack][i];
	vposition_list[yard_position[no2].s]
	  = min(vposition_list[yard_position[no2].s], yard_position[no2].t);
      }

      for(i = ship_n_tier[ship_stack]; i < problem->ship_n_tier[ship_stack];
	  ++i) {
	no2 = ship_block[ship_stack][i];
	stack_priority[yard_position[no2].s]
	  = min(stack_priority[yard_position[no2].s], i);
      }

      for(i = 0; i < problem->yard_n_stack; ++i) {
	if(i == src_stack || ystate[i].n_tier == problem->yard_s_height) {
	  continue;
	}

	cblocking = (stack_priority[i] < ship_position[no].t);
	cblocking4 = 0;

	if(!cblocking) {
	  for(j = ystate[i].n_tier - 1; j >= 0 && cblocking4 == 0; --j) {
	    ship_stack2 = ship_position[yard_block[i][j].no].s;
	    if(ship_stack2 == ship_stack) {
	      continue;
	    }

	    for(k = ship_position[yard_block[i][j].no].t + 1;
		k < problem->ship_n_tier[ship_stack2]; ++k) {
	      no2 = ship_block[ship_stack2][k];
	      yard_stack2 = yard_position[no2].s;
	      if(yard_stack2 != i
		 && !yard_block[yard_stack2][yard_position[no2].t].blocking
		 && yard_position[no2].t > vposition_list[yard_stack2]) {
		cblocking4 = 1;
		break;
	      }
	    }
	  }
	}

	k = -2;
	spr = spr2 = problem->max_ship_tier;
	for(j = 0; j < ystate[i].n_tier; ++j) {
	  int pr = ship_position[yard_block[i][j].no].t
	    - ship_n_tier[ship_position[yard_block[i][j].no].s];
	  spr = min(spr, pr);
	  if(ship_position[yard_block[i][j].no].s == ship_stack) {
	    spr2 = min(spr2, pr);
	    k = j;
	  }
	}

#if 0
	if(cblocking) {
	  printf("%d: blocking, %d, %d\n", i + 1, stack_priority[i], spr);
	} else if(cblocking4 == 1) {
	  printf("%d: blocking4, %d, %d\n", i + 1, stack_priority[i], spr);
	} else {
	  printf("%d: %d, %d\n", i + 1, stack_priority[i], spr);
	}
#endif

#if 1
	if(cblocking == 0 && cblocking4 == 0 && k == ystate[i].n_tier - 1) {
	  if(spr2 < min_score) {
	    min_score = spr2;
	    dst_stack = i;
	  }
	  continue;
	}
#endif
	score = (problem->max_ship_tier + 1)
	  *((problem->max_ship_tier + 1)*(2*cblocking + cblocking4)
	    + (problem->max_ship_tier - spr))
	  + problem->max_ship_tier;
	if(score < min_score) {
	  dst_stack = i;
	  min_score = score;
	}
      }

#if 0
      printf("dst_stack=%d\n", dst_stack + 1);
#endif

      cblocking4 = 0;
      if(min_score
	 >= 2*(problem->max_ship_tier + 1)*(problem->max_ship_tier + 1)) {
	yard_block[dst_stack][ystate[dst_stack].n_tier].blocking = 1;
      } else {
	yard_block[dst_stack][ystate[dst_stack].n_tier].blocking = 0;
	if(min_score
	   >= (problem->max_ship_tier + 1)*(problem->max_ship_tier + 1)) {
	  cblocking4 = 1;
	}
      }

      if(dst_stack == -1) {
	fprintf(stderr, "no stack found\n");
	exit(1);
      }

#if 0
      printf("dst_stack=%d\n", dst_stack + 1);
#endif

      yard_position[no].s = dst_stack;
      yard_position[no].t = ystate[dst_stack].n_tier;
      if(ystate[dst_stack].n_tier > 0) {
	yard_block[dst_stack][ystate[dst_stack].n_tier].n_blocking
	  = yard_block[dst_stack][ystate[dst_stack].n_tier - 1].n_blocking
	  + yard_block[dst_stack][ystate[dst_stack].n_tier].blocking;
      }

      if(cblocking4) {
	for(i = ystate[dst_stack].n_tier - 1; i >= 0; --i) {
	  ship_stack2 = ship_position[yard_block[dst_stack][i].no].s;
	  if(ship_stack2 == ship_stack) {
	    continue;
	  }

	  for(j = ship_position[yard_block[dst_stack][i].no].t + 1;
	      j < problem->ship_n_tier[ship_stack2]; ++j) {
	    no2 = ship_block[ship_stack2][j];
	    yard_stack = yard_position[no2].s;
	    if(yard_stack != dst_stack
	       && vposition_list[yard_stack] < yard_position[no2].t
	       && !yard_block[yard_stack][yard_position[no2].t].blocking
	       && !blocking4_matrix[no][no2]) {
	      blocking4_matrix[no][no2] = blocking4_matrix[no2][no] = 1;
	      ++n_blocking4[no];
	      ++n_blocking4[no2];
	    }
	  }
	}
      }

      yard_block[dst_stack][ystate[dst_stack].n_tier++].no = no;
      if(ystate[dst_stack].n_stacked < problem->yard_s_height) {
	++ystate[dst_stack].n_stacked;
      }
      add_relocation(csolution, src_stack, dst_stack, no);
      if(csolution->n_relocation > ub) {
	break;
      }

#if 0
      print_state(problem, nstate, stdout);

      for(i = 0; i < problem->yard_n_stack; ++i) {
	for(j = 0; j < ystate[i].n_tier; ++j) {
	  if(yard_block[i][j].blocking || n_blocking4[yard_block[i][j].no]) {
	    printf("%s(%u,%u) ",
		   problem->name[yard_block[i][j].no],
		   yard_block[i][j].blocking, n_blocking4[yard_block[i][j].no]);
	  }
	}
      }
      printf("\n");
      for(i = 0; i < problem->yard_n_block; ++i) {
	for(j = i + 1; j < problem->yard_n_block; ++j) {
	  if(blocking4_matrix[i][j]) {
	    printf("<%s,%s> ", problem->name[i], problem->name[j]);
	  }
	}
      }
      printf("\n");
#endif
    }

    ystate[src_stack].n_stacked = 0;
    move_all_blocks(problem, nstate);

    if(csolution->n_relocation >= ub
       && nstate->ship_n_block < problem->ship_n_block) {
      break;
    }

#if 0
    print_state(problem, nstate, stdout);
#endif

  }

  free_state(nstate);
  free(vposition_list);

#if 0
  if(ub < 0 || solution->n_relocation < ub) {
    print_solution(problem, solution, stdout);
  }
  exit(1);
#endif

  return(csolution);
}
