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
 *  $Id: problem.c,v 1.2 2019/06/11 12:13:41 tanaka Exp tanaka $
 *  $Revision: 1.2 $
 *  $Date: 2019/06/11 12:13:41 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "define.h"
#include "problem.h"

problem_t *create_problem(int yard_n_stack, int yard_s_height, int yard_n_block,
			  int ship_n_stack, int ship_n_block, int type,
			  int max_ship_tier)
{
  int i;
  problem_t *problem = (problem_t *) calloc(1, sizeof(problem_t));

  problem->type = type;
  problem->yard_n_block = yard_n_block;
  problem->yard_n_stack = yard_n_stack;
  problem->yard_s_height = yard_s_height;
  problem->ship_n_block = ship_n_block;
  problem->ship_n_stack = ship_n_stack;
  problem->max_ship_tier = max_ship_tier;

  problem->no = (int *) calloc((size_t) yard_n_block, sizeof(int));

  problem->name = (char **) malloc((size_t) yard_n_block*sizeof(char *));

  if(type == 0) {
    for(i = 1, problem->name_length = 0; i <= yard_n_block;
	i *= 10, ++problem->name_length);
  } else {
    for(i = 1, problem->name_length = 1; i <= max_ship_tier;
	i *= 10, ++problem->name_length);
  }

  problem->name[0]
    = (char *) calloc((size_t) yard_n_block*(problem->name_length + 1), 1);
  for(i = 1; i < yard_n_block; ++i) {
    problem->name[i] = problem->name[i - 1] + (problem->name_length + 1);
  }

  problem->yard_position
    = (coordinate_t *) calloc((size_t) 2*yard_n_block, sizeof(coordinate_t));
  problem->ship_position = problem->yard_position + yard_n_block;

  problem->yard_block
    = (int **) malloc((size_t) (yard_n_stack + ship_n_stack)*sizeof(int *));
  problem->ship_block = problem->yard_block + yard_n_stack;
  problem->yard_block[0] = NULL;

  problem->yard_n_tier
    = (int *) calloc((size_t) (yard_n_stack + ship_n_stack), sizeof(int));
  problem->ship_n_tier = problem->yard_n_tier + yard_n_stack;

  return(problem);
}

void free_problem(problem_t *problem)
{
  if(problem != NULL) {
    free(problem->yard_n_tier);
    free(problem->yard_block[0]);
    free(problem->yard_block);
    free(problem->yard_position);
    free(problem->name);
    free(problem->no);
    free(problem);
  }
}
