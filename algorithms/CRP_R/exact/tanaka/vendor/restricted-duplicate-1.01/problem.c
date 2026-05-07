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
 *  $Id: problem.c,v 1.6 2015/06/11 03:23:35 tanaka Exp tanaka $
 *  $Revision: 1.6 $
 *  $Date: 2015/06/11 03:23:35 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "define.h"
#include "problem.h"

uchar verbose;
int tlimit;

problem_t *create_problem(int n_stack, int s_height, int n_block)
{
  int i;
  problem_t *problem = (problem_t *) calloc(1, sizeof(problem_t));

  problem->n_block = n_block;
  problem->n_stack = n_stack;
  problem->s_height = s_height;
  problem->position
    = (coordinate_t *) calloc((size_t) n_block, sizeof(coordinate_t));
  problem->block = (block_t **) malloc((size_t) n_stack*sizeof(block_t *));
  problem->block[0] = (block_t *) calloc((size_t) n_stack*s_height,
					 sizeof(block_t));
  for(i = 1; i < n_stack; i++) {
    problem->block[i] = problem->block[i - 1] + s_height;
  }
  problem->n_tier = (int *) calloc((size_t) n_block, sizeof(int));

  return(problem);
}

void free_problem(problem_t *problem)
{
  if(problem != NULL) {
    free(problem->block[0]);
    free(problem->block);
    free(problem->n_tier);
    free(problem->position);
    free(problem);
  }
}
