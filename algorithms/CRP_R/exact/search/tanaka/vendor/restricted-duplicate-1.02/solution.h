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
 *  $Id: solution.h,v 1.3 2015/06/11 03:26:15 tanaka Exp tanaka $
 *  $Revision: 1.3 $
 *  $Date: 2015/06/11 03:26:15 $
 *  $Author: tanaka $
 *
 */
#ifndef SOLUTION_H
#define SOLUTION_H
#include "define.h"
#include "problem.h"

typedef struct {
  int src;
  int dst;
  int no;
} relocation_t;

typedef struct {
  int n_relocation;
  int n_block;
  relocation_t *relocation;
} solution_t;

typedef struct {
  int n_stacked;
  int min_priority;
  int min_priority2;
} block_info_t;

typedef struct {
  int stack;
  int n_space;
  int n_stacked;
  int min_priority;
  int min_priority2;
} stack_info_t;

typedef struct {
  int n_block;
  int lb1;
  int *n_tier;
  block_t **block;
  block_info_t **bi;
  stack_info_t *info;
} state_t;

solution_t *create_solution(void);
void free_solution(solution_t *);
void copy_solution(solution_t *, solution_t *);
void add_relocation(solution_t *, int, int, int);
state_t *create_state(problem_t *);
state_t *initialize_state(problem_t *, state_t *);
void copy_state(problem_t *, state_t *, state_t *);
state_t *duplicate_state(problem_t *, state_t *);
void free_state(state_t *state);
int retrieve_all_blocks(problem_t *, state_t *);
int stack_info_comp(const void *, const void *);

#endif /* !SOLUTION_H */
