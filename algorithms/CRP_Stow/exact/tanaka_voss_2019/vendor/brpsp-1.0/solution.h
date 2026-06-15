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
 *  $Id: solution.h,v 1.3 2019/06/11 12:13:41 tanaka Exp tanaka $
 *  $Revision: 1.3 $
 *  $Date: 2019/06/11 12:13:41 $
 *  $Author: tanaka $
 *
 */
#ifndef SOLUTION_H
#define SOLUTION_H
#include "define.h"
#include "problem.h"

typedef struct {
  int no;
  int src;
  int dst;
} relocation_t;

typedef struct {
  int n_block;
  int n_relocation;
  relocation_t *relocation;
#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
  int n_target;
  int *target;
#endif /* DOMINANCE_CHECK */
#endif /* RESTRICTED */
} solution_t;

typedef struct {
  int n_tier;
  int n_stacked;
  int n_target;
#ifdef DOMINANCE_CHECK
  int last_modified;
#endif /* DOMINANCE_CHECK */
} yard_stack_state_t;

typedef struct {
  int no;
  int n_blocking;
  uchar blocking;
} block_t;

typedef struct {
  int ship_n_block;
  int n_blocking;
  coordinate_t *yard_position;
  block_t **yard_block;

#ifdef DOMINANCE_CHECK
  int *last_relocation;
#endif /* DOMINANCE_CHECK */
  int *ship_n_tier;
  yard_stack_state_t *ystate;

  uchar **blocking4_matrix;
  int *n_blocking4;
} state_t;

solution_t *create_solution(void);
void free_solution(solution_t *);
void copy_solution(solution_t *, solution_t *);
void add_relocation(solution_t *, int, int, int);
#ifdef RESTRICTED
#ifdef DOMINANCE_CHECK
void add_target(solution_t *, int);
#endif /* DOMINANCE_CHECK */
#endif /* RESTRICTED */
state_t *create_state(problem_t *);
state_t *initialize_state(problem_t *, state_t *);
void copy_state(problem_t *, state_t *, state_t *);
state_t *duplicate_state(problem_t *, state_t *);
void free_state(state_t *state);
void move_all_blocks(problem_t *, state_t *);
uchar move_all_blocks2(problem_t *, state_t *, solution_t *);
void move_all_blocks3(problem_t *, state_t *, int *, int);
void check_4blocking(problem_t *, state_t *);

#endif /* !SOLUTION_H */
