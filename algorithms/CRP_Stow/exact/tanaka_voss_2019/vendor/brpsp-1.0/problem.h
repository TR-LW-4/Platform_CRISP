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
 *  $Id: problem.h,v 1.2 2019/06/11 12:13:41 tanaka Exp tanaka $
 *  $Revision: 1.2 $
 *  $Date: 2019/06/11 12:13:41 $
 *  $Author: tanaka $
 *
 */
#ifndef PROBLEM_H
#define PROBLEM_H
#include "define.h"

typedef struct {
  int s;
  int t;
} coordinate_t;

typedef struct {
  int type;

  int yard_n_block;
  int yard_n_stack;
  int yard_s_height;
  int ship_n_block;
  int ship_n_stack;
  int max_ship_tier;

  coordinate_t *yard_position;
  coordinate_t *ship_position;
  int *no;
  int name_length;
  char **name;

  int *yard_n_tier;
  int **yard_block;

  int *ship_n_tier;
  int **ship_block;

  double stime;
  double time;
} problem_t;

extern uchar verbose;
extern uchar lower_bound_only;
extern int tlimit;

problem_t *create_problem(int, int, int, int, int, int, int);
void free_problem(problem_t *);

#endif /* !PROBLEM_H */
