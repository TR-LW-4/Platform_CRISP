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
 *  $Id: main.c,v 1.13 2015/10/15 01:51:25 tanaka Exp tanaka $
 *  $Revision: 1.13 $
 *  $Date: 2015/10/15 01:51:25 $
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
#include "solve.h"

static problem_t *read_file(char *, int, int);
static void usage(char *);
static void remove_comments(char *);
static int blockdata_comp(const void *, const void *);

int main(int argc, char **argv)
{
  char **agv;
  int n_stack, s_height;
  uchar ret;
  problem_t *problem;
  solution_t *solution;
  
  verbose = FALSE;
  tlimit = -1;
  n_stack = s_height = 0;
  for(agv = argv + 1, argc--; argc > 0 && agv[0][0] == '-'; argc--, agv++) {
    switch(agv[0][1]) {
    default:
    case 'h':
      usage(argv[0]);
      return(1);
      break;
    case 'v':
      verbose = TRUE;
      break;
    case 's':
      verbose = FALSE;
      break;
    case 'S':
      if(argc == 1) {
	usage(argv[0]);
	return(1);
      }
      n_stack = (uint) atoi(agv[1]);
      agv++;
      argc--;
      break;
    case 'T':
      if(argc == 1) {
	usage(argv[0]);
	return(1);
      }
      s_height = (uint) atoi(agv[1]);
      agv++;
      argc--;
      break;
    case 't':
      if(argc == 1) {
	usage(argv[0]);
	return(1);
      }
      tlimit = (int) atoi(agv[1]);
      agv++;
      argc--;
      break;
    }
  }

  problem = read_file((argc >= 1)?agv[0]:NULL, n_stack, s_height);

  if(problem == NULL) {
    return(0);
  }

  if(verbose == TRUE) {
    print_problem(problem, stdout);
  }

  solution = create_solution();

  timer_start(problem);

  ret = solve(problem, solution);

  print_time(problem);
  if(ret == TRUE) {
    fprintf(stderr, "opt=%d\n", solution->n_relocation);
  } else {
    fprintf(stderr, "best=%d\n", solution->n_relocation);
  }

  print_solution(problem, solution, stdout);

  free_solution(solution);

  return(0);
}

void usage(char *name)
{
  fprintf(stdout, "Usage: %s [-v|-s] [-S S] [-T T] [-t L] [input file]\n",
	  name);
  fprintf(stdout, "b&b algorithm for the block relocation problem.\n");
  fprintf(stdout, " -v|-s: verbose|silent\n");
  fprintf(stdout, " -S  S: number of stacks.\n");
  fprintf(stdout, " -T  T: maximum number of tiers.\n");
  fprintf(stdout, " -t  L: time limit.\n");
  fprintf(stdout, "\n");
}

typedef struct {
  int s;
  int t;
  int priority;
} blockdata_t;

problem_t *read_file(char *filename, int n_stack, int s_height)
{
  int n_block;
  FILE *fp;
  problem_t *problem;
  char buf[MAXBUFLEN];

  if(filename == NULL) {
    fp = stdin;
  } else {
#ifdef _MSC_VER
    if(fopen_s(&fp, filename, "r") != 0) {
#else /* !_MSC_VER */
    if((fp = fopen(filename, "r")) == NULL) {
#endif /* !_MSC_VER */
      fprintf(stderr, "Failed to open file: %s\n", filename);
      return(NULL);
    }
  }

  problem = NULL;
  while(fgets(buf, MAXBUFLEN, fp) != NULL) {
    int dn_stack;
    remove_comments(buf);
#ifdef _MSC_VER
    if(sscanf_s(buf, "%d %d", &dn_stack, &n_block) == 2) {
#else /* !_MSC_VER */
    if(sscanf(buf, "%d %d", &dn_stack, &n_block) == 2) {
#endif /* !_MSC_VER */
      n_stack = max(n_stack, dn_stack);
      if(s_height == 0) {
	s_height = n_block;
      }
      problem = create_problem(n_stack, s_height, n_block);
      break;
    }
  }

  if(problem != NULL) {
    int i;
    int n_tier = 0, current_block = 0, current_stack = 0, current_tier = 0;
    blockdata_t *blockdata = calloc((size_t) n_block, sizeof(blockdata_t));

    while(current_stack < n_stack && current_block < n_block
	  && fgets(buf, MAXBUFLEN, fp) != NULL) {
      char *nptr = buf, *endptr;

      while(1) {
	i = (int) strtol(nptr, &endptr, 10);
	if(nptr == endptr) {
	  break;
	} else {
	  nptr = endptr;
	}

	if(n_tier == 0) {
	  problem->n_tier[current_stack] = n_tier = i;
	  problem->s_height = max(problem->s_height, i);
	  if(n_tier == 0) {
	    if(++current_stack == n_stack) {
	      break;
	    }
	  }
	  current_tier = 0;
	} else {
	  blockdata[current_block].s = current_stack;
	  blockdata[current_block].t = current_tier;
	  blockdata[current_block].priority = i;
	  if(++current_block == n_block) {
	    break;
	  }
	  if(++current_tier == n_tier) {
	    if(++current_stack == n_stack) {
	      break;
	    }
	    n_tier = current_tier = 0;
	  }
	}
      }
    }

    if(n_block != current_block) {
      free_problem(problem);
      problem = NULL;
    } else {
      int i;

      qsort((void *) blockdata, n_block, sizeof(blockdata_t), blockdata_comp);
      for(i = 0; i < n_block; i++) {
	problem->position[i].s = blockdata[i].s;
	problem->position[i].t = blockdata[i].t;
	problem->block[blockdata[i].s][blockdata[i].t].no = i;
	problem->block[blockdata[i].s][blockdata[i].t].priority
	  = blockdata[i].priority;
      }

      problem->min_priority = blockdata[0].priority;
      problem->max_priority = blockdata[n_block - 1].priority;
    }

    free(blockdata);
  }

  if(filename != NULL) {
    fclose(fp);
  }

  return(problem);
}

void remove_comments(char *c)
{
  char *a = c;

  for(; *c == ' ' || *c == '\t'; c++);
  for(; *c != '\0' && *c != '#' && *c != '\n'; *a++ = *c, c++);
  *a='\0';
}

int blockdata_comp(const void *a, const void *b)
{
  blockdata_t *x = (blockdata_t *) a;
  blockdata_t *y = (blockdata_t *) b;

  if(x->priority > y->priority) {
    return(1);
  } else if(x->priority < y->priority) {
    return(-1);
  } else if(x->t < y->t) {
    return(1);
  } else if(x->t > y->t) {
    return(-1);
  } else if(x->s > y->s) {
    return(1);
  } else if(x->s < y->s) {
    return(-1);
  }

  return(0);
}
