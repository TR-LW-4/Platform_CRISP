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
 *  $Id: main.c,v 1.4 2019/06/11 12:17:59 tanaka Exp tanaka $
 *  $Revision: 1.4 $
 *  $Date: 2019/06/11 12:17:59 $
 *  $Author: tanaka $
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "define.h"
#include "print.h"
#include "problem.h"
#include "solution.h"
#include "solve.h"
#include "heuristics.h"

static problem_t *read_file(char *, int, int);
static void usage(char *);
static void string_tolower(char *);
static void remove_comments(char *);
static int blockdata_comp(const void *, const void *);

int main(int argc, char **argv)
{
  char **agv;
  int yard_n_stack, yard_s_height;
  uchar ret;
  problem_t *problem;
  solution_t *solution;
  
  verbose = FALSE;
  lower_bound_only = FALSE;
  tlimit = -1;
  yard_n_stack = yard_s_height = 0;
  for(agv = argv + 1, argc--; argc > 0 && agv[0][0] == '-'; --argc, ++agv) {
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
    case 'l':
      lower_bound_only = TRUE;
      break;
    case 'S':
      if(argc == 1) {
	usage(argv[0]);
	return(1);
      }
      yard_n_stack = (uint) atoi(agv[1]);
      ++agv;
      --argc;
      break;
    case 'T':
      if(argc == 1) {
	usage(argv[0]);
	return(1);
      }
      yard_s_height = (uint) atoi(agv[1]);
      ++agv;
      --argc;
      break;
    case 't':
      if(argc == 1) {
	usage(argv[0]);
	return(1);
      }
      tlimit = (int) atoi(agv[1]);
      ++agv;
      --argc;
      break;
    }
  }

  problem = read_file((argc >= 1)?agv[0]:NULL, yard_n_stack, yard_s_height);

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
  fprintf(stdout, "Usage: %s [-v|-s] [-S S] [-T T] [-l] [-t L] [input file]\n",
	  name);
#ifdef RESTRICTED
  fprintf(stdout, "b&b algorithm for the restricted block relocation problem"
	  "with a stowage plan.\n");
#else /* !RESTRICTED */
  fprintf(stdout, "b&b algorithm for the unrestricted block relocation problem"
	  "with a stowage plan.\n");
#endif /* !RESTRICTED */
  fprintf(stdout, " -v|-s: verbose|silent\n");
  fprintf(stdout, " -l   : lower bound only\n");
  fprintf(stdout, " -S  S: number of yard stacks.\n");
  fprintf(stdout, " -T  T: maximum number of yard stack tiers.\n");
  fprintf(stdout, " -t  L: time limit.\n");
  fprintf(stdout, "\n");
}

static char *key_list[] = {
  "yard bay",
  "number of containers",
  "number of yard stacks",
  "tier of yard stacks",
  "number of vessel stacks",
  "tier of vessel stacks",
  "",
};

static char stack_label[]
= "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

typedef struct {
  int no;
  coordinate_t yard;
  coordinate_t ship;
} blockdata_t;

problem_t *read_file(char *filename, int yard_n_stack, int yard_s_height)
{
  int i;
  int dyard_n_stack = 0, yard_n_block = 0;
  int ship_n_stack = 0, ship_n_block = 0;
  int type = -1, file_type = 0;
  FILE *fp;
  char buf[MAXBUFLEN];
  problem_t *problem;
  blockdata_t *blockdata;

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
#ifdef _MSC_VER
    }
#else /* !_MSC_VER */
    }
#endif /* !_MSC_VER */
  }

  problem = NULL;
  while(fgets(buf, MAXBUFLEN, fp) != NULL) {
    remove_comments(buf);
    string_tolower(buf);

    for(i = 0; key_list[i][0] != '\0'; ++i) {
      if(strstr(buf, key_list[i]) != NULL) {
	type = i;
	break;
      }

    }

    if(type >= 0) {
      file_type = 1;
      break;
    }

#ifdef _MSC_VER
    if(sscanf_s(buf, "%d %d %d %d",
		&dyard_n_stack, &yard_n_block,
		&ship_n_stack, &ship_n_block) == 4) {
#else /* !_MSC_VER */
    if(sscanf(buf, "%d %d %d %d",
	      &dyard_n_stack, &yard_n_block,
	      &ship_n_stack, &ship_n_block) == 4) {
#endif /* !_MSC_VER */
      yard_n_stack = max(yard_n_stack, dyard_n_stack);
      if(yard_s_height == 0) {
	yard_s_height = yard_n_block;
      }
      problem = create_problem(yard_n_stack, yard_s_height, yard_n_block,
			       ship_n_stack, ship_n_block, 0, 0);
      break;
#ifdef _MSC_VER
    }
#else /* !_MSC_VER */
    }
#endif /* !_MSC_VER */
  }

  if(file_type == 1) {
    int c = 0;
    int dyard_s_height = 0;
    int max_ship_tier = 0;
    int *ship_n_tier = NULL;

    while(type != 0 && fgets(buf, MAXBUFLEN, fp) != NULL) { 
      remove_comments(buf);
      string_tolower(buf);

      if(type == -1) {
	for(i = 0; key_list[i][0] != '\0'; ++i) {
	  if(strstr(buf, key_list[i]) != NULL) {
	    type = i;
	    break;
	  }
	}
      } else {
	i = (int) strtol(buf, NULL, 10);

	switch(type) {
	default:
	case 1:
	  yard_n_block = ship_n_block = i;
	  type = -1;
	  break;
	case 2:
	  dyard_n_stack = i;
	  type = -1;
	  break;
	case 3:
	  dyard_s_height = i;
	  type = -1;
	  break;
	case 4:
	  ship_n_stack = i;
	  ship_n_tier = (int *) calloc((size_t) ship_n_stack, sizeof(int));
	  type = -1;
	  break;
	case 5:
	  if(ship_n_tier == NULL) {
	    fprintf(stderr, "%s field should be specified before %s field\n",
		    key_list[4], key_list[5]);
	    exit(1);
	  }

	  ship_n_tier[c++] = i;
	  if(c == ship_n_stack) {
	    type = -1;
	  }
	  max_ship_tier = max(max_ship_tier, i);

	  break;
	}
      }
    }

    yard_s_height = max(yard_s_height, dyard_s_height);
    yard_n_stack = max(yard_n_stack, dyard_n_stack);

    if(yard_n_block > 0 && yard_n_stack > 0
       && yard_s_height > 0 && ship_n_stack > 0) {
      problem = create_problem(yard_n_stack, yard_s_height, yard_n_block,
			       ship_n_stack, ship_n_block, 1, max_ship_tier);
      memcpy((void *) problem->ship_n_tier, (void *) ship_n_tier,
	     (size_t) ship_n_stack*sizeof(int));
    }

    free(ship_n_tier);
  }

  if(problem == NULL) {
    goto read_problem_end;
  }

  blockdata = calloc((size_t) yard_n_block, sizeof(blockdata_t));

  if(file_type == 0) {
    int n_tier = -1, current_block = 0, current_stack = 0, current_tier = 0;

    while(current_stack < dyard_n_stack && fgets(buf, MAXBUFLEN, fp) != NULL) {
      char *nptr, *endptr;

      remove_comments(buf);
      if(*buf == '\0' || *buf == '\r') {
	continue;
      }

      nptr = buf;

      while(1) {
	i = (int) strtol(nptr, &endptr, 10);
	if(nptr == endptr) {
	  break;
	} else {
	  nptr = endptr;
	}

	if(n_tier == -1) {
	  problem->yard_n_tier[current_stack] = n_tier = i;
	  problem->yard_s_height = max(problem->yard_s_height, i);
	  if(n_tier == 0) {
	    if(++current_stack == dyard_n_stack) {
	      break;
	    }
	    n_tier = -1;
	  }
	  current_tier = 0;
	} else {
	  if(current_block == yard_n_block) {
	    break;
	  }

	  blockdata[current_block].no = i;
	  blockdata[current_block].yard.s = current_stack;
	  blockdata[current_block].yard.t = current_tier;
	  blockdata[current_block].ship.s = -1;
	  blockdata[current_block++].ship.t = -1;
	  if(++current_tier == n_tier) {
	    if(++current_stack == dyard_n_stack) {
	      break;
	    }
	    n_tier = -1;
	    current_tier = 0;
	  }
	}
      }
    }

    if(yard_n_block != current_block) {
      free_problem(problem);
      problem = NULL;
      free(blockdata);

      goto read_problem_end;
    }

    n_tier = current_block = current_stack = current_tier = 0;

    while(current_stack < ship_n_stack && current_block < ship_n_block
	  && fgets(buf, MAXBUFLEN, fp) != NULL) {
      char *nptr, *endptr;
      remove_comments(buf);
      if(*buf == '\0' || *buf == '\r') {
	continue;
      }
      nptr = buf;

      while(1) {
	i = (int) strtol(nptr, &endptr, 10);
	if(nptr == endptr) {
	  break;
	} else {
	  nptr = endptr;
	}

	if(n_tier == 0) {
	  problem->ship_n_tier[current_stack] = n_tier = i;

	  if(n_tier == 0) {
	    if(++current_stack == ship_n_stack) {
	      break;
	    }
	  }
	  current_tier = 0;
	} else {
	  int j;

	  for(j = 0; j < yard_n_block && blockdata[j].no != i; ++j);
	  if(j == yard_n_block) {
	    current_stack = ship_n_stack;
	    break;
	  }

	  blockdata[j].ship.s = current_stack;
	  blockdata[j].ship.t = current_tier;
	  if(++current_block == ship_n_block) {
	    break;
	  }
	  if(++current_tier == n_tier) {
	    if(++current_stack == ship_n_stack) {
	      break;
	    }
	    n_tier = current_tier = 0;
	  }
	}
      }
    }

    if(ship_n_block != current_block) {
      free_problem(problem);
      problem = NULL;
      free(blockdata);

      goto read_problem_end;
    }

    qsort((void *) blockdata, yard_n_block, sizeof(blockdata_t),
	  blockdata_comp);
  } else {
    int current_block = 0, current_stack = 0, current_tier = 0;
    int *cumulative_ship_n_tier
      = malloc((size_t) (ship_n_stack + 1)*sizeof(int));

    cumulative_ship_n_tier[0] = 0;
    for(i = 0; i < ship_n_stack; ++i) {
      cumulative_ship_n_tier[i + 1]
	= cumulative_ship_n_tier[i] + problem->ship_n_tier[i];
    }

    while(fgets(buf, MAXBUFLEN, fp) != NULL) {
      remove_comments(buf);
      if(*buf == '\0' || *buf == '\r') {
	continue;
      }

      if(strstr(buf, "Stack") != NULL || strstr(buf, "stack") != NULL) {
	current_tier = 0;
	current_stack = (int) strtol(buf + strlen("stack"), NULL, 10);

	if(current_stack >= yard_n_stack) {
	  free(cumulative_ship_n_tier);
	  free(blockdata);
	  free_problem(problem);
	  problem = NULL;

	  goto read_problem_end;
	}
      } else {
	char *s = buf, *nptr;

	while(*s != '\0') {
	  char *p;
	  int ship_stack = 0, ship_tier = 0;

	  for(; *s == ' '; ++s);

	  if((p = strchr(stack_label, (int) *s)) != NULL) {
	    ship_stack = (int) (p - stack_label);
	    problem->name[current_block][0] = *s;
	  } else {
	    ++s;
	    continue;
	  }

	  if(*++s != '_') {
	    continue;
	  }
	  ++s;

	  ship_tier = (int) strtol(s, &nptr, 10);
	  if(nptr == s) {
	    continue;
	  }

	  if(current_block > yard_n_block
	     || ship_tier > problem->ship_n_tier[ship_stack]) {
	    free(cumulative_ship_n_tier);
	    free(blockdata);
	    free_problem(problem);
	    problem = NULL;

	    goto read_problem_end;
	  }

	  for(i = 1; s < nptr;
	      problem->name[current_block][i] = *s, ++i, ++s);
	  for(; i < problem->name_length;
	      problem->name[current_block][i++] = ' ');
	  blockdata[current_block].no
	    = cumulative_ship_n_tier[ship_stack] + ship_tier;
	  blockdata[current_block].yard.s = current_stack;
	  blockdata[current_block].yard.t = current_tier++;
	  blockdata[current_block].ship.s = ship_stack;
	  blockdata[current_block++].ship.t = ship_tier;

	  problem->yard_n_tier[current_stack] = current_tier;
	}
      }
    }

    if(yard_n_block != current_block) {
      free_problem(problem);
      problem = NULL;
      free(blockdata);

      goto read_problem_end;
    }

    free(cumulative_ship_n_tier);
  }

  problem->yard_block[0]
    = (int *) calloc((size_t) (problem->yard_n_stack*problem->yard_s_height
			       + problem->ship_n_block), sizeof(int));
  for(i = 1; i < problem->yard_n_stack; ++i) {
    problem->yard_block[i]
      = problem->yard_block[i - 1] + problem->yard_s_height;
  }

  problem->ship_block[0]
    = problem->yard_block[0] + problem->yard_n_stack*problem->yard_s_height;
  for(i = 1; i < problem->ship_n_stack; ++i) {
    problem->ship_block[i]
      = problem->ship_block[i - 1] + problem->ship_n_tier[i - 1];
  }

  if(file_type == 0) {
    char stack_name_format[32];
#ifdef _MSC_VER
    _snprintf_s(stack_name_format, 32,
		_TRUNCATE, "%%%dd", problem->name_length);
#else /* _MSC_VER */
    snprintf(stack_name_format, 32, "%%%dd", problem->name_length);
#endif /* _MSC_VER */

    for(i = 0; i < yard_n_block; ++i) {
      problem->no[i] = blockdata[i].no;
      if(file_type == 0) {
#ifdef _MSC_VER
	_snprintf_s(prob->name[i], problem->name_length + 1, _TRUNCATE,
		    (const char *) stack_name_format, problem->no[i]);
#else /* _MSC_VER */
	snprintf(problem->name[i], problem->name_length + 1,
		 (const char *) stack_name_format, problem->no[i]);
#endif /* _MSC_VER */
      }
    }
  }

  for(i = 0; i < yard_n_block; ++i) {
    problem->yard_position[i] = blockdata[i].yard;
    problem->ship_position[i] = blockdata[i].ship;
    problem->yard_block[blockdata[i].yard.s][blockdata[i].yard.t] = i;
    if(blockdata[i].ship.s >= 0) {
      problem->ship_block[blockdata[i].ship.s][blockdata[i].ship.t] = i;
    }
  }

  free(blockdata);

  read_problem_end:

  if(filename != NULL) {
    fclose(fp);
  }

  return(problem);
}

void string_tolower(char *str)
{
  char *c;

  for(c =str; *c != '\0'; ++c) {
    *c = tolower((unsigned char) *c);
  }
}

void remove_comments(char *c)
{
  char *a = c;

  for(; *c == ' ' || *c == '\t'; ++c);
  for(; *c != '\0' && *c != '#' && *c != '\n'; *a++ = *c, ++c);
  *a='\0';
}

int blockdata_comp(const void *a, const void *b)
{
  blockdata_t *x = (blockdata_t *) a;
  blockdata_t *y = (blockdata_t *) b;

  if((x->ship).s > (y->ship).s) {
    return(1);
  } else if((x->ship).s < (y->ship).s) {
    return(-1);
  } else if((x->ship).t > (y->ship).t) {
    return(1);
  } else if((x->ship).t < (y->ship).t) {
    return(-1);
  }

  return(0);
}
