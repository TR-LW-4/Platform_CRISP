//
// Copyright 2019-2021 Shunji Tanaka and Stefan Voss.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//   1. Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//   2. Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials
//      provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  $Id: solve.cpp,v 1.8 2021/03/15 04:11:58 tanaka Exp tanaka $
//  $Revision: 1.8 $
//  $Date: 2021/03/15 04:11:58 $
//  $Author: tanaka $
//
//
#include <cstdlib>
#include <iostream>

#include "instance.hpp"
#include "ipmodel.hpp"
#include "solve.hpp"
#include "solution.hpp"

int solve(const Instance& instance, const Parameter& parameter)
{
  IPModel m(instance);

  if(m.solve(parameter)) {
    std::cerr << "solved" << std::endl;
  } else {
    std::cerr << "not solved" << std::endl;
  }

  std::cerr << "iterations=" << m.number_of_iterations() << std::endl;
  std::cerr << "total_time=" << m.total_time() << std::endl;
  if(parameter.upperBounding) {
    std::cerr << "lb_time=" << m.lb_time() << std::endl;
    std::cerr << "ub_time=" << m.ub_time() << std::endl;
  }
  std::cerr << "total_ip_time=" << m.total_ip_time() << std::endl;
  std::cerr << "initial_number_of_variables=";
  std::cerr << m.initial_number_of_variables() << std::endl;
  std::cerr << "initial_number_of_constraints=";
  std::cerr << m.initial_number_of_constraints() << std::endl;
  std::cerr << "final_number_of_variables=";
  std::cerr << m.final_number_of_variables() << std::endl;
  std::cerr << "final_number_of_constraints=";
  std::cerr << m.final_number_of_constraints() << std::endl;

  if(m.lower_bound() == m.upper_bound()) {
    std::cerr << "optimal_value=" << m.lower_bound() << std::endl;
  } else {
    std::cerr << "lower_bound=" << m.lower_bound() << std::endl;
    if(m.upper_bound() < 0) {
      std::cerr << "upper_bound=infeasible" << std::endl;
    } else {
      std::cerr << "upper_bound=" << m.upper_bound() << std::endl;
    }
  }

  if(m.upper_bound() != -1) {
    try {
      Solution* sol = m.best_solution();
      sol->print(std::cout, instance);
      return EXIT_SUCCESS;
    } catch(const char *e) {
      std::cerr << e << std::endl;
    }
  }

  return EXIT_FAILURE;
}
