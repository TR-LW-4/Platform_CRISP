#Copyright 2020 Tiziano Bacci, Paolo Ventura
#This file is part of BC-RBRP.

#    BC-RBRP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    BC-RBRP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with BC-RBRP.  If not, see <http://www.gnu.org/licenses/>.

LIBGUROBI = /opt/gurobi811/linux64/lib -lgurobi_c++ -lgurobi81
INCGUROBI = /opt/gurobi811/linux64/include
FLAGS = -O3 -Wno-unused-result -Wno-deprecated-declarations
#FLAGS = -g -fPIC

BC_RBRP.exe: Makefile BC_RBRP.o rBRP_BSheu.o rBRP_MIP3.o
	g++ -DIL_STD $(FLAGS) -o $(BINDIR)BC_RBRP.exe BC_RBRP.o rBRP_BSheu.o rBRP_MIP3.o -L$(LIBGUROBI)

BC_RBRP.o: BC_RBRP.cpp Makefile
	g++ $(FLAGS) -c BC_RBRP.cpp -DMAIN

rBRP_MIP3.o: rBRP_MIP3.cpp rBRP_MIP3.h Makefile
	g++ -DIL_STD $(FLAGS) -c rBRP_MIP3.cpp -I$(INCGUROBI)	

rBRP_BSheu.o: rBRP_BSheu.cpp Makefile
	g++ $(FLAGS) -c rBRP_BSheu.cpp

clean: 
	rm -f *.o *~ *.lp *.log $(BINDIR)BC_RBRP.exe
