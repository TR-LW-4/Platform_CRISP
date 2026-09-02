#ifndef heuristic_H
#define heuristic_H

#include <vector>

struct CMMove
{
    int container;
    int src;
    int dst;  // -1 means retrieval

    CMMove(int c, int s, int d) : container(c), src(s), dst(d) {}
};

int block_heuristic(
    std::vector < std::vector <int> > bay,
    int m,
    int h,
    int nels,
    int k,
    std::vector < std::vector< std::vector<int> > > & heurPath,
    std::vector<CMMove> & moves
);
bool find_element(int l, std::vector < std::vector <int> > node, int & row, int & col);
int chkemptystack(std::vector < std::vector <int> > bay, int m);
int min_el_i(std::vector < std::vector <int> > bay, int i);
int max_in_choosestack(int * choosestack, int el, int m);
void print_node(std::vector< std::vector<int> > bay, int m);
#endif
