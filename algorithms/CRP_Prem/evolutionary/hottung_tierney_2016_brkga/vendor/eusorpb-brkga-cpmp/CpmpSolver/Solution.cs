using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CpmpSolver
{
    public class Solution
    {
        public Bay BayInstance;
        public double[] Chromosome;
        protected List<Tuple<int, int>> Moves = new List<Tuple<int, int>>();
        public List<int> BestSolutionMoves = new List<int>();
        public int MovesCount = int.MaxValue;
        public int ChromosomeLength;
    }
}
