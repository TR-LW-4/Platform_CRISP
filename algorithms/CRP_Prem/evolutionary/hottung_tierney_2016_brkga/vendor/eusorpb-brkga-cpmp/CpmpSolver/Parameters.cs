using CommandLine;
using CommandLine.Text;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CpmpSolver
{
    public  class Parameters
    {
        [Option('s', "seed", DefaultValue = 1, HelpText = "Seed value")]
        public int GlobalSeed { get; set;}
        private Random _random;
        private int _randomSeed;
        public int GetSeed()
        {
            if (_random == null || _randomSeed != GlobalSeed)
            {
                _random = new Random(GlobalSeed);
                _randomSeed = GlobalSeed;
            }
            return _random.Next();
        }
        [Option('i', "instance", Required = true, HelpText = "Path of instance file")]
        public string InstancePath { get; set; }
        [Option('a', "doOptimalMoves", DefaultValue = true)]
        public bool DoOptimalMoves { get; set; }
        [Option('b', "fillStackConsiderNewClearStacks", DefaultValue = false)]
        public bool FillStackConsiderNewClearStacks { get; set; }
        [Option('c', "fillStackConsiderGiverHeight", DefaultValue = true)]
        public bool FillStackConsiderGiverHeight { get; set; }
        [Option('d', "fillStackConsiderReceiverHeight", DefaultValue = true)]
        public bool FillStackConsiderReceiverHeight { get; set; }
        [Option('e', "clearStackConsiderReceiverHeight", DefaultValue = true)]
        public bool ClearStackConsiderReceiverHeight { get; set; }
        [Option('f', "lowerBoundMultiplicatorForUpperBound", DefaultValue = 1.96275)]
        public double LowerBoundMultiplicatorForUpperBound { get; set; }
        [Option('g', "needStackHeightToBeConsideredSortedBoundry", DefaultValue = 0.4)]
        public double NeedStackHeightToBeConsideredSortedBoundry { get; set; }
        [Option('h', "countClearMoveVariationsMultiplicator", DefaultValue = 2)]
        public int CountClearMoveVariationsMultiplicator { get; set; }
        [Option('j', "clearStackConsiderCleanReceiver", DefaultValue = false)]
        public bool ClearStackConsiderCleanReceiver { get; set; }
        [Option('k', "clearStackConsiderPriorityGap1", DefaultValue = false)]
        public bool ClearStackConsiderPriorityGap1 { get; set; }
        [Option('l', "clearStackConsiderPriorityGap2", DefaultValue = false)]
        public bool ClearStackConsiderPriorityGap2 { get; set; }
        [Option('m', "clearStackCleanPiortiyGap", DefaultValue = true)]
        public bool ClearStackCleanPiortiyGap { get; set; }
        [Option('n', "fillStackConsiderLowestPriority", DefaultValue = false)]
        public bool FillStackConsiderLowestPriority { get; set; }

        [Option('p', "populationSize", DefaultValue = 190)]
        public int PopulationSize { get; set; }
        [Option('q', "elitePopulationSize", DefaultValue = (int)(0.072063*190))]
        public int ElitePopulationSize { get; set; }
        [Option('r', "mutationOffspringSize", DefaultValue = (int)(190*0.097999))]
        public int MutationOffspringSize { get; set; }
        [Option('t', "offspringSize", DefaultValue = (int)(0.685401*190))]
        public int OffspringSize { get; set; }
        [Option('u', "maxGenerationCount", DefaultValue = 1164)]
        public int MaxGenerationCount { get; set; }
        [Option('v', "matingPoolSize", DefaultValue = 200)]
        public int MatingPoolSize { get; set; }
        [Option('w', "maxGenerationsWithoutImprovement", DefaultValue = 74)]
        public int MaxGenerationsWithoutImprovement { get; set; }
        [Option('x', "biasedPercentage", DefaultValue = 65)]
        public int BiasedPercentage { get; set; }
        [Option('y', "maxCpuTime", DefaultValue = 1200)]
        public int MaxCpuTime { get; set; }
        [Option('z', "minShareOfValidSolutionsOfPopulationSize", DefaultValue = 0.210087)]
        public double MinShareOfValidSolutionsOfPopulationSize { get; set; }

        [HelpOption]
        public string GetUsage()
        {
            return HelpText.AutoBuild(this,
     (HelpText current) => HelpText.DefaultParsingErrorsHandler(this, current));

        }
    }
}
