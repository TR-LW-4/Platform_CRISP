using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Mime;
using System.Text;
using System.Threading.Tasks;

namespace CpmpSolver
{
    class Program
    {
        private const String Version = "0.1.6";
        private const String SolutionType = "FillClearS2";


        static void Main(string[] args)
        {           
            RunWithParameters(args);
        }



        static void RunWithParameters(string[] args)
        {
            Parameters parameters = new Parameters();
            CommandLine.Parser.Default.ParseArguments(args, parameters);

            EvolutionControl.PopulationSize = parameters.PopulationSize;
            EvolutionControl.ElitePopulationSize = parameters.ElitePopulationSize;
            EvolutionControl.MutationOffspringSize = parameters.MutationOffspringSize;
            EvolutionControl.BiasedPercentage = parameters.BiasedPercentage;
            EvolutionControl.OffspringSize = parameters.OffspringSize;
            EvolutionControl.MaxGenerationCount = parameters.MaxGenerationCount;
            EvolutionControl.MaxGenerationsWithoutImprovement = parameters.MaxGenerationsWithoutImprovement;
            EvolutionControl.MaxCpuTime = parameters.MaxCpuTime;
            EvolutionControl.MinShareOfValidSolutionsOfPopulationSize = parameters.MinShareOfValidSolutionsOfPopulationSize;
                 
            ProcessFile(parameters.InstancePath,parameters);

        }
       
        /// <summary>
        /// Process a .bay file
        /// </summary>
        /// <param name="path"></param>
        public static void ProcessFile(string path,Parameters parameters)
        {
            Bay bay;
            //Read the file
            try
            {
                bay = BayReader.ReadBayFile(path);
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception reading" + path + " bay file:" + e.Message);
                return;
            }
            Console.WriteLine("Start optimization of " + path);
            TimeSpan begin = Process.GetCurrentProcess().TotalProcessorTime;
            Solution solution = EvolutionControl.DoEvolution(bay,parameters);
            double duration = (Process.GetCurrentProcess().TotalProcessorTime - begin).TotalSeconds;
            Console.WriteLine("### PM Solution");
            Console.WriteLine("### Duration: " + duration);
            Console.WriteLine("### Number of moves: " + solution.MovesCount);
            for (int i = 0; i < solution.BestSolutionMoves.Count; i = i + 2)
            {
                Console.WriteLine("### Move " + (i / 2 + 1) + ": (" + solution.BestSolutionMoves.ElementAt(i) + "," + solution.BestSolutionMoves.ElementAt(i + 1) + ")");
            }
            //Make sure that the solution is valid
            BayCopy bayCopy = bay.GetCopy();
            Functions.DoLastMoves(bayCopy, solution.BestSolutionMoves);
            if (!bayCopy.IsSorted())
                throw new Exception("Something went wrong here. Solution is not valid!");
        }

    }
}
