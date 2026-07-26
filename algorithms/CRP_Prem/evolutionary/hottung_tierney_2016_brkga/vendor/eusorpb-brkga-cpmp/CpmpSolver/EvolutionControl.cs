using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;

namespace CpmpSolver
{
    public class EvolutionControl
    {
        public static int PopulationSize = 100;
        public static int ElitePopulationSize = 30;
        public static int MutationOffspringSize = 40;
        public static int OffspringSize = 80;
        public static int MaxGenerationCount = 3500;//2000;
        public static int MatingPoolSize = 200;
        public static int MaxGenerationsWithoutImprovement = 100;//50;
        public static int BiasedPercentage = 80;
        //Max CPU time in seconds
        public static int MaxCpuTime = 60;
        //Required share of solutions that create a clear bay
        public static double MinShareOfValidSolutionsOfPopulationSize = 0.2;
        //Length of the chromosome
        //public static readonly Random MyRandom = new Random(Parameters.GetSeed());     
        public static FastRandom MyRandom;
        //Best chromosome found so far
        public static Solution Solution;
        //Lowest priority of all containers (highest value)
        public static int LowestPriorityValue;
        //Lower bound for the solution
        public static int LowerBound;



        public static int MaxNumberOfCreationTries = 50000;
        /// <summary>
        /// Generate the initial solution 
        /// </summary>
        /// <returns></returns>
        private static List<Solution> GeneratePopulation(Bay bay, int size, int minCompleteSolutions, Parameters parameters)
        {
            var population = new List<Solution>();
            int completeSolutions = 0;
            int creationTry = 0;
            //While there are not enough individuals in the population add a random individual
            while (population.Count != size)
            {
               var solution = new SolutionFillClearS2(bay,parameters);
               creationTry++;
                if (solution.MovesCount != int.MaxValue)
                    completeSolutions++;               
                if(completeSolutions < minCompleteSolutions && population.Count == size-(minCompleteSolutions-completeSolutions)
                    && creationTry < MaxNumberOfCreationTries)
                    continue;
               population.Add(solution);
               if (solution.MovesCount == bay.LowerBound)
                   break;
            }
            return population;
        }

        /// <summary>
        /// Combine to parents to a child
        /// </summary>
        /// <param name="parent1">Parent from the elitists</param>
        /// <param name="parent2">Parent form the non-elitists</param>
        /// <returns>Child chromosome</returns>
        private static Solution CrossOver(Solution parent1, Solution parent2,Parameters parameters)
        {
            var child = new double[parent1.ChromosomeLength];
            //Iterate over each key
            for (int i = 0; i < parent1.ChromosomeLength; i++)
            {
                //For each key there is a change to be copied to the child
                int r = MyRandom.Next(0, 100);
                if (r < BiasedPercentage)
                    child[i] = parent1.Chromosome[i];
                else
                    child[i] = parent2.Chromosome[i];
            }
            return new SolutionFillClearS2(parent1.BayInstance,child,parameters);
        }

        /// <summary>
        /// Main loop
        /// </summary>
        /// <param name="bay"></param>
        /// <returns></returns>
        public static Solution DoEvolution(Bay bay,Parameters parameters)
        {            
            var begin = Process.GetCurrentProcess().TotalProcessorTime;
            MyRandom = new FastRandom(parameters.GetSeed());
            int lowerBound = bay.LowerBound;
            Console.WriteLine("Lower bound: "+lowerBound);
            List<Solution> population = GeneratePopulation(bay, PopulationSize,(int) (PopulationSize*MinShareOfValidSolutionsOfPopulationSize),parameters); //TODO to parameter
            Solution bestSolution = population.OrderByDescending(s => s.MovesCount).Last();
            Console.WriteLine("Incumbent update: " + bestSolution.MovesCount);
            if (bestSolution.MovesCount == bay.LowerBound)
                return bestSolution;
            int generationCount = 0;
            int generationsWithoutImprovement = 0;

            while (generationCount < MaxGenerationCount &&
                   generationsWithoutImprovement < MaxGenerationsWithoutImprovement &&
                   (Process.GetCurrentProcess().TotalProcessorTime - begin).TotalSeconds < MaxCpuTime)
            {

                generationCount++;
                //Crossover
                var offspring = new List<Solution>();
                //Create the offspring by selecting random parents
                while (offspring.Count < OffspringSize)
                {
                    Solution parent1 = population.ElementAt(MyRandom.Next(0, ElitePopulationSize));
                    Solution parent2 = population.ElementAt(MyRandom.Next(ElitePopulationSize, population.Count));
                    Solution child = CrossOver(parent1, parent2, parameters);
                    offspring.Add(child);
                }
                List<Solution> mutationOffspring = GeneratePopulation(bay, MutationOffspringSize,0,parameters);
                //Create the offspring by creating random chromosomes
                
                //Order them by their fitness
                population = population.OrderBy(c => c.MovesCount).ToList();
                //Remove all chromosomes but the best
                population.RemoveRange(ElitePopulationSize, population.Count - ElitePopulationSize);
                //Add offspring and mutation chromosomes to population
                population.AddRange(offspring);
                population.AddRange(mutationOffspring);
                //Find the best chromosome of this population
                int bestOfGeneration =
                    population.OrderByDescending(c => c.MovesCount).Last().MovesCount;
                //Console.WriteLine(bestOfGeneration);
                //Update incumbent
                if (bestOfGeneration < bestSolution.MovesCount)
                {
                    bestSolution = population.OrderByDescending(c => c.MovesCount).Last();
                    Console.WriteLine("Incumbent update: " + bestSolution.MovesCount);
                    //writer.WriteLine((Process.GetCurrentProcess().TotalProcessorTime - begin).TotalSeconds+";"+generationCount+";"+generationsWithoutImprovement+";"+improvement);
                    generationsWithoutImprovement = -1;
                    if(bestSolution.MovesCount < bay.LowerBound)
                        throw new Exception("Lower bound is to high");
                    if (bestSolution.MovesCount == bay.LowerBound)
                        return bestSolution;
                }
                generationsWithoutImprovement++;

            }
            return bestSolution;
        }
    }
}

