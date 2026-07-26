using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CpmpSolver
{
    public class Functions
    {
        public static int ComputeLowerBound(BayCopy bay)
        {
            int overstowage = 0;
            int lowestPiority = bay.OriginalBay.LowestPriority;
            var lbDemand = new Dictionary<int, int>(lowestPiority + 1);
            var lbSupply = new Dictionary<int, int>(lowestPiority + 1);
            for (int i = 0; i <= lowestPiority + 1; i++)
            {
                lbDemand[i] = 0;
                lbSupply[i] = 0;
                
            }
            var empty = new Dictionary<int, bool>(bay.OriginalBay.Width);
            var lbHighestWplaced = new Dictionary<int, int>(bay.OriginalBay.Width);
            var lbStackHasOverstowage = new Dictionary<int, bool>(bay.OriginalBay.Width);
            for (int i = 0; i <= bay.OriginalBay.Width; i++)
            {
                lbHighestWplaced[i] = -1;
                lbStackHasOverstowage[i] = false;               
            }
            for (int i = 0; i < bay.OriginalBay.Width; i++)
            {
                if (bay.BayArray[i][0] == -1)
                    empty[i] = true;
                else
                    empty[i] = false;
            }
            int stackOverstowed;
            int minBadlyPlaced = int.MaxValue;
            int supplyValue = 0;

            for (int ss = 0; ss < bay.OriginalBay.Width; ++ss)
            {
                bool afterHighestWp = false;
                bool overstowed = false;
                stackOverstowed = 0;
                int bottom = bay.BayArray[ss][0];
                if (bottom > -1)
                {
                    lbHighestWplaced[ss] = bottom;
                }

                for (int tt = 1; tt < bay.OriginalBay.Height; tt++)
                {
                    int current = bay.BayArray[ss][tt];
                    //if (current == -1)
                    //    continue;
                    int below = bay.BayArray[ss][tt - 1];


                    if (!overstowed && (below < current))
                    {
                        supplyValue = below;
                        lbHighestWplaced[ss] = below;
                        overstowed = true;
                        lbStackHasOverstowage[ss] = true;
                        afterHighestWp = true;
                    }
                    if (!afterHighestWp && bottom > -1 && current == -1)
                    {
                        afterHighestWp = true;
                        supplyValue = below;
                    }
                    if (afterHighestWp)
                    {
                        lbSupply[supplyValue]++;
                    }
                    if (overstowed)
                    {
                        if (current != -1)
                        {
                            lbDemand[current]++;
                            stackOverstowed++;
                        }
                    }
                    else if (current > -1)
                    {
                        lbHighestWplaced[ss] = current;
                    }
                }
                overstowage += stackOverstowed;
                minBadlyPlaced = Math.Min(minBadlyPlaced, stackOverstowed);
            }
            overstowage += minBadlyPlaced;
            // Compute the maximum cumulative demand surplus g*
            // Note that D(g) is computed on the fly by accumulating the values in the demand vector
            int cmDemand = 0;
            int numEmptyStacks = empty.Count(s => s.Value==true);

            int cmSupply = numEmptyStacks*bay.OriginalBay.Height;
            int maxCmDmdSup = 0;
            int gstar = 0;
            for (int gg = lowestPiority; gg > -1; --gg)
            {
                cmDemand += lbDemand[gg];
                cmSupply += lbSupply[gg];
                int tmpCmDmdSup = cmDemand - cmSupply;
                if (tmpCmDmdSup > maxCmDmdSup)
                {
                    maxCmDmdSup = tmpCmDmdSup;
                    gstar = gg;
                }
            }
            int nsgx = Math.Max(0, (int) Math.Ceiling(maxCmDmdSup/(double) bay.OriginalBay.Height));
            var ngstar = new List<int>();


            for (int ss = 0; ss < bay.OriginalBay.Width; ++ss)
            {
                if (lbHighestWplaced[ss] < gstar)
                {
                    ngstar.Add(0);
                    if (!empty[ss])
                    {
                        // iterate up stack ss until the current tier is overstowing
                        for (int tt = 0;
                            tt < bay.OriginalBay.Height && bay.BayArray[ss][tt] > -1 &&
                            (tt == 0 || bay.BayArray[ss][tt] <= bay.BayArray[ss][tt - 1]);
                            tt++)
                        {
                            if (bay.BayArray[ss][tt] < gstar)
                                ngstar[ngstar.Count - 1]++;
                        }
                    }
                }
            }
            ngstar.Sort();
            for (int ii = 0; ii < nsgx; ++ii)
            {
                overstowage += ngstar[ii];
            }
            return overstowage;
        }

        /// <summary>
        /// Returns true if the containers are well sorted
        /// </summary>
        /// <param name="bay"></param>
        /// <returns></returns>
        public static bool IsSorted(BayCopy bay)
        {
            //Iterate other each bay
            for (int i = 0; i < bay.OriginalBay.Width; i++)
            {
                //Priority of the container below
                int containerBelow = bay.BayArray[i][0];
                for (int z = 1; z < bay.CurStackHeight[i]; z++)
                {
                    //Return false if a container with a higher priority is above a container with a lower priority
                    if (bay.BayArray[i][z] > containerBelow)
                        return false;
                    containerBelow = bay.BayArray[i][z];
                }
            }
            return true;
        }



        /// <summary>
        /// Checks whether a stack is sorted
        /// </summary>
        /// <param name="stack">The stack as an array</param>
        /// <returns>Returns true if the stack is sorted</returns>
        public static bool StackIsSorted(int[] stack)
        {
            //An empty stack is always sorted
            if (stack[0] == -1)
                return true;
            int containerBelow = stack[0];
            for (int z = 1; z < stack.Length; z++)
            {
                //Return false if a container with a higher priority is above a container with a lower priority
                if (stack[z] > containerBelow)
                {
                    return false;
                }
                containerBelow = stack[z];
            }
            return true;
        }

        public static double CleanSupply(BayCopy bay)
        {
            double value = 0;
            int[] slotsG = new int[bay.OriginalBay.LowestPriority+1];
            for (int i = 0; i < bay.OriginalBay.Width; i++)
            {
                if (bay.StackIsSorted(i))
                {
                    if (bay.CurStackHeight[i] != -1)
                    {
                        slotsG[bay.BayArray[i][bay.CurStackHeight[i]]] += bay.OriginalBay.Height -1 - bay.CurStackHeight[i];
                    }
                    else
                    {
                        slotsG[bay.OriginalBay.LowestPriority] += bay.OriginalBay.Height;
                    }

                }
            }
            for (int i = 1; i <= bay.OriginalBay.LowestPriority;i++)
            {
                value += Math.Pow(10, i-1)*slotsG[i];
            }
            if(value < 0)
                throw new Exception("overflow");
            return value;
        }

        /// <summary>
        /// Moves containers in the given order
        /// </summary>
        /// <param name="bay">The bay</param>
        /// <param name="moves">The moves that should be done as a list</param>
        public static void DoLastMoves(BayCopy bay, List<int> moves)
        {
            //Do each move in the list
            for (int i = 0; i < moves.Count; i = i+2)
            {
                bay.Move(moves[i], moves[i+1]);
            }
        }
    }
}
