using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CpmpSolver
{
    public class Bay
    {
        /// <summary>
        /// Width of the bay
        /// </summary>
        public int Width;

        /// <summary>
        /// Height of the bay
        /// </summary>
        public int Height;

        /// <summary>
        /// Count of containers
        /// </summary>
        public int ContainerCount;

        /// <summary>
        /// The array representing the arrangement of containers in the bay.
        /// If there is now container on a given stack at a given height the value is -1
        /// </summary>
        public int[][] BayArray;

        /// <summary>
        /// The current height of a stack
        /// </summary>
        public int[] CurStackHeight;

        public int LowerBound;

        public int LowestPriority;
        public int HeighestPiority;

        

        /// <summary>
        /// Generate a new bay object
        /// </summary>
        /// <param name="width">Width of the bay</param>
        /// <param name="height">Height of the bay</param>
        /// <param name="containerCount">Count of containers</param>
        /// <param name="bayArray">The array representing the arrangement of containers in the bay</param>
        /// <param name="curStackHeight">The current height of a stack</param>
        public Bay(int width, int height, int containerCount, int[][] bayArray, int[] curStackHeight)
        {
            BayArray = bayArray;
            Width = width;
            Height = height;
            ContainerCount = containerCount;
            CurStackHeight = curStackHeight;
            LowestPriority = bayArray.Max(s => s.Max());
            HeighestPiority = bayArray.Where(s => s.Max() != -1).Min(s => s.Where(ss => ss != -1).Min());
            LowerBound = Functions.ComputeLowerBound(this.GetCopy());
        }

        public BayCopy GetCopy()
        {
            return new BayCopy(this,BayArray,CurStackHeight);
        }
    }

    public class BayCopy
    {
        public Bay OriginalBay;
        /// <summary>
        /// The array representing the arrangement of containers in the bay.
        /// If there is now container on a given stack at a given height the value is -1
        /// </summary>
        public int[][] BayArray;

        /// <summary>
        /// The current height of a stack
        /// </summary>
        public int[] CurStackHeight;

        public BayCopy(Bay bay, IEnumerable<int[]> bayArray, int[] curStackHeight)
        {
            OriginalBay = bay;
            BayArray = bayArray.Select(s => s.ToArray()).ToArray();
            CurStackHeight = (int[])curStackHeight.Clone();
        }

        public  bool StackIsSorted(int stack)
        {
            int[] s = BayArray[stack];
            //An empty stack is always sorted
            if (s[0] == -1)
                return true;
            for (int z = 1; z < s.Length; z++)
            {
                //Return false if a container with a higher priority is above a container with a lower priority
                if (s[z] > s[z-1])
                {
                    return false;
                }
            }
            return true;
        }

        public int StackIsSortedTo(int stack)
        {
            int[] s = BayArray[stack];
            //An empty stack is always sorted
            if (s[0] == -1)
                return -1;
            for (int z = 1; z < s.Length; z++)
            {
                //Return false if a container with a higher priority is above a container with a lower priority
                if (s[z] > s[z - 1])
                {
                    return z;
                }
            }
            return -1;
        }

        /// <summary>
        /// Returns true if the containers are well sorted
        /// </summary>
        /// <param name="bay"></param>
        /// <returns></returns>
        public bool IsSorted()
        {
            //Iterate other each bay
            for (int i = 0; i < OriginalBay.Width; i++)
            {
                if(!StackIsSorted(i))
                    return false;
            }
            return true;
        }

        private int _temp;
        /// <summary>
        /// Moves one container from a given stack to another given stack
        /// </summary>
        /// <param name="stackFrom">The source stack</param>
        /// <param name="stackTo">The target stack</param>
        public void Move(int stackFrom, int stackTo)
        {
            _temp = BayArray[stackFrom][CurStackHeight[stackFrom]];
            BayArray[stackFrom][CurStackHeight[stackFrom]] = -1;
            CurStackHeight[stackFrom]--;

            BayArray[stackTo][CurStackHeight[stackTo]+1] = _temp;
            CurStackHeight[stackTo]++;
        }
    }
}
