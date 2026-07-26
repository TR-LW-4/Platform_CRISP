using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CpmpSolver
{
    public class BayReader
    {
        public static Bay ReadBayFile(String path)
        {
            string[] lines = System.IO.File.ReadAllLines(path);

            //read the first lines of the file
            int l = 0;
            int height;
            int width;
            int containerCount;
            if (lines[1].StartsWith("Description"))
            {
                height =
                    Convert.ToInt32(lines[3].Substring(lines[3].IndexOf(":", StringComparison.Ordinal) + 2)) + 2;
                width =
                    Convert.ToInt32(lines[4].Substring(lines[4].IndexOf(":", StringComparison.Ordinal) + 2));
                containerCount =
                    Convert.ToInt32(lines[5].Substring(lines[5].IndexOf(":", StringComparison.Ordinal) + 2));
                l = 6;
            }
            else if (lines[1].Contains(':'))
            {
                l = 4;
                width =
                    Convert.ToInt32(lines[1].Substring(lines[1].IndexOf(":", StringComparison.Ordinal) + 2));
                height =
                    Convert.ToInt32(lines[2].Substring(lines[2].IndexOf(":", StringComparison.Ordinal) + 2));
                containerCount =
                    Convert.ToInt32(lines[3].Substring(lines[3].IndexOf(":", StringComparison.Ordinal) + 2));
            }
            else
            {
                width = Convert.ToInt32(lines[0].Split(' ').First());
                height = lines.Max(c => c.Split(' ').Count()) +1;
                containerCount = Convert.ToInt32(lines[0].Split(' ').Last()); ;
                l = 1;
            }

            var bayArray = new int[width][];
            var curStackHeight = new int[width];
            //Iterate other all stacks/lines
            for (int i = 0; i < width; i++)
            {
                curStackHeight[i] = -1;
                string[] stackString;
                //Read each line representing the containers on a stack
                if(l!=1)
                    stackString = lines[i+l].Substring(lines[i+l].IndexOf(":", StringComparison.Ordinal) + 2).Split(' ');
                else
                    stackString = lines[i+l].Split(' ').Skip(1).ToArray();
                var stack = new int[height];
                int stackHeight = -1;
                for (int z = 0; z < Math.Max(stackString.Length,height); z++)
                {
                    //If there are no containers on stack i at the height z set the stack value to -1
                    if (z < height)
                    {
                        stack[z] = -1;                      
                    }

                    if(z < stackString.Length && stackString[z] != "")
                    {
                        stackHeight++;
                        //Update the current height of the stack i
                        curStackHeight[i] = stackHeight;
                        //Add the container to the stack i
                        stack[stackHeight] = Convert.ToInt32(stackString[z]);
                    }
                }
                bayArray[i] = stack;
            }
            //Create a new bay and return it
            var bay = new Bay(width, height, containerCount, bayArray, curStackHeight);
            return bay;
        }
    }
}
