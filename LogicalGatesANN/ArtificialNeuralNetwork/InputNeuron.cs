using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConstellationRecognitionANN
{
    public class InputNeuron
    {
        private double input;

        public double Input
        {
            get { return input; }
            set { input = value; }
        }

        public InputNeuron()
        {

        }

        public InputNeuron(double input)
        {
            this.input = input;
        }
    }
}
