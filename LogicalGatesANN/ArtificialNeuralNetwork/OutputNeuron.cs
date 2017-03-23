using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConstellationRecognitionANN
{
    public class OutputNeuron
    {
        private List<double> weights;

        public List<double> Weights
        {
            get { return weights; }
            set { weights = value; }
        }

        private double bias;

        public double Bias
        {
            get { return bias; }
            set { bias = value; }
        }

        private double soma;

        public double Soma
        {
            get { return soma; }
            set { soma = value; }
        }

        private double value;

        public double Value
        {
            get { return this.value; }
            set { this.value = value; }
        }

        public OutputNeuron(List<double> weights, double bias)
        {
            this.weights = weights;
            this.bias = bias;
        }
    }
}
