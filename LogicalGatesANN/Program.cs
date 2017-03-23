using ConstellationRecognitionANN.ArtificialNeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LogicalGatesANN
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork neuralNetwork = new NeuralNetwork(2, 1, 3, 1);
            neuralNetwork.backPropagate(50000, 0.000001, 0.9, "Input.txt", "Output.txt");

            Console.WriteLine("0 OR 0 = 0 | Interpolated: -1 OR -1 = -1");
            Console.WriteLine("0 OR 1 = 1 | Interpolated: -1 OR  1 =  1");
            Console.WriteLine("1 OR 0 = 1 | Interpolated:  1 OR -1 =  1");
            Console.WriteLine("1 OR 1 = 1 | Interpolated:  1 OR  1 =  1");
            

            Console.Write("\nEnter first interpolated input: ");
            double inputOne = Convert.ToInt32(Console.ReadLine());

            Console.Write("Enter second interpolated input: ");
            double inputTwo = Convert.ToInt32(Console.ReadLine());

            Console.WriteLine(neuralNetwork.forwardPass(new double[] { inputOne, inputTwo }));
            Console.ReadKey();
        }
    }
}
