using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConstellationRecognitionANN.ArtificialNeuralNetwork
{
    public class NeuralNetwork
    {

        private Layer inputLayer;
        private Layer hiddenLayer;
        private Layer outputLayer;

        /*
         * Neural Network Constructor
         * Parameters - Number of Input Neurons, Number of HiddenLayer
         *              Number of Hidden Layer Neurons, Number of Output Neurons
         */ 
        public NeuralNetwork(int numInputs, int numHidLayer
            , int numHidNeurons, int numOutNeur)
        {
            inputLayer = new Layer();
            hiddenLayer = new Layer();
            outputLayer = new Layer();

            Random rand = new Random();

            for (int i = 0; i < numInputs; i++)
            {
                inputLayer.Add(new InputNeuron(rand.NextDouble()));
            }

            List<double> randomHiddenWeights = new List<double>();
            for (int i = 0; i < numInputs; i++)
            {
                randomHiddenWeights.Add(rand.NextDouble());
            }

            for (int i = 0; i < numHidLayer; i++)
            {
                for (int j = 0; j < numHidNeurons; j++)
                {
                    hiddenLayer.Add(new HiddenNeuron(randomHiddenWeights, rand.NextDouble()));
                }
            }

            List<double> randomOutputWeights = new List<double>();
            for (int i = 0; i < numHidNeurons; i++)
            {
                randomOutputWeights.Add(rand.NextDouble());
            }

            for (int i = 0; i < numOutNeur; i++)
            {
                outputLayer.Add(new OutputNeuron(randomOutputWeights, 0.0));
            }
        }

        /*
         * Computes hidden layer neurons value.
         */ 
        public void computeHiddenLayerNeuronValue()
        {
            int index = 0;
            foreach (HiddenNeuron neuron in hiddenLayer)
            {
                foreach (double weight in neuron.Weights)
                {
                    InputNeuron inNeuron = (InputNeuron)inputLayer[index];
                    neuron.Soma += inNeuron.Input * weight;
                    index++;
                }
                neuron.Soma += neuron.Bias;
                neuron.Value = tansigAF(neuron.Soma);
                index = 0;
            }
        }

        /*
         * Computes output neurons value.
         */ 
        public double computeOutputNeuronValue()
        {
            int index = 0;
            double retVal = 0.0;
            foreach (OutputNeuron neuron in outputLayer)
            {
                foreach (double weight in neuron.Weights)
                {
                    HiddenNeuron hidNeuron = (HiddenNeuron)hiddenLayer[index];
                    neuron.Soma += weight * hidNeuron.Value;
                }
                neuron.Soma += neuron.Bias;
                neuron.Value = tansigAF(neuron.Soma);
                retVal = neuron.Value;
            }
            return retVal;
        }

        /*
         * Forward pass
         * Parameter - data for input
         */ 
        public double forwardPass(double[] data)
        {
            for (int j = 0; j < inputLayer.Count; j++)
            {
                InputNeuron inNeuron = (InputNeuron)inputLayer[j];
                inNeuron.Input = data[j];
                inputLayer[j] = inNeuron;
            }
            for (int j = 0; j < hiddenLayer.Count; j++)
            {
                HiddenNeuron hidNeuron = (HiddenNeuron)hiddenLayer[j];
                hidNeuron.Soma = 0.0;
                hiddenLayer[j] = hidNeuron;
            }
            for (int j = 0; j < outputLayer.Count; j++)
            {
                OutputNeuron outNeuron = (OutputNeuron)outputLayer[j];
                outNeuron.Soma = 0.0;
                outputLayer[j] = outNeuron;
            }
            return forwardPass();
        }

        /*
         * Forward pass
         */ 
        public double forwardPass()
        {
            computeHiddenLayerNeuronValue();
            return computeOutputNeuronValue();
        }

        /*
         * Tansig activation function
         */ 
        public double tansigAF(double soma)
        {
            return (Math.Exp(soma) - Math.Exp(-soma)) / (Math.Exp(soma) + Math.Exp(-soma));
        }

        public double interpolate(double input)
        {
            return 2 - ((input - 1) / (1 - 0)) - 1;
        }

        /*
         * Backpropagation
         * Parameters - Number of epochs, Learning rate, Target Error
         *              Input File Path, Output File Path 
         *    
         */ 
        public void backPropagate(int epoch, double learningRate, double targetError
            , string inputFilePath, string outputFilePath)
        {
            List<List<double>> trainData = new List<List<double>>();
            string[] inputContent = File.ReadAllLines(inputFilePath);

            List<double> outData = new List<double>();
            string[] outputContent = File.ReadAllLines(outputFilePath);

            int dataIndex = 0;

            foreach (string line in inputContent)
            {
                trainData.Add(new List<double>());
                string[] data = line.Split(',');
                foreach (string d in data)
                {
                    trainData[dataIndex].Add(Convert.ToDouble(d));
                }
                dataIndex++;
            }
            
            foreach (string line in outputContent)
            {
                outData.Add(Convert.ToDouble(line));
            }

            List<List<double>> hiddenNewWeights = new List<List<double>>();
            for (int i = 0; i < hiddenLayer.Count; i++)
            {
                hiddenNewWeights.Add(new List<double>());
                for (int j = 0; j < inputLayer.Count; j++)
                {
                    hiddenNewWeights[i].Add(0.5);
                }
            }

            List<double> outputNewWeights = new List<double>();
            for (int i = 0; i < hiddenLayer.Count; i++)
            {
                outputNewWeights.Add(0.5);
            }

            List<double> outputNewBias = new List<double>();
            for (int i = 0; i < outputLayer.Count; i++)
            {
                outputNewBias.Add(0.5);
            }

            List<double> hiddenNewBias = new List<double>();
            for (int i = 0; i < hiddenLayer.Count; i++)
            {
                hiddenNewBias.Add(0.5);
            }

            double[] backPropInputs = backPropInputs = new double[trainData[0].Count];
            
            int trainIndex = 0;

            double error = 1.0;

            for (int i = 0; i < epoch; i++)
            {

                if (error <= targetError)
                    break;

                List<double> outputOldWeights = outputNewWeights;
                List<List<double>> hiddenOldWeights = hiddenNewWeights;
                List<double> outputOldBias = outputNewBias;
                List<double> hiddenOldBias = hiddenNewBias;

                // Update Output Layer
                for (int j = 0; j < outputLayer.Count; j++)
                {
                    OutputNeuron outNeuron = (OutputNeuron)outputLayer[j];
                    for (int k = 0; k < outNeuron.Weights.Count; k++)
                    {
                        outNeuron.Weights[k] = outputOldWeights[k];
                    }
                    outNeuron.Bias = outputOldBias[j];
                    outNeuron.Soma = 0.0;
                    outputLayer[j] = outNeuron;
                }

                // Update Hidden Layer
                for (int j = 0; j < hiddenLayer.Count; j++)
                {
                    HiddenNeuron hidNeuron = (HiddenNeuron)hiddenLayer[j];
                    for (int k = 0; k < hidNeuron.Weights.Count; k++)
                    {
                        hidNeuron.Weights[k] = hiddenOldWeights[j][k];
                    }
                    hidNeuron.Bias = hiddenOldBias[j];
                    hidNeuron.Soma = 0.0;
                    hiddenLayer[j] = hidNeuron;
                }

                for (int j = 0; j < backPropInputs.Length; j++)
                {
                    backPropInputs[j] = trainData[trainIndex][j];
                }

                for (int j = 0; j < inputLayer.Count; j++)
                {
                    InputNeuron inNeuron = (InputNeuron)inputLayer[j];
                    inNeuron.Input = backPropInputs[j];
                    inputLayer[j] = inNeuron;
                }

                double backPropOut = outData[trainIndex];

                double output = forwardPass();

                error = 0.5 * (Math.Pow(backPropOut - output, 2));

                double dE_dOut = output - backPropOut;
                double dOut_dNetO = 1 - (Math.Pow(output, 2));

                // Hidden to Output Weights
                int index = 0;
                foreach (HiddenNeuron neuron in hiddenLayer)
                {
                    double dNetO_dW = neuron.Value;
                    double dE_dW = dE_dOut * dOut_dNetO * dNetO_dW;
                    outputNewWeights[index] = outputOldWeights[index] - (learningRate * dE_dW);
                    index++;
                }

                // Output Bias
                index = 0;
                foreach (OutputNeuron outNeuron in outputLayer)
                {
                    double dNetO_dB = 1.0;
                    double dE_dB = dE_dOut * dOut_dNetO * dNetO_dB;
                    outputNewBias[index] = outputOldBias[index] - (learningRate * dE_dB);
                }

                List<double> dNetO_dOutH = new List<double>();

                // Hidden Neuron
                index = 0;
                foreach (HiddenNeuron neuron in hiddenLayer)
                {
                    OutputNeuron outNeuron = (OutputNeuron)outputLayer[0];
                    dNetO_dOutH.Add(outNeuron.Weights[index]);
                    index++;
                }

                List<List<double>> dOutH_dNetOH = new List<List<double>>();

                // Hidden Neuron Net Value 
                index = 0;
                foreach (HiddenNeuron neuron in hiddenLayer)
                {
                    dOutH_dNetOH.Add(new List<double>());
                    dOutH_dNetOH[index].Add(1 - (Math.Pow(neuron.Value, 2)));
                    index++;
                }

                List<List<double>> dNetOH_dW = new List<List<double>>();

                // Net OH Weight
                index = 0;
                int backPropInputIndex = 0;
                foreach (HiddenNeuron hidNeuron in hiddenLayer)
                {
                    dNetOH_dW.Add(new List<double>());
                    foreach (InputNeuron inNeuron in inputLayer)
                    {
                        dNetOH_dW[index].Add(backPropInputs[backPropInputIndex]);
                        backPropInputIndex++;
                    }
                    backPropInputIndex = 0;
                    index++;
                }

                List<List<double>> dE_dWI = new List<List<double>>();

                // Input to Hidden Weight
                index = 0;
                int dNetOH_dW_Index = 0;
                foreach (HiddenNeuron hidNeuron in hiddenLayer)
                {
                    dE_dWI.Add(new List<double>());
                    foreach (InputNeuron inNeuron in inputLayer)
                    {
                        dE_dWI[index].Add(dE_dOut * dOut_dNetO * dNetO_dOutH[index] * dOutH_dNetOH[index][0]
                            * dNetOH_dW[index][dNetOH_dW_Index]);
                        dNetOH_dW_Index++;
                    }
                    dNetOH_dW_Index = 0;
                    index++;
                }

                for (int j = 0; j < hiddenNewWeights.Count; j++)
                {
                    for (int k = 0; k < hiddenNewWeights[j].Count; k++)
                    {
                        hiddenNewWeights[j][k] = hiddenOldWeights[j][k]
                            - (learningRate * dE_dWI[j][k]);
                    }
                }

                List<double> dNetOH_dB = new List<double>();
                for (int j = 0; j < hiddenLayer.Count; j++)
                {
                    dNetOH_dB.Add(1.0);
                }

                List<double> dE_dBI = new List<double>();
                for (int j = 0; j < hiddenLayer.Count; j++)
                {
                    dE_dBI.Add(dE_dOut * dOut_dNetO * dNetO_dOutH[j] *
                        dOutH_dNetOH[j][0] * dNetOH_dB[j]);
                }

                for (int j = 0; j < hiddenOldBias.Count; j++)
                {
                    hiddenNewBias[j] = hiddenOldBias[j] - (learningRate * dE_dBI[j]);
                }

                trainIndex++;

                if (trainIndex > trainData.Count - 1)
                    trainIndex = 0;
            }
        }

        public override string ToString()
        {
            string retVal = "";
            int inputNeuronCount = 1;
            foreach (InputNeuron neuron in inputLayer)
            {
                retVal += "Input Neuron: " + inputNeuronCount + "\r\n";
                retVal += "Value: 0\r\n";
                inputNeuronCount++;
            }
            int hiddenNeuronCount = 1, hiddenWeightCount = 1;
            foreach (HiddenNeuron neuron in hiddenLayer)
            {
                retVal += "Hidden Neuron: " + hiddenNeuronCount + "\r\n";
                retVal += "Bias: " + neuron.Bias + "\r\n";
                foreach (double weight in neuron.Weights)
                {
                    retVal += "Weight: " + hiddenWeightCount + "\r\n";
                    retVal += "Value: " + weight + "\r\n";
                    hiddenWeightCount++;
                }
                hiddenWeightCount = 1;
                hiddenNeuronCount++;
            }
            int outputNeuronCount = 1, outWeightCount = 1;
            foreach (OutputNeuron neuron in outputLayer)
            {
                retVal += "Output Neuron: " + outputNeuronCount + "\r\n";
                retVal += "Bias: " + neuron.Bias + "\r\n";
                foreach (double weight in neuron.Weights)
                {
                    retVal += "Weight: " + outWeightCount + "\r\n";
                    retVal += "Value: " + weight + "\r\n";
                    outWeightCount++;
                }
                outWeightCount = 1;
                outputNeuronCount++;
            }
            return retVal;
        }

    }
}
