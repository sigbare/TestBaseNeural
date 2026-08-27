namespace TestAi.Core.Abstractions;

public interface INeuralLayer
{
    public double[] Forward(double[] inputs);
    public double[] Backward(
        double[] outputGradients, double learningRate);
}