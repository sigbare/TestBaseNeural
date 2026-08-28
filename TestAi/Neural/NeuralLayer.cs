using System.Numerics;
using System.Runtime.CompilerServices;
using TestAi.Core.Abstractions;
using TestAi.Core.Models.Common;

namespace TestAi.Neural;

public sealed class NeuralLayer : INeuralLayer
{
    private readonly int _inputSize;
    private readonly int _outputSize;
    private readonly ActivationType _activationType;

    private readonly double[] _weights;
    private readonly double[] _biases;

    private double[]? _lastInputs;
    private double[]? _lastPreActivations;

    public NeuralLayer(
        int inputSize,
        int outputSize,
        ActivationType activationType,
        Random random)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputSize);
        ArgumentNullException.ThrowIfNull(random);

        _inputSize = inputSize;
        _outputSize = outputSize;
        _activationType = activationType;

        _weights = new double[inputSize * outputSize];
        _biases = new double[outputSize];

        var limit = Math.Sqrt(
            6.0 / (inputSize + outputSize));

        for (var i = 0; i < _weights.Length; i++)
        {
            _weights[i] =
                (random.NextDouble() * 2.0 - 1.0) * limit;
        }

        Array.Clear(_biases);
    }

    public int GetLayerSize()
        => _outputSize;

    public double[] Forward(double[] inputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);

        if (inputs.Length != _inputSize)
        {
            throw new ArgumentException(
                $"Input length must be {_inputSize}, " +
                $"but was {inputs.Length}.",
                nameof(inputs));
        }



        _lastInputs ??= new double[_inputSize];
        _lastPreActivations ??= new double[_outputSize];

        inputs.CopyTo(_lastInputs);

        var outputs = new double[_outputSize];

        if (_outputSize < 32)
        {
            ForwardSequential(inputs, outputs);
        }
        else
        {
            ForwardParallel(inputs, outputs);
        }

        return outputs;
    }

    public double[] Backward(
    double[] outputGradients,
    double learningRate)
    {
        ArgumentNullException.ThrowIfNull(outputGradients);

        if (_lastInputs is null ||
            _lastPreActivations is null)
        {
            throw new InvalidOperationException(
                "Forward must be called before Backward.");
        }

        if (outputGradients.Length != _outputSize)
        {
            throw new ArgumentException(
                $"Gradient length must be {_outputSize}, " +
                $"but was {outputGradients.Length}.",
                nameof(outputGradients));
        }

        var inputs = _lastInputs;
        var weights = _weights;
        var biases = _biases;
        var preActivations = _lastPreActivations;

        var inputGradients = new double[_inputSize];

        var vectorSize = Vector<double>.Count;
        var simdInputLength = _inputSize - _inputSize % vectorSize;

        for (var neuron = 0; neuron < _outputSize; neuron++)
        {
            var activationGradient =
                ActivationDerivative(preActivations[neuron]);

            var localGradient =
                outputGradients[neuron] * activationGradient;

            var offset = neuron * _inputSize;

            var gradientVector =
                new Vector<double>(localGradient);

            var learningRateGradientVector =
                new Vector<double>(
                    learningRate * localGradient);

            var input = 0;

            for (; input < simdInputLength; input += vectorSize)
            {
                var weightIndex = offset + input;

                var weightsVector =
                    new Vector<double>(weights, weightIndex);

                var inputsVector =
                    new Vector<double>(inputs, input);

                var inputGradientsVector =
                    new Vector<double>(inputGradients, input);

                inputGradientsVector +=
                    gradientVector * weightsVector;

                inputGradientsVector.CopyTo(
                    inputGradients,
                    input);
                weightsVector -=
                    learningRateGradientVector * inputsVector;

                weightsVector.CopyTo(
                    weights,
                    weightIndex);
            }

            for (; input < _inputSize; input++)
            {
                var weightIndex = offset + input;
                var oldWeight = weights[weightIndex];

                inputGradients[input] +=
                    localGradient * oldWeight;

                weights[weightIndex] =
                    oldWeight -
                    learningRate * localGradient * inputs[input];
            }

            biases[neuron] -= localGradient;
        }

        return inputGradients;
    }

    public void Save(BinaryWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.Write(_inputSize);
        writer.Write(_outputSize);

        writer.Write(_weights.Length);

        foreach (var weight in _weights)
            writer.Write(weight);

        writer.Write(_biases.Length);

        foreach (var bias in _biases)
            writer.Write(bias);
    }

    public void Load(BinaryReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);

        var inputSize = reader.ReadInt32();
        var outputSize = reader.ReadInt32();

        if (inputSize != _inputSize ||
    outputSize != _outputSize)
        {
            throw new InvalidOperationException(
                "Layer dimensions in the file do not match the current architecture.");
        }

        var weightsLength = reader.ReadInt32();

        if (weightsLength != _weights.Length)
        {
            throw new InvalidOperationException(
                "The number of weights does not match.");
        }

        for (var i = 0; i < _weights.Length; i++)
            _weights[i] = reader.ReadDouble();

        var biasesLength = reader.ReadInt32();

        if (biasesLength != _biases.Length)
        {
            throw new InvalidOperationException(
                "The number of biases does not match.");
        }

        for (var i = 0; i < _biases.Length; i++)
            _biases[i] = reader.ReadDouble();
    }

    private void ForwardSequential(
    ReadOnlySpan<double> inputs,
    Span<double> outputs)
    {
        var weights = _weights;
        var biases = _biases;
        var preActivations = _lastPreActivations!;

        var vectorSize = Vector<double>.Count;

        var simdInputLength = _inputSize - _inputSize % vectorSize;

        for (var neuron = 0; neuron < _outputSize; neuron++)
        {
            var offset = neuron * _inputSize;

            var sum = biases[neuron];
            var input = 0;

            for (; input < simdInputLength; input += vectorSize)
            {
                var weightsVector =
                    new Vector<double>(weights, offset + input);

                var inputsVector =
                    new Vector<double>(inputs.Slice(input, vectorSize));

                sum += Vector.Dot(
                    weightsVector,
                    inputsVector);
            }

            for (; input < _inputSize; input++)
            {
                sum +=
                    weights[offset + input] *
                    inputs[input];
            }

            preActivations[neuron] = sum;
            outputs[neuron] = Activate(sum);
        }
    }

    private void ForwardParallel(
        ReadOnlySpan<double> inputs,
        double[] outputs)
    {
        var inputArray = inputs.ToArray();

        var weights = _weights;
        var biases = _biases;
        var preActivations = _lastPreActivations!;

        var vectorSize = Vector<double>.Count;
        var simdInputLength =
            _inputSize - _inputSize % vectorSize;


        Parallel.For(
            0,
            _outputSize,
            neuron =>
            {
                var offset = neuron * _inputSize;

                var sum = biases[neuron];
                var input = 0;


                for (; input < simdInputLength; input += vectorSize)
                {
                    var weightsVector =
                        new Vector<double>(
                            weights,
                            offset + input);

                    var inputsVector =
                        new Vector<double>(
                            inputArray,
                            input);

                    sum += Vector.Dot(
                        weightsVector,
                        inputsVector);
                }

                for (; input < _inputSize; input++)
                {
                    sum +=
                        weights[offset + input] *
                        inputArray[input];
                }

                preActivations[neuron] = sum;
                outputs[neuron] = Activate(sum);
            });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double Activate(double value)
    {
        return _activationType switch
        {
            ActivationType.LeakyRelu =>
                value >= 0.0
                    ? value
                    : value * 0.01,

            ActivationType.Relu =>
                value >= 0.0
                    ? value
                    : 0.0,

            ActivationType.Sigmoid =>
                Sigmoid(value),

            ActivationType.Tanh =>
                Math.Tanh(value),

            ActivationType.None =>
                value,

            _ =>
                throw new ArgumentOutOfRangeException(
                    nameof(_activationType),
                    _activationType,
                    "Unsupported activation type.")
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double ActivationDerivative(double value)
    {
        return _activationType switch
        {
            ActivationType.LeakyRelu =>
                value >= 0.0
                    ? 1.0
                    : 0.01,

            ActivationType.Relu =>
                value >= 0.0
                    ? 1.0
                    : 0.0,

            ActivationType.Sigmoid =>
                Sigmoid(value) * (1.0 - Sigmoid(value)),

            ActivationType.Tanh =>
                1.0 - Math.Pow(Math.Tanh(value), 2.0),

            ActivationType.None =>
                1.0,

            _ =>
                throw new ArgumentOutOfRangeException(
                    nameof(_activationType),
                    _activationType,
                    "Unsupported activation type.")
        };


    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double Sigmoid(double value)
    {
        return value >= 0.0
            ? 1.0 / (1.0 + Math.Exp(-value))
            : Math.Exp(value) / (1.0 + Math.Exp(value));
    }
}
