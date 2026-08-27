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

        if (_outputSize < 64)
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

        var inputGradients = new double[_inputSize];

        for (var neuron = 0; neuron < _outputSize; neuron++)
        {
            var activationGradient =
                ActivationDerivative(
                    _lastPreActivations[neuron]);

            var localGradient =
                outputGradients[neuron] * activationGradient;

            var offset = neuron * _inputSize;

            for (var input = 0; input < _inputSize; input++)
            {
                var weightIndex = offset + input;


                var oldWeight = _weights[weightIndex];

                inputGradients[input] +=
                    localGradient * oldWeight;

                var weightGradient =
                    localGradient * _lastInputs[input];

                _weights[weightIndex] -=
                    learningRate * weightGradient;
            }

            _biases[neuron] -=
                learningRate * localGradient;
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
                "Размер слоя в файле не совпадает " +
                "с текущей архитектурой.");
        }

        var weightsLength = reader.ReadInt32();

        if (weightsLength != _weights.Length)
        {
            throw new InvalidOperationException(
                "Количество весов не совпадает.");
        }

        for (var i = 0; i < _weights.Length; i++)
            _weights[i] = reader.ReadDouble();

        var biasesLength = reader.ReadInt32();

        if (biasesLength != _biases.Length)
        {
            throw new InvalidOperationException(
                "Количество bias не совпадает.");
        }

        for (var i = 0; i < _biases.Length; i++)
            _biases[i] = reader.ReadDouble();
    }

    private void ForwardSequential(
        ReadOnlySpan<double> inputs,
        Span<double> outputs)
    {
        for (var neuron = 0; neuron < _outputSize; neuron++)
        {
            var offset = neuron * _inputSize;
            var sum = _biases[neuron];

            for (var input = 0; input < _inputSize; input++)
            {
                sum +=
                    _weights[offset + input] *
                    inputs[input];
            }

            _lastPreActivations![neuron] = sum;
            outputs[neuron] = Activate(sum);
        }
    }

    private void ForwardParallel(
        ReadOnlySpan<double> inputs,
        double[] outputs)
    {
        var inputArray = inputs.ToArray();

        Parallel.For(
            0,
            _outputSize,
            neuron =>
            {
                var offset = neuron * _inputSize;
                var sum = _biases[neuron];

                for (var input = 0; input < _inputSize; input++)
                {
                    sum +=
                        _weights[offset + input] *
                        inputArray[input];
                }

                _lastPreActivations![neuron] = sum;
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

            ActivationType.None => value,

            _ => value
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

            ActivationType.None => 1.0,

            _ => 1.0
        };
    }
}
