using TestAi.Core.Models.Common;
using TestAi.Neural;

namespace TestAi;

public sealed class NeuralNetwork
{
    private readonly List<NeuralLayer> _hiddenLayers = [];
    private readonly NeuralLayer _outputLayer;

    private readonly int _classCount;

    public NeuralNetwork(
        int inputSize,
        IReadOnlyList<int> hiddenSizes,
        int classCount)
    {
        if (inputSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputSize));

        if (hiddenSizes is null || hiddenSizes.Count == 0)
            throw new ArgumentException(
                "must be init 1 layer.",
                nameof(hiddenSizes));

        if (hiddenSizes.Any(size => size <= 0))
            throw new ArgumentException(
                "must be positive size.",
                nameof(hiddenSizes));

        if (classCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(classCount));

        _classCount = classCount;

        var random = new Random();

        var previousSize = inputSize;

        foreach (var hiddenSize in hiddenSizes)
        {
            _hiddenLayers.Add(new NeuralLayer(
                previousSize,
                hiddenSize,
                ActivationType.LeakyRelu,
                random));

            previousSize = hiddenSize;
        }

        _outputLayer = new NeuralLayer(
            previousSize,
            classCount,
            ActivationType.None,
            random);
    }

    public double[] Predict(double[] input)
    {
        ValidateInput(input);

        var values = input;

        foreach (var layer in _hiddenLayers)
            values = layer.Forward(values);

        var logits = _outputLayer.Forward(values);

        return MathFunctions.Softmax(logits);
    }

    public int PredictClass(double[] input)
    {
        var probabilities = Predict(input);

        return MathFunctions.ArgMax(probabilities);
    }

    public double Train(
        double[] input,
        int correctClass,
        double learningRate)
    {
        ValidateInput(input);

        if (correctClass < 0 || correctClass >= _classCount)
            throw new ArgumentOutOfRangeException(nameof(correctClass));

        var layerOutputs = new double[_hiddenLayers.Count][];
        var values = input;

        for (var i = 0; i < _hiddenLayers.Count; i++)
        {
            values = _hiddenLayers[i].Forward(values);
            layerOutputs[i] = values;
        }

        var logits = _outputLayer.Forward(values);
        var probabilities = MathFunctions.Softmax(logits);

        var loss = MathFunctions.CrossEntropyLoss(
            probabilities,
            correctClass);


        var gradients = new double[_classCount];

        for (var i = 0; i < _classCount; i++)
            gradients[i] = probabilities[i];

        gradients[correctClass] -= 1.0;

        gradients = _outputLayer.Backward(
            gradients,
            learningRate);

        for (var i = _hiddenLayers.Count - 1; i >= 0; i--)
        {
            gradients = _hiddenLayers[i].Backward(
                gradients,
                learningRate);
        }

        return loss;
    }

    public void Save(string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        var directory = Path.GetDirectoryName(filePath);

        if (!string.IsNullOrWhiteSpace(directory))
            Directory.CreateDirectory(directory);

        using var stream = new FileStream(
            filePath,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None);

        using var writer = new BinaryWriter(stream);

        writer.Write(1);

        writer.Write(_hiddenLayers.Count);

        foreach (var layer in _hiddenLayers)
            layer.Save(writer);

        _outputLayer.Save(writer);

        writer.Flush();
    }

    public void Load(string filePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException(
                "Файл модели не найден.",
                filePath);
        }

        using var stream = new FileStream(
            filePath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read);

        using var reader = new BinaryReader(stream);

        var version = reader.ReadInt32();

        if (version != 1)
        {
            throw new InvalidOperationException(
                $"Неподдерживаемая версия модели: {version}.");
        }

        var hiddenLayerCount = reader.ReadInt32();

        if (hiddenLayerCount != _hiddenLayers.Count)
        {
            throw new InvalidOperationException(
                "Количество скрытых слоёв в файле " +
                "не совпадает с текущей архитектурой сети.");
        }

        foreach (var layer in _hiddenLayers)
            layer.Load(reader);


        _outputLayer.Load(reader);
    }


    private void ValidateInput(double[] input)
    {
        ArgumentNullException.ThrowIfNull(input);
    }
}
