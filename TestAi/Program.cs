using System.Diagnostics;
using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using TestAi;

const string trainFilePath =
    @"G:\UG\TestAi\TestAi\Data\train.csv";

const string testFilePath =
    @"G:\UG\TestAi\TestAi\Data\test.csv";

const string modelFilePath =
    @"G:\UG\TestAi\TestAi\Data\model.bin";

var config = new CsvConfiguration(
    CultureInfo.InvariantCulture)
{
    HasHeaderRecord = true,
    Delimiter = ",",
    HeaderValidated = null,
    MissingFieldFound = null
};

const double validationRate = 0.1d;

var images = LoadImages(
    trainFilePath,
    config,
    hasLabel: true);

var randomz = new Random(42);

images = images
    .OrderBy(_ => randomz.Next())
    .ToList();

var testImages = images.Slice(images.Count - (int)(images.Count * validationRate), (int)(images.Count * validationRate));

var trainImages = images.Slice(0, images.Count - (int)(images.Count * validationRate));

Console.WriteLine($"Train images: {trainImages.Count}");
Console.WriteLine($"Test images:  {testImages.Count}");
Console.WriteLine();

var network = new NeuralNetwork(
    inputSize: 784,
    hiddenSizes: [256, 128,64,32],
    classCount: 10);

const int epochs = 20;
const double learningRate = 0.001;

var watch = Stopwatch.StartNew();

for (var epoch = 1; epoch <= epochs; epoch++)
{
    var epochWatch = Stopwatch.StartNew();

    var random = Random.Shared;

    trainImages = trainImages
        .OrderBy(_ => random.Next())
        .ToList();

    foreach (var image in trainImages)
    {
        var input = ConvertInput(image);

        network.Train(
            input,
            image.Label,
            learningRate);
    }

    var trainResult = Evaluate(
        network,
        trainImages);

    var testResults = Evaluate(
        network,
        testImages);

    Console.WriteLine(
        $"Epoch {epoch}/{epochs} | " +
        $"Train loss: {trainResult.Loss:F4} | " +
        $"Train accuracy: {trainResult.Accuracy:P2} | " +
        $"Test loss: {testResults.Loss:F4} | " +
        $"Test accuracy: {testResults.Accuracy:P2} | " +
        $"Time: {epochWatch.Elapsed.TotalSeconds:F2} sec");
}

network.Save(modelFilePath);

Console.WriteLine();
Console.WriteLine($"Model saved: {modelFilePath}");
Console.WriteLine();


var testResult = Evaluate(
    network,
    testImages);

Console.WriteLine("Test results:");
Console.WriteLine($"Test loss:     {testResult.Loss:F4}");
Console.WriteLine($"Test accuracy: {testResult.Accuracy:P2}");
Console.WriteLine($"Total time:    {watch.Elapsed.TotalSeconds:F2} sec");

static List<MnistImage> LoadImages(
    string filePath,
    CsvConfiguration config,
    bool hasLabel)
{
    using var reader = new StreamReader(filePath);
    using var csv = new CsvReader(reader, config);

    csv.Context.RegisterClassMap(
        new MnistImageMap(hasLabel));

    return [.. csv.GetRecords<MnistImage>()];
}

static double[] ConvertInput(MnistImage image)
{
    var input = new double[784];

    for (var i = 0; i < input.Length; i++)
        input[i] = image.Pixels[i] / 255.0;

    return input;
}

static EvaluationResult Evaluate(
    NeuralNetwork network,
    List<MnistImage> images)
{
    var totalLoss = 0.0;
    var correctCount = 0;

    foreach (var image in images)
    {
        var input = ConvertInput(image);
        var probabilities = network.Predict(input);

        var probability = Math.Max(
            probabilities[image.Label],
            1e-15);

        totalLoss += -Math.Log(probability);

        var predictedClass = MathFunctions.ArgMax(probabilities);

        if (predictedClass == image.Label)
            correctCount++;
    }

    return new EvaluationResult(
        Loss: totalLoss / images.Count,
        Accuracy: (double)correctCount / images.Count);
}

readonly record struct EvaluationResult(
    double Loss,
    double Accuracy);
