using System.Diagnostics;
using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using TestAi.Core.Models;
using TestAi.Core.Models.Common;
using TestAi.Core.Tools;
using TestAi.View;

namespace TestAi
{
    public class Operations
    {
        public string TrainFilePath { get; set; }
        public string ModelFilePath { get; set; }
        public string ModelName { get; set; }
        public double ValidationRate { get; set; } = 0.05d;

        private List<MnistImage>? _mnistImages = null;
        private readonly Random _random = new(42);
        private readonly CsvConfiguration _config = new(
            CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true,
            Delimiter = ",",
            HeaderValidated = null,
            MissingFieldFound = null
        };

        public Operations()
        {
            ModelName = "def";
            TrainFilePath = $@"{AppDomain.CurrentDomain.BaseDirectory}\data\train.csv";
            ModelFilePath = $@"{AppDomain.CurrentDomain.BaseDirectory}\{ModelName}.bin";
        }

        public void SettingsProgramm()
        {
            while (true)
            {
                Console.Clear();
                PrintSettingsHeader();

                Console.ForegroundColor = ConsoleColor.White;
                Console.WriteLine($"  1. Train File Path: {TrainFilePath}");
                Console.WriteLine($"  2. Model File Path: {ModelFilePath}");
                Console.WriteLine($"  3. Model Name: {ModelName}");
                Console.WriteLine($"  4. Validation Rate: {ValidationRate:P1}");
                Console.ResetColor();

                PrintSeparator();

                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine("  [q]  Back to main menu");
                Console.ResetColor();

                PrintSeparator();

                Console.Write("Select an option: ");
                var choice = Console.ReadLine()?.ToLower();

                if (choice == "q") break;

                switch (choice)
                {
                    case "1":
                        UpdateTrainFilePath();
                        break;
                    case "2":
                        UpdateModelFilePath();
                        break;
                    case "3":
                        UpdateModelName();
                        break;
                    case "4":
                        UpdateValidationRate();
                        break;
                    default:
                        PrintErrorMessage("Invalid option. Please choose 1-4 or 'q'.");
                        break;
                }
            }
        }

        public void StartTrainNewModel()
        {
            Console.Clear();
            PrintTrainingHeader();

            if (!ValidateTrainFilePath())
                return;

            Console.WriteLine("Loading training data...");
            _mnistImages ??= LoadImages(TrainFilePath, _config, hasLabel: true);

            if (_mnistImages == null || _mnistImages.Count == 0)
            {
                PrintErrorMessage("Failed to load images. Please check the training data path.");
                WaitForKey();
                return;
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✓ Loaded {_mnistImages.Count} images successfully.");
            Console.ResetColor();

            Console.WriteLine("Splitting dataset into training and validation sets...");

            var shuffledImages = _mnistImages.OrderBy(_ => _random.Next()).ToList();
            var validationCount = (int)(shuffledImages.Count * ValidationRate);
            var trainCount = shuffledImages.Count - validationCount;

            var trainImages = shuffledImages.Slice(0, trainCount);
            var validationImages = shuffledImages.Slice(trainCount, validationCount);

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"  Training images:   {trainImages.Count}");
            Console.WriteLine($"  Validation images: {validationImages.Count}");
            Console.ResetColor();
            Console.WriteLine();

            var neuralSettings = GetNeuralSettings();
            var activationType = SelectActivationFunction();

            var network = new NeuralNetwork(
                inputSize: 784,
                hiddenSizes: neuralSettings.HiddenSize,
                classCount: 10,
                activationType);

            Console.WriteLine("Starting training...");
            Console.WriteLine();

            TrainNetwork(network, neuralSettings, validationImages, trainImages);

            Console.WriteLine("Saving model...");
            network.Save(ModelFilePath);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✓ Model saved successfully!");
            Console.WriteLine($"  Location: {ModelFilePath}");
            Console.ResetColor();

            Console.WriteLine();
            WaitForKey();
        }

        #region Private Helper Methods

        private void UpdateTrainFilePath()
        {
            Console.WriteLine($"Current path: {TrainFilePath}");
            Console.Write("Enter new path (or press Enter to cancel): ");
            var newPath = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(newPath))
            {
                Console.WriteLine("Update cancelled.");
                return;
            }

            if (!File.Exists(newPath))
            {
                PrintErrorMessage("File not found at the specified location.");
                return;
            }

            TrainFilePath = newPath;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("✓ Train file path updated successfully.");
            Console.ResetColor();
        }

        private void UpdateModelFilePath()
        {
            Console.WriteLine($"Current path: {ModelFilePath}");
            Console.Write("Enter new model file path (e.g., G:\\Models\\model.bin): ");
            var newPath = Console.ReadLine();

            if (!string.IsNullOrWhiteSpace(newPath))
            {
                ModelFilePath = newPath;
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("✓ Model file path updated successfully.");
                Console.ResetColor();
            }
            else
            {
                Console.WriteLine("Update cancelled.");
            }
        }

        private void UpdateModelName()
        {
            Console.WriteLine($"Current name: {ModelName}");
            Console.Write("Enter new model name: ");
            var newName = Console.ReadLine();

            if (!string.IsNullOrWhiteSpace(newName))
            {
                ModelName = newName;
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("✓ Model name updated successfully.");
                Console.ResetColor();
            }
            else
            {
                Console.WriteLine("Update cancelled.");
            }
        }

        private void UpdateValidationRate()
        {
            while (true)
            {
                Console.WriteLine($"Current validation rate: {ValidationRate:P1}");
                Console.Write("Enter new validation rate (0.01-0.50) or press Enter to cancel: ");
                var input = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(input))
                {
                    Console.WriteLine("Update cancelled.");
                    break;
                }

                if (double.TryParse(input, NumberStyles.Any, CultureInfo.InvariantCulture, out double rate))
                {
                    if (rate > 0.0 && rate < 1.0)
                    {
                        ValidationRate = rate;
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.WriteLine($"✓ Validation rate set to {ValidationRate:P1}");
                        Console.ResetColor();
                        break;
                    }

                    PrintErrorMessage("Rate must be between 0 and 1 (e.g., 0.05 for 5%).");
                }
                else
                {
                    PrintErrorMessage("Invalid input. Please enter a decimal number.");
                }
            }
        }

        private bool ValidateTrainFilePath()
        {
            if (File.Exists(TrainFilePath))
                return true;

            PrintErrorMessage($"Train file not found at: {TrainFilePath}");

            while (true)
            {
                Console.WriteLine("Would you like to update the file path? (y/n)");
                var choice = Console.ReadLine()?.ToLower();

                if (choice == "n")
                    return false;

                if (choice == "y")
                {
                    Console.Write("Enter correct path: ");
                    var newPath = Console.ReadLine();

                    if (!string.IsNullOrWhiteSpace(newPath) && File.Exists(newPath))
                    {
                        TrainFilePath = newPath;
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.WriteLine("✓ Path updated successfully.");
                        Console.ResetColor();
                        return true;
                    }

                    PrintErrorMessage("File not found. Please try again.");
                }
                else
                {
                    PrintErrorMessage("Invalid input. Please enter 'y' or 'n'.");
                }
            }
        }

        private SettingsNeural GetNeuralSettings()
        {
            Console.WriteLine("Configure neural network settings:");
            PrintSeparator();

            Console.Write("Use default settings? (Epochs=5, LR=0.01, Hidden=[32]) (y/n): ");
            var choice = Console.ReadLine()?.ToLower();

            if (choice == "y")
            {
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine("Using default settings.");
                Console.ResetColor();
                return new SettingsNeural();
            }

            int epochs = ReadInteger("Enter number of epochs", 5);
            double lr = ReadDouble("Enter learning rate", 0.01);
            var hiddenSizes = ReadHiddenLayerSizes();

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"Settings applied: Epochs={epochs}, LR={lr:F4}, Layers=[{string.Join(", ", hiddenSizes)}]");
            Console.ResetColor();

            return new SettingsNeural(epochs, lr, hiddenSizes);
        }

        private int ReadInteger(string prompt, int defaultValue)
        {
            Console.Write($"{prompt} (default {defaultValue}): ");
            var input = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(input))
                return defaultValue;

            return int.TryParse(input, out int result) ? result : defaultValue;
        }

        private double ReadDouble(string prompt, double defaultValue)
        {
            Console.Write($"{prompt} (default {defaultValue:F2}): ");
            var input = Console.ReadLine();

            if (string.IsNullOrWhiteSpace(input))
                return defaultValue;

            return double.TryParse(input, NumberStyles.Any, CultureInfo.InvariantCulture, out double result)
                ? result
                : defaultValue;
        }

        private List<int> ReadHiddenLayerSizes()
        {
            while (true)
            {
                Console.Write("Enter hidden layer sizes separated by spaces (e.g., '128 64 32') or press Enter for [32]: ");
                var input = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(input))
                    return [32];

                try
                {
                    var sizes = input.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                                     .Select(int.Parse)
                                     .Where(x => x > 0)
                                     .ToList();

                    if (sizes.Count > 0)
                        return sizes;

                    PrintErrorMessage("Layer sizes must be positive integers.");
                }
                catch (FormatException)
                {
                    PrintErrorMessage("Invalid format. Use numbers separated by spaces.");
                }
            }
        }

        private ActivationType SelectActivationFunction()
        {
            Console.Clear();
            PrintLogo();
            PrintSeparator();

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("             SELECT ACTIVATION FUNCTION");
            Console.ResetColor();

            PrintSeparator();

            Console.WriteLine();
            Console.WriteLine("  1. ReLU");
            Console.WriteLine("     f(x) = max(0, x)");
            Console.WriteLine("     Returns 0 for negatives, x for positives.");
            Console.WriteLine("     Commonly used in hidden layers.");
            Console.WriteLine();

            Console.WriteLine("  2. Leaky ReLU");
            Console.WriteLine("     f(x) = x,              if x > 0");
            Console.WriteLine("     f(x) = alpha * x,      if x <= 0");
            Console.WriteLine("     Prevents dying ReLU problem. alpha = 0.01.");
            Console.WriteLine();

            Console.WriteLine("  3. None (Linear)");
            Console.WriteLine("     f(x) = x");
            Console.WriteLine("     No activation. Used for regression output.");
            Console.WriteLine();

            Console.WriteLine("  4. Sigmoid");
            Console.WriteLine("     f(x) = 1 / (1 + e^(-x))");
            Console.WriteLine("     Maps values to (0, 1). Used for binary classification.");
            Console.WriteLine();

            Console.WriteLine("  5. Tanh");
            Console.WriteLine("     f(x) = (e^x - e^(-x)) / (e^x + e^(-x))");
            Console.WriteLine("     Maps values to (-1, 1). Zero-centered output.");
            Console.WriteLine();

            PrintSeparator();

            while (true)
            {
                Console.Write("Select activation function (1-5): ");
                var input = Console.ReadLine();

                if (int.TryParse(input, out int choice))
                {
                    var result = choice switch
                    {
                        1 => ActivationType.Relu,
                        2 => ActivationType.LeakyRelu,
                        3 => ActivationType.None,
                        4 => ActivationType.Sigmoid,
                        5 => ActivationType.Tanh,
                        _ => (ActivationType?)null
                    };

                    if (result.HasValue)
                    {
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.WriteLine($"✓ Selected: {result}");
                        Console.ResetColor();
                        return result.Value;
                    }
                }

                PrintErrorMessage("Invalid selection. Please enter a number between 1 and 5.");
            }
        }

        private void TrainNetwork(
            NeuralNetwork network,
            SettingsNeural settings,
            List<MnistImage> validationImages,
            List<MnistImage> trainImages)
        {
            var totalEpochs = settings.Epoch;

            for (var epoch = 1; epoch <= totalEpochs; epoch++)
            {
                var epochWatch = Stopwatch.StartNew();

                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"Epoch {epoch}/{totalEpochs}");
                Console.ResetColor();

                for (var i = 0; i < trainImages.Count; i++)
                {
                    var image = trainImages[i];
                    var input = ConvertInput(image);

                    network.Train(input, image.Label, settings.LearningRate);

                    DrawProgressBar("Training", i + 1, trainImages.Count);
                }

                Console.WriteLine();

                var trainResult = Evaluate(network, trainImages);
                var validationResult = Evaluate(network, validationImages);

                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine(
                    $"Results | " +
                    $"Train Loss: {trainResult.Loss:F4} | " +
                    $"Train Acc: {trainResult.Accuracy:P2} | " +
                    $"Val Loss: {validationResult.Loss:F4} | " +
                    $"Val Acc: {validationResult.Accuracy:P2} | " +
                    $"Time: {epochWatch.Elapsed.TotalSeconds:F2}s"
                );
                Console.ResetColor();
                Console.WriteLine();
            }
        }

        private static List<MnistImage> LoadImages(
            string filePath,
            CsvConfiguration config,
            bool hasLabel)
        {
            using var reader = new StreamReader(filePath);
            using var csv = new CsvReader(reader, config);

            csv.Context.RegisterClassMap(new MnistImageMap(hasLabel));

            return [.. csv.GetRecords<MnistImage>()];
        }

        private static double[] ConvertInput(MnistImage image)
        {
            var input = new double[784];

            for (var i = 0; i < input.Length; i++)
                input[i] = image.Pixels[i] / 255.0;

            return input;
        }

        private static EvaluationResult Evaluate(
            NeuralNetwork network,
            List<MnistImage> images)
        {
            var totalLoss = 0.0;
            var correctCount = 0;

            foreach (var image in images)
            {
                var input = ConvertInput(image);
                var probabilities = network.Predict(input);

                var probability = Math.Max(probabilities[image.Label], 1e-15);
                totalLoss += -Math.Log(probability);

                if (MathFunctions.ArgMax(probabilities) == image.Label)
                    correctCount++;
            }

            return new EvaluationResult(
                Loss: totalLoss / images.Count,
                Accuracy: (double)correctCount / images.Count);
        }

        private static void DrawProgressBar(string label, int current, int total)
        {
            const int barWidth = 30;
            var progress = (double)current / total;
            var filledWidth = (int)(progress * barWidth);

            var bar = new string('█', filledWidth) + new string('░', barWidth - filledWidth);

            Console.Write($"\r{label}: [{bar}] {progress:P1} ({current}/{total})");
        }

        private static void WaitForKey()
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine("Press any key to continue...");
            Console.ResetColor();
            Console.ReadKey(true);
        }

        #endregion

        #region Console Styling (Matches ConsoleMessages Style)

        private static void PrintLogo()
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("""
                  __  __ _   _ ___ ____ _____   _    ___
                 |  \/  | \ | |_ _/ ___|_   _| / \  |_ _|
                 | |\/| |  \| || |\___ \ | |  / _ \  | |
                 | |  | | |\  || | ___) || | / ___ \ | |
                 |_|  |_|_| \_|___|____/ |_|/_/   \_\___|

                            M N I S T   A I
            """);
            Console.ResetColor();
        }

        private static void PrintSeparator()
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine(new string('─', 55));
            Console.ResetColor();
        }

        private static void PrintSettingsHeader()
        {
            Console.Clear();
            PrintLogo();
            PrintSeparator();

            Console.ForegroundColor = ConsoleColor.Blue;
            Console.WriteLine("                     SETTINGS");
            Console.ResetColor();

            PrintSeparator();
            Console.WriteLine();
        }

        private static void PrintTrainingHeader()
        {
            Console.Clear();
            PrintLogo();
            PrintSeparator();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("                 MODEL TRAINING");
            Console.ResetColor();

            PrintSeparator();
            Console.WriteLine();
        }

        private static void PrintErrorMessage(string message)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"✗ {message}");
            Console.ResetColor();
        }

        #endregion

        #region Nested Types

        private struct SettingsNeural
        {
            public int Epoch { get; set; } = 5;
            public double LearningRate { get; set; } = 0.01;
            public IReadOnlyList<int> HiddenSize { get; set; } = [32];

            public SettingsNeural(int epoch, double learningRate, IReadOnlyList<int> hiddenSize)
            {
                Epoch = epoch;
                LearningRate = learningRate;
                HiddenSize = hiddenSize;
            }

            public SettingsNeural() { }
        }

        private readonly record struct EvaluationResult(double Loss, double Accuracy);

        #endregion
    }
}