using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using TestAi.Core.Tools;

namespace TestAi
{
    public static class TestAi
    {

        public static void StartTests()
        {
            var baseDirectory = AppDomain.CurrentDomain.BaseDirectory;

            var testFilePath = Path.Combine(
                baseDirectory,
                "data",
                "test.csv");

            var modelFilePath = Path.Combine(
                baseDirectory,
                "def.bin");

            Console.Clear();

            Console.WriteLine("========================================");
            Console.WriteLine("              TEST MODE");
            Console.WriteLine("========================================");
            Console.WriteLine();

            var selectedTestPath = AskForFilePath(
                title: "Test dataset",
                currentPath: testFilePath,
                requiredExtension: ".csv");

            if (selectedTestPath is null)
                return;

            testFilePath = selectedTestPath;

            var selectedModelPath = AskForFilePath(
                title: "Neural network model",
                currentPath: modelFilePath,
                requiredExtension: ".bin");

            if (selectedModelPath is null)
                return;

            modelFilePath = selectedModelPath;

            Console.Clear();
            Console.WriteLine("========================================");
            Console.WriteLine("             LOADING FILES");
            Console.WriteLine("========================================");
            Console.WriteLine();
            Console.WriteLine($"Test dataset: {testFilePath}");
            Console.WriteLine($"Model file:   {modelFilePath}");
            Console.WriteLine();

            try
            {
                var config = new CsvConfiguration(
                    CultureInfo.InvariantCulture)
                {
                    HasHeaderRecord = true,
                    Delimiter = ",",
                    HeaderValidated = null,
                    MissingFieldFound = null
                };

                Console.WriteLine("Loading test dataset...");

                var images = LoadImages(
                    testFilePath,
                    config,
                    hasLabel: false);

                Console.WriteLine($"Loaded images: {images.Count}");
                Console.WriteLine();

                Console.WriteLine("Loading neural network model...");

                var network = LoadNetwork(modelFilePath);

                Console.WriteLine("Model loaded successfully.");
                Console.WriteLine();
                Console.WriteLine("Press any key to start testing...");
                Console.ReadKey(true);

                for (var imageIndex = 0; imageIndex < images.Count; imageIndex++)
                {
                    var image = images[imageIndex];

                    var input = ConvertInput(image);
                    var probabilities = network.Predict(input);

                    var predictedNumber = MathFunctions.ArgMax(probabilities);
                    var confidence = probabilities[predictedNumber];

                    DrawScreen(
                        image,
                        probabilities,
                        predictedNumber,
                        confidence,
                        imageIndex + 1,
                        images.Count);

                    Console.WriteLine();
                    Console.WriteLine("Press SPACE to show the next image.");
                    Console.WriteLine("Press Q or ESC to exit.");

                    while (true)
                    {
                        var key = Console.ReadKey(true).Key;

                        if (key == ConsoleKey.Q ||
                            key == ConsoleKey.Escape)
                        {
                            Console.Clear();
                            Console.WriteLine("Testing was cancelled.");
                            return;
                        }

                        if (key == ConsoleKey.Spacebar)
                            break;
                    }
                }

                Console.Clear();
                Console.WriteLine("========================================");
                Console.WriteLine("          TESTING COMPLETED");
                Console.WriteLine("========================================");
                Console.WriteLine();
                Console.WriteLine($"Processed images: {images.Count}");
                Console.WriteLine();
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey(true);
            }
            catch (Exception exception)
            {
                Console.Clear();

                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("========================================");
                Console.WriteLine("                ERROR");
                Console.WriteLine("========================================");
                Console.ResetColor();

                Console.WriteLine();
                Console.WriteLine(exception.Message);
                Console.WriteLine();
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey(true);
            }
        }

        private static string? AskForFilePath(
            string title,
            string currentPath,
            string requiredExtension)
        {
            while (true)
            {
                Console.Clear();

                Console.WriteLine("========================================");
                Console.WriteLine($"          {title.ToUpperInvariant()}");
                Console.WriteLine("========================================");
                Console.WriteLine();

                Console.WriteLine("Current path:");
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine(currentPath);
                Console.ResetColor();

                Console.WriteLine();
                Console.WriteLine("Choose an option:");
                Console.WriteLine("  [Y] Change path");
                Console.WriteLine("  [N] Use current path");
                Console.WriteLine("  [Q] Exit");
                Console.WriteLine();
                Console.Write("> ");

                var answer = Console.ReadLine()?.Trim();

                if (string.Equals(answer, "q", StringComparison.OrdinalIgnoreCase))
                    return null;

                if (string.Equals(answer, "n", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(answer, "no", StringComparison.OrdinalIgnoreCase))
                {
                    if (!File.Exists(currentPath))
                    {
                        ShowError(
                            $"The default file was not found:\n{currentPath}");

                        continue;
                    }

                    if (!HasRequiredExtension(currentPath, requiredExtension))
                    {
                        ShowError(
                            $"The file must have the {requiredExtension} extension.");

                        continue;
                    }

                    return Path.GetFullPath(currentPath);
                }

                if (string.Equals(answer, "y", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(answer, "yes", StringComparison.OrdinalIgnoreCase))
                {
                    return ReadCustomFilePath(title, requiredExtension);
                }

                ShowError("Invalid option. Please enter Y, N, or Q.");
            }
        }

        private static string? ReadCustomFilePath(
            string title,
            string requiredExtension)
        {
            while (true)
            {
                Console.Clear();

                Console.WriteLine("========================================");
                Console.WriteLine($"          CHANGE {title.ToUpperInvariant()}");
                Console.WriteLine("========================================");
                Console.WriteLine();

                Console.WriteLine($"Enter the full path to the file (*{requiredExtension}):");
                Console.WriteLine("Type 'back' to return to the previous menu.");
                Console.WriteLine();
                Console.Write("> ");

                var input = Console.ReadLine()?.Trim();

                if (string.Equals(input, "back", StringComparison.OrdinalIgnoreCase))
                    return null;

                if (string.IsNullOrWhiteSpace(input))
                {
                    ShowError("The path cannot be empty.");
                    continue;
                }

                if (!File.Exists(input))
                {
                    ShowError("The specified file does not exist.");
                    continue;
                }

                if (!HasRequiredExtension(input, requiredExtension))
                {
                    ShowError(
                        $"Invalid file extension. Expected: {requiredExtension}");

                    continue;
                }

                var fullPath = Path.GetFullPath(input);

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine();
                Console.WriteLine("Path successfully changed:");
                Console.WriteLine(fullPath);
                Console.ResetColor();

                Console.WriteLine();
                Console.WriteLine("Press any key to continue...");
                Console.ReadKey(true);

                return fullPath;
            }
        }

        private static bool HasRequiredExtension(
            string filePath,
            string requiredExtension)
        {
            return string.Equals(
                Path.GetExtension(filePath),
                requiredExtension,
                StringComparison.OrdinalIgnoreCase);
        }

        private static void ShowError(string message)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine();
            Console.WriteLine($"Error: {message}");
            Console.ResetColor();

            Console.WriteLine();
            Console.WriteLine("Press any key to try again...");
            Console.ReadKey(true);
        }

        private static void DrawImage(
        MnistImage image,
        int startColumn,
        int startRow)
        {
            const int imageSize = 28;

            const string shades = " ░▒▓▓███";

            for (var row = 0; row < imageSize; row++)
            {
                for (var column = 0; column < imageSize; column++)
                {
                    var index = row * imageSize + column;

                    var pixel = Convert.ToInt32(image.Pixels[index]);

                    pixel = Math.Clamp(pixel, 0, 255);

                    var shadeIndex =
                        pixel * (shades.Length - 1) / 255;

                    var symbol = shades[shadeIndex];

                    WriteAt(
                        startColumn + column * 2,
                        startRow + row,
                        $"{symbol}{symbol}");
                }
            }
        }

        private static void DrawProbabilities(
                double[] probabilities,
                int predictedNumber,
                double confidence,
                int startColumn,
                int startRow)
        {
            const int barWidth = 30;

            for (var number = 0; number < probabilities.Length; number++)
            {
                var probability = probabilities[number];
                var percent = probability * 100;

                var filledLength =
                    (int)Math.Round(probability * barWidth);

                filledLength = Math.Clamp(
                    filledLength,
                    0,
                    barWidth);

                var emptyLength = barWidth - filledLength;

                var bar =
                    new string('█', filledLength) +
                    new string('░', emptyLength);

                var result = number == predictedNumber
                    ? " ← Selected Number"
                    : string.Empty;

                Console.SetCursorPosition(
                    startColumn,
                    startRow + number);

                if (number == predictedNumber)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.Write(
                        $"{number}: {percent,6:F2}% {bar}{result}");
                    Console.ResetColor();
                }
                else
                {
                    Console.Write(
                        $"{number}: {percent,6:F2}% {bar}");
                }
            }

            WriteAt(
                startColumn,
                startRow + probabilities.Length + 2,
                $"Maximum probability: {confidence * 100:F2}%");
        }

        private static void WriteAt(
                int column,
                int row,
                string text)
        {
            if (column < 0 || row < 0)
                return;

            if (row >= Console.BufferHeight)
                return;

            Console.SetCursorPosition(column, row);

            var availableWidth = Console.BufferWidth - column;

            if (availableWidth <= 0)
                return;

            if (text.Length > availableWidth)
                text = text[..availableWidth];

            Console.Write(text);
        }

        private static void DrawScreen(
                MnistImage image,
                double[] probabilities,
                int predictedNumber,
                double confidence,
                int currentImage,
                int totalImages)
        {
            Console.Clear();

            Console.CursorVisible = false;

            const int imageWidth = 56;
            const int probabilityStartColumn = 62;

            WriteAt(0, 0, $"Image {currentImage} / {totalImages}");
            WriteAt(probabilityStartColumn, 0, "Recognition Probabilities");

            WriteAt(0, 1, new string('─', imageWidth));
            WriteAt(
                probabilityStartColumn,
                1,
                new string('─', 45));

            DrawImage(image, 0, 2);

            DrawProbabilities(
                probabilities,
                predictedNumber,
                confidence,
                probabilityStartColumn,
                2);

            WriteAt(0, 32, new string('─', 110));

            WriteAt(
                0,
                34,
                $"Result: {predictedNumber}");

            WriteAt(
                0,
                35,
                $"Confidence: {confidence * 100:F2}%");

            WriteAt(
                0,
                37,
                "[Space] Next image    [Q/Esc] Exit");
        }


        private static List<MnistImage> LoadImages(
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
        private static double[] ConvertInput(MnistImage image)
        {
            var input = new double[784];

            for (var i = 0; i < input.Length; i++)
                input[i] = Convert.ToDouble(image.Pixels[i]) / 255.0;

            return input;
        }


        private static NeuralNetwork LoadNetwork(string path)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(path);

            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    "Model file was not found.",
                    path);
            }

            int inputSize = 0;
            int classCount;
            int[] hiddenSizes;


            using (var stream = new FileStream(
                       path,
                       FileMode.Open,
                       FileAccess.Read,
                       FileShare.Read))
            using (var reader = new BinaryReader(stream))
            {
                var version = reader.ReadInt32();

                if (version != 1)
                {
                    throw new InvalidOperationException(
                        $"Unsupported model version: {version}.");
                }

                var hiddenLayerCount = reader.ReadInt32();

                if (hiddenLayerCount <= 0)
                {
                    throw new InvalidOperationException(
                        "The model must contain at least one hidden layer.");
                }

                hiddenSizes = new int[hiddenLayerCount];

                for (var i = 0; i < hiddenLayerCount; i++)
                {
                    var layerInputSize = reader.ReadInt32();
                    var layerOutputSize = reader.ReadInt32();

                    if (i == 0)
                    {
                        inputSize = layerInputSize;
                    }
                    else
                    {
                        var previousHiddenSize = hiddenSizes[i - 1];

                        if (layerInputSize != previousHiddenSize)
                        {
                            throw new InvalidOperationException(
                                "Hidden layer dimensions are not connected correctly.");
                        }
                    }

                    hiddenSizes[i] = layerOutputSize;

                    SkipLayerData(reader);
                }


                var outputInputSize = reader.ReadInt32();
                classCount = reader.ReadInt32();

                if (outputInputSize != hiddenSizes[^1])
                {
                    throw new InvalidOperationException(
                        "The output layer input size does not match the last hidden layer.");
                }

                SkipLayerData(reader);
            }


            var network = new NeuralNetwork(
                inputSize,
                hiddenSizes,
                classCount);

            network.Load(path);

            return network;
        }

        private static void SkipLayerData(BinaryReader reader)
        {
            var weightsLength = reader.ReadInt32();

            if (weightsLength < 0)
            {
                throw new InvalidOperationException(
                    "Invalid weights length in model file.");
            }

            for (var i = 0; i < weightsLength; i++)
                reader.ReadDouble();

            var biasesLength = reader.ReadInt32();

            if (biasesLength < 0)
            {
                throw new InvalidOperationException(
                    "Invalid biases length in model file.");
            }

            for (var i = 0; i < biasesLength; i++)
                reader.ReadDouble();
        }

    }
}