namespace TestAi.View;

public static class ConsoleMessages
{
    private const string Logo = """
          __  __ _   _ ___ ____ _____   _    ___
         |  \/  | \ | |_ _/ ___|_   _| / \  |_ _|
         | |\/| |  \| || |\___ \ | |  / _ \  | |
         | |  | | |\  || | ___) || | / ___ \ | |
         |_|  |_|_| \_|___|____/ |_|/_/   \_\___|
    
                    M N I S T   A I
    """;

    public static void WelcomeMessage()
    {
        Console.Clear();

        PrintLogo();

        PrintSeparator();

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("Welcome to MNIST Image AI Trainer!");
        Console.ResetColor();

        Console.WriteLine();
        Console.WriteLine(
            """
            This program trains a fully connected neural network
            to recognize handwritten digits from the MNIST dataset.

            Each image has a size of 28 x 28 pixels and is converted
            into a vector containing 784 numerical values.

            The neural network processes these values through hidden
            layers and predicts one of ten digits: 0-9.

            During training, the network adjusts its weights using
            backpropagation and minimizes the classification error.
            """);

        Console.WriteLine();
        PrintSeparator();

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine("Press any key to continue...");
        Console.ResetColor();

        Console.ReadKey(true);
    }

    public static void InterfaceMenu(bool clearConsole = true)
    {
        if (clearConsole)
            Console.Clear();

        PrintLogo();
        PrintSeparator();

        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("                    MAIN MENU");
        Console.ResetColor();

        PrintSeparator();

        PrintMenuItem("1", "Start training a new model");
        PrintMenuItem("2", "Train the current model", ConsoleColor.DarkGray);
        PrintMenuItem("3", "Test the model");
        PrintMenuItem("4", "Settings");
        PrintMenuItem("5", "Exit");

        PrintSeparator();

        Console.Write("Select an option: ");
    }

    public static void PrintTrainingHeader()
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

    public static void PrintTestingHeader()
    {
        Console.Clear();

        PrintLogo();
        PrintSeparator();

        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine("                   MODEL TESTING");
        Console.ResetColor();

        PrintSeparator();
        Console.WriteLine();
    }

    public static void PrintSettingsHeader()
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

    public static void PrintExitMessage()
    {
        Console.Clear();

        PrintLogo();
        PrintSeparator();

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("Thank you for using MNIST Image AI Trainer!");
        Console.WriteLine("Goodbye!");
        Console.ResetColor();

        Console.WriteLine();
    }

    private static void PrintLogo()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine(Logo);
        Console.ResetColor();
    }

    private static void PrintSeparator()
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine(new string('─', fiftyFive()));
        Console.ResetColor();
    }

    private static void PrintMenuItem(
        string key,
        string description,
        ConsoleColor color = ConsoleColor.White)
    {
        Console.ForegroundColor = color;
        Console.WriteLine($"  [{key}]  {description}");
        Console.ResetColor();
    }

    private static int fiftyFive()
    {
        return 55;
    }
}
