
using TestAi;
using TestAi.View;

var main = new Operations();
var programmFlage = true;

ConsoleMessages.WelcomeMessage();
ConsoleMessages.InterfaceMenu(false);


while (programmFlage)
{
    var choice = Console.ReadLine();

    if (!int.TryParse(choice, out var number))
    {
        ErrorChose();
        continue;
    }

    Action handle = number switch
    {
        1 => main.StartTrainNewModel,
        3 => TestAi.TestAi.StartTests,
        4 => main.SettingsProgramm,
        5 => ExitProgramm,
        _ => ErrorChose
    };

    handle();
    ConsoleMessages.InterfaceMenu(true);
}


void ExitProgramm()
{
    programmFlage = false;
}

static void ErrorChose()
    => Console.WriteLine("Pleas use number from 1 to 4");




