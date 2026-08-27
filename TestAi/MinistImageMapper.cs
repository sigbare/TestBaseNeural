using CsvHelper.Configuration;

namespace TestAi;

public sealed class MnistImageMap : ClassMap<MnistImage>
{
    public MnistImageMap(bool hasLabel)
    {
        if (hasLabel)
        {
            Map(m => m.Label).Name("label");
        }

        Map(m => m.Pixels).Convert(args =>
        {
            var pixels = new List<int>(784);

            for (var i = 0; i < 784; i++)
            {
                var value = args.Row.GetField($"pixel{i}");

                if (!int.TryParse(value, out var pixelValue))
                    pixelValue = 0;

                pixels.Add(pixelValue);
            }

            return pixels;
        });
    }
}