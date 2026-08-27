using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using TestAi;

namespace Benchmarks;


[MemoryDiagnoser]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[SimpleJob(warmupCount: 3, iterationCount:3)]
public class ToolsBench
{
    [Params(10,100,1000,10000)]
    public int Size { get; set; }

    private double[] _data;

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        
        _data = new double[Size];

        for (var i = 0; i < Size; i++)
        {
            _data[i] = random.NextDouble() * 100;
        }
    }
    
    [BenchmarkCategory("SoftMax")]
    [Benchmark()]
    public void SoftMax()
    {
        var result = MathFunctions.Softmax(_data.AsSpan());
    }
    
}