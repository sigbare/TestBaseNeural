```

BenchmarkDotNet v0.15.8, Windows 10 (10.0.19045.6456/22H2/2022Update)
AMD Athlon 3000G with Radeon Vega Graphics 3.50GHz, 1 CPU, 4 logical and 2 physical cores
.NET SDK 10.0.111
  [Host]     : .NET 10.0.11 (10.0.11, 10.0.1126.37416), X64 RyuJIT x86-64-v3
  Job-YLPXKJ : .NET 10.0.11 (10.0.11, 10.0.1126.37416), X64 RyuJIT x86-64-v3

IterationCount=3  WarmupCount=3  

```
| Method  | Size  | Mean        | Error       | StdDev    | Gen0    | Allocated |
|-------- |------ |------------:|------------:|----------:|--------:|----------:|
| SoftMax | 10    |    120.6 ns |    229.4 ns |  12.57 ns |  0.0497 |     104 B |
| SoftMax | 100   |    942.1 ns |    291.8 ns |  16.00 ns |  0.3929 |     824 B |
| SoftMax | 1000  |  9,098.6 ns |  2,002.1 ns | 109.74 ns |  3.8300 |    8024 B |
| SoftMax | 10000 | 91,031.8 ns | 13,770.6 ns | 754.81 ns | 36.9873 |   80024 B |
