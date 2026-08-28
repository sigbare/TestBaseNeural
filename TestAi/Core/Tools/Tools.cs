using System.Numerics;
using System.Runtime.CompilerServices;

namespace TestAi.Core.Tools;

public static class MathFunctions
{
    public static double[] Softmax(ReadOnlySpan<double> values)
    {
        if (values.IsEmpty)
            return [];

        var result = new double[values.Length];

        Softmax(values, result);

        return result;
    }

    public static void Softmax(
        ReadOnlySpan<double> values,
        Span<double> result)
    {
        if (values.Length != result.Length)
        {
            throw new ArgumentException(
                "Input and output spans must have the same length.");
        }

        if (values.IsEmpty)
            return;

        var max = FindMax(values);
        var sum = 0.0;

        for (var i = 0; i < values.Length; i++)
        {
            var exp = Math.Exp(values[i] - max);

            result[i] = exp;
            sum += exp;
        }

        var inverseSum = 1.0 / sum;

        Multiply(result, inverseSum);
    }

    public static double CrossEntropyLoss(
        ReadOnlySpan<double> logits,
        int correctClass)
    {
        if ((uint)correctClass >= (uint)logits.Length)
            throw new ArgumentOutOfRangeException(nameof(correctClass));

        if (logits.IsEmpty)
            throw new ArgumentException(
                "Logits cannot be empty.",
                nameof(logits));

        var max = FindMax(logits);
        var expSum = 0.0;

        for (var i = 0; i < logits.Length; i++)
        {
            expSum += Math.Exp(logits[i] - max);
        }

        // log(sum(exp(logits))) - logits[correctClass]
        return Math.Log(expSum) + max - logits[correctClass];
    }


    public static int ArgMax(ReadOnlySpan<double> values)
    {
        if (values.IsEmpty)
            throw new ArgumentException(
                "Values cannot be empty.",
                nameof(values));

        var vectorSize = Vector<double>.Count;
        var index = 0;

        var max = values[0];


        if (values.Length < vectorSize * 2)
        {
            for (var z = 1; z < values.Length; z++)
            {
                if (!(values[z] > max)) continue;
                max = values[z];
                index = z;
            }

            return index;
        }

        var maxVector = new Vector<double>(values[0]);
        var indexVector = new Vector<double>(0.0);

        var i = 0;

        for (; i <= values.Length - vectorSize; i += vectorSize)
        {
            var current = new Vector<double>(
                values.Slice(i, vectorSize));

            var mask = Vector.GreaterThan(current, maxVector);

            maxVector = Vector.ConditionalSelect(
                mask,
                current,
                maxVector);

            var indices = CreateIndices(i);
            indexVector = Vector.ConditionalSelect(
                mask,
                indices,
                indexVector);
        }

        max = maxVector[0];
        index = (int)indexVector[0];

        for (var lane = 1; lane < vectorSize; lane++)
        {
            if (!(maxVector[lane] > max)) continue;
            max = maxVector[lane];
            index = (int)indexVector[lane];
        }

        for (; i < values.Length; i++)
        {
            if (!(values[i] > max)) continue;
            max = values[i];
            index = i;
        }

        return index;
    }

    private static double FindMax(ReadOnlySpan<double> values)
    {
        var vectorSize = Vector<double>.Count;
        var i = 0;

        if (values.Length < vectorSize * 2)
        {
            var scalarMax = values[0];

            for (i = 1; i < values.Length; i++)
            {
                if (values[i] > scalarMax)
                    scalarMax = values[i];
            }

            return scalarMax;
        }

        var maxVector = new Vector<double>(values[0]);

        for (; i <= values.Length - vectorSize; i += vectorSize)
        {
            var current = new Vector<double>(
                values.Slice(i, vectorSize));

            maxVector = Vector.Max(maxVector, current);
        }

        var max = maxVector[0];

        for (var lane = 1; lane < vectorSize; lane++)
        {
            if (maxVector[lane] > max)
                max = maxVector[lane];
        }

        for (; i < values.Length; i++)
        {
            if (values[i] > max)
                max = values[i];
        }

        return max;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Multiply(
        Span<double> values,
        double multiplier)
    {
        var vectorSize = Vector<double>.Count;
        var i = 0;

        var multiplierVector = new Vector<double>(multiplier);

        for (; i <= values.Length - vectorSize; i += vectorSize)
        {
            var current = new Vector<double>(
                values.Slice(i, vectorSize));

            (current * multiplierVector)
                .CopyTo(values.Slice(i, vectorSize));
        }

        for (; i < values.Length; i++)
        {
            values[i] *= multiplier;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<double> CreateIndices(int start)
    {
        var result = new double[Vector<double>.Count];

        for (var i = 0; i < result.Length; i++)
            result[i] = start + i;

        return new Vector<double>(result);
    }
}