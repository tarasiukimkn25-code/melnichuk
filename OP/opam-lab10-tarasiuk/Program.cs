using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace OpamLab10;

internal static class Program
{
    private const int N = 30;
    private const int K = 12;
    private static readonly string Root = AppContext.BaseDirectory;
    private static readonly string DataPath = Path.GetFullPath("data/zahar_berkut.txt");

    private static void Main()
    {
        Console.OutputEncoding = Encoding.UTF8;
        var text = File.ReadAllText(DataPath, Encoding.UTF8);
        Directory.CreateDirectory("results");
        var report = new Dictionary<string, object>();
        report["variant"] = new { N, K, a = N % 3, b = N % 4, c = N % 5 };
        report["string_traps"] = StringExperiments.Run(text);
        report["unique_substring"] = RunSubstringBenchmarks(text);
        report["search"] = RunSearchCases(text);
        report["knapsack"] = RunKnapsack();
        var json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText("results/measurements.json", json, Encoding.UTF8);
        Console.WriteLine(json);
    }

    private static object RunSubstringBenchmarks(string text)
    {
        var rows = new List<object>();
        foreach (var size in new[] { 1_000, 10_000, 50_000 })
        {
            var sample = text[..Math.Min(size, text.Length)];
            var slow = Measure(() => Algorithms.LongestUniqueNaive(sample));
            var fast = Measure(() => Algorithms.LongestUniqueWindow(sample));
            if (slow.result != fast.result) throw new InvalidOperationException("Substring algorithms disagree");
            rows.Add(new { chars = sample.Length, answer = fast.result, naive_ms = slow.ms, window_ms = fast.ms, ratio = fast.ms == 0 ? 0 : slow.ms / fast.ms });
        }
        return rows;
    }

    private static object RunSearchCases(string text)
    {
        const string word = "Беркут";
        const string missing = "НевигаданеСлово";
        var present = Algorithms.NaiveIndexOf(text, word);
        var absent = Algorithms.NaiveIndexOf(text, missing);
        const int repeatedCount = 10_000;
        var repeated = new string('І', repeatedCount);
        var pattern = new string('І', 1_019) + 'Ї'; // K=12 -> 13; final character is Ї.
        var late = Algorithms.NaiveIndexOf(repeated, pattern);
        return new
        {
            prose_present = new { word, present.index, present.comparisons, framework_index = text.IndexOf(word, StringComparison.Ordinal) },
            prose_missing = new { word = missing, absent.index, absent.comparisons, framework_index = text.IndexOf(missing, StringComparison.Ordinal) },
            repeated_near_match = new { text_chars = repeatedCount, pattern_chars = pattern.Length, late.index, late.comparisons,
                framework_index = repeated.IndexOf(pattern, StringComparison.Ordinal) }
        };
    }

    private static object RunKnapsack()
    {
        var random = new Random(N);
        var capacity = 10 + K;
        var tests = new List<(Knapsack.Item[] items, long greedy, long optimal)>();
        for (var t = 0; t < 100; t++)
        {
            var count = random.Next(5, 16);
            var items = Enumerable.Range(0, count).Select(i => new Knapsack.Item(i + 1, random.Next(1, 11), random.Next(1, 101))).ToArray();
            tests.Add((items, Knapsack.Greedy(items, capacity), Knapsack.Exhaustive(items, capacity)));
        }
        var counter = new[] { new Knapsack.Item(1, 6, 12), new Knapsack.Item(2, 5, 9), new Knapsack.Item(3, 5, 9) };
        var times = new List<object>();
        foreach (var count in new[] { 15, 20 })
        {
            var items = Enumerable.Range(0, count).Select(i => new Knapsack.Item(i + 1, random.Next(1, 11), random.Next(1, 101))).ToArray();
            Knapsack.Exhaustive(items, capacity); // warm-up
            var sw = Stopwatch.StartNew();
            var value = Knapsack.Exhaustive(items, capacity);
            sw.Stop();
            times.Add(new { count, value, elapsed_ms = sw.Elapsed.TotalMilliseconds });
        }
        return new
        {
            capacity_kg = capacity,
            random_seed = N,
            trials = tests.Count,
            greedy_optimal = tests.Count(x => x.greedy == x.optimal),
            greedy_below_optimal = tests.Count(x => x.greedy < x.optimal),
            mean_greedy_gap = tests.Average(x => (double)(x.optimal - x.greedy)),
            counterexample = new { capacity = 10, greedy = Knapsack.Greedy(counter, 10), exhaustive = Knapsack.Exhaustive(counter, 10), items = counter },
            exhaustive_times = times
        };
    }

    private static (T result, double ms) Measure<T>(Func<T> action)
    {
        action();
        var values = new List<double>();
        T result = default!;
        for (var i = 0; i < 3; i++) { var sw = Stopwatch.StartNew(); result = action(); sw.Stop(); values.Add(sw.Elapsed.TotalMilliseconds); }
        return (result, values.Min());
    }
}

internal static class Algorithms
{
    public static int LongestUniqueNaive(string s)
    {
        var best = 0;
        for (var start = 0; start < s.Length; start++)
        {
            var seen = new HashSet<char>();
            for (var end = start; end < s.Length; end++)
            {
                if (!seen.Add(s[end])) break;
                best = Math.Max(best, end - start + 1);
            }
        }
        return best;
    }

    public static int LongestUniqueWindow(string s)
    {
        var last = new Dictionary<char, int>(); var start = 0; var best = 0;
        for (var i = 0; i < s.Length; i++)
        {
            if (last.TryGetValue(s[i], out var previous) && previous >= start) start = previous + 1;
            last[s[i]] = i; best = Math.Max(best, i - start + 1);
        }
        return best;
    }

    public static (int index, long comparisons) NaiveIndexOf(string text, string pattern)
    {
        long comparisons = 0;
        for (var i = 0; i <= text.Length - pattern.Length; i++)
        {
            var j = 0;
            while (j < pattern.Length) { comparisons++; if (text[i + j] != pattern[j]) break; j++; }
            if (j == pattern.Length) return (i, comparisons);
        }
        return (-1, comparisons);
    }
}

internal static class StringExperiments
{
    public static object Run(string prose)
    {
        var rows = new List<object>();
        var sourceWord = prose.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            .First(word => word.Any(char.IsLetter));
        var upperWord = sourceWord.ToUpperInvariant();
        foreach (var n in new[] { 10_000, 100_000 })
        {
            var plus = Time(() => { var s = ""; for (var i = 0; i < n; i++) s += "x"; return s.Length; });
            var builder = Time(() => { var b = new StringBuilder(); for (var i = 0; i < n; i++) b.Append('x'); return b.Length; });
            var lower = Time(() => { var nmatch = 0; for (var i = 0; i < n; i++) if (sourceWord.ToLower() == upperWord.ToLower()) nmatch++; return nmatch; });
            var ordinal = Time(() => { var nmatch = 0; for (var i = 0; i < n; i++) if (string.Equals(sourceWord, upperWord, StringComparison.OrdinalIgnoreCase)) nmatch++; return nmatch; });
            rows.Add(new { operations = n, concat_ms = plus, string_builder_ms = builder, lowercase_compare_ms = lower, ordinal_ignore_case_ms = ordinal });
        }
        var words = new[] { "ґанок", "ґрунт", "гілка", "єдність", "їжак", "яблуко", "Ґудзик", "ґава", "знання", "Іван", "щастя", "човен", "юність", "файл", "хмара", "цукор", "світ", "річка", "пісня", "осінь" };
        return new { timing = rows, sorting = new { ordinal = words.Order(StringComparer.Ordinal).ToArray(), ukrainian_culture = words.Order(StringComparer.Create(CultureInfo.GetCultureInfo("uk-UA"), false)).ToArray() } };
    }
    private static double Time<T>(Func<T> action) { action(); var best = double.MaxValue; for (int i=0;i<3;i++){var sw=Stopwatch.StartNew();action();sw.Stop();best=Math.Min(best,sw.Elapsed.TotalMilliseconds);}return best; }
}

internal static class Knapsack
{
    public record Item(int Id, int Weight, int Value);
    public static long Greedy(Item[] items, int capacity)
    {
        long value = 0; var left = capacity;
        foreach (var item in items.OrderByDescending(x => (double)x.Value / x.Weight)) if (item.Weight <= left) { left -= item.Weight; value += item.Value; }
        return value;
    }
    public static long Exhaustive(Item[] items, int capacity)
    {
        long best = 0;
        void Visit(int i, int weight, long value) { if (weight > capacity) return; if (i == items.Length) { best = Math.Max(best, value); return; } Visit(i + 1, weight, value); Visit(i + 1, weight + items[i].Weight, value + items[i].Value); }
        Visit(0, 0, 0); return best;
    }
}
