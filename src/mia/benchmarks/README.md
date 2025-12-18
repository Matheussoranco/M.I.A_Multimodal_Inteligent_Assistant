# AGI Benchmark Framework for M.I.A

This module provides a comprehensive framework for evaluating M.I.A against state-of-the-art AGI benchmarks.

## Supported Benchmarks

### 1. ARC-AGI (Abstract Reasoning Corpus)
Tests fluid intelligence and abstract pattern recognition.
- **Paper**: "On the Measure of Intelligence" (Chollet, 2019)
- **Focus**: Visual pattern transformation, analogy reasoning
- **Metrics**: Accuracy, transformation detection rate

### 2. GAIA (General AI Assistant)
Evaluates real-world assistant capabilities.
- **Paper**: "GAIA: A Benchmark for General AI Assistants" (2023)
- **Focus**: Web browsing, file handling, multi-step reasoning
- **Metrics**: Task completion rate, step efficiency

### 3. SWE-bench (Software Engineering)
Tests ability to resolve real GitHub issues.
- **Paper**: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (2023)
- **Focus**: Code understanding, bug localization, patch generation
- **Metrics**: Resolved rate, patch quality

### 4. GPQA (Graduate-level Science)
Graduate-level science questions requiring deep expertise.
- **Paper**: "GPQA: A Graduate-Level Google-Proof Q&A Benchmark" (2023)
- **Focus**: Physics, Chemistry, Biology reasoning
- **Metrics**: Accuracy per domain

### 5. WebVoyager
Tests autonomous web navigation capabilities.
- **Paper**: "WebVoyager: Building an End-to-End Web Agent" (2024)
- **Focus**: Browser automation, form filling, information extraction
- **Metrics**: Task success rate, action efficiency

### 6. OSWorld (Operating System)
Evaluates OS-level interaction and automation.
- **Paper**: "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks" (2024)
- **Focus**: Desktop automation, file management, app interaction
- **Metrics**: Task completion, action accuracy

### 7. MMMU (Multimodal Understanding)
Tests reasoning across multiple academic disciplines with images.
- **Paper**: "MMMU: A Massive Multi-discipline Multimodal Understanding Benchmark" (2024)
- **Focus**: 30 academic subjects, visual + text reasoning
- **Metrics**: Accuracy per discipline

## Installation

```bash
# Install extra dependencies
pip install -r requirements-extras.txt

# For specific benchmarks:
pip install datasets  # For GAIA, GPQA, MMMU
pip install swe-bench  # For SWE-bench
pip install gymnasium mss  # For OSWorld
```

## Usage

### Running a Single Benchmark

```python
from mia.benchmarks import ARCAGIBenchmark, AGIBenchmarkRunner

# Create agent wrapper
def agent_fn(task, context):
    # Your agent logic here
    return {"answer": "..."}

# Run single benchmark
benchmark = ARCAGIBenchmark()
runner = AGIBenchmarkRunner()
results = runner.run_single(benchmark, agent_fn, num_samples=100)

print(f"Accuracy: {results.metrics.accuracy:.2%}")
```

### Running All Benchmarks

```python
from mia.benchmarks import AGIBenchmarkRunner

runner = AGIBenchmarkRunner()

# Run all benchmarks
results = runner.run_all(agent_fn, num_samples=50)

# Generate report
report = runner.generate_suite_report(results, "benchmark_results")
```

### Command Line Interface

```bash
# Run specific benchmark
python -m mia.benchmarks.runner --benchmark arc-agi --samples 100

# Run category of benchmarks
python -m mia.benchmarks.runner --category reasoning --samples 50

# Run all benchmarks
python -m mia.benchmarks.runner --all --samples 25 --output results/
```

## Benchmark Categories

| Category | Benchmarks | Focus |
|----------|------------|-------|
| Reasoning | ARC-AGI, GPQA | Abstract/Scientific reasoning |
| Coding | SWE-bench | Software engineering |
| Agentic | GAIA, WebVoyager, OSWorld | Tool use, navigation |
| Multimodal | MMMU | Vision + language |

## Advanced Reasoning Module

The `mia.reasoning` module provides state-of-the-art reasoning strategies:

### Chain-of-Thought (CoT)
```python
from mia.reasoning import ChainOfThoughtReasoner

reasoner = ChainOfThoughtReasoner(llm)
trace = reasoner.reason("What is 15% of 340?")
print(trace.final_answer)
```

### Tree-of-Thought (ToT)
```python
from mia.reasoning import TreeOfThoughtReasoner

reasoner = TreeOfThoughtReasoner(llm, branch_factor=3, max_depth=4)
trace = reasoner.reason("Design a sorting algorithm for nearly-sorted arrays")
```

### Self-Consistency
```python
from mia.reasoning import SelfConsistencyReasoner

reasoner = SelfConsistencyReasoner(llm, num_samples=5)
trace = reasoner.reason("Is this statement true or false: ...")
print(f"Confidence: {trace.confidence:.2%}")
```

### ReAct (Reasoning + Acting)
```python
from mia.reasoning import ReActReasoner

tools = {
    "search": search_function,
    "calculate": calculate_function,
}

reasoner = ReActReasoner(llm, tools=tools)
trace = reasoner.reason("Find the population of Tokyo and convert to millions")
```

### Reflection
```python
from mia.reasoning import ReflectionReasoner

reasoner = ReflectionReasoner(llm, max_reflections=3)
trace = reasoner.reason("Write a regex to match email addresses")
```

## Meta-Cognitive Features

```python
from mia.reasoning.metacognition import MetaCognitiveLayer, SelfReflection

# Task analysis
meta = MetaCognitiveLayer(llm)
analysis = meta.analyze_task("Solve this differential equation")
print(f"Difficulty: {analysis.difficulty}")
print(f"Recommended strategy: {analysis.recommended_strategy}")

# Self-reflection
reflection = SelfReflection(llm)
feedback = reflection.reflect_on_attempt(task, attempt, result, success=False)
improved = reflection.improve_based_on_critique(reasoning, critique)
```

## Creating Custom Benchmarks

```python
from mia.benchmarks.base import BaseBenchmark, BenchmarkTask, TaskResult

class MyCustomBenchmark(BaseBenchmark):
    @property
    def name(self) -> str:
        return "my-custom"
    
    @property
    def description(self) -> str:
        return "My custom benchmark"
    
    def _load_tasks(self) -> List[BenchmarkTask]:
        # Load your tasks
        return [...]
    
    def _evaluate_task(self, task, agent_response, agent_fn) -> TaskResult:
        # Evaluate response
        return TaskResult(
            task_id=task.task_id,
            success=...,
            score=...,
        )
```

## Performance Tips

1. **Use appropriate batch sizes**: Larger batches for simpler benchmarks
2. **Enable caching**: Many datasets support HuggingFace caching
3. **Parallel execution**: Some benchmarks support parallel task execution
4. **Monitor resources**: Use meta-cognitive layer for adaptive behavior

## Output Format

Results are saved in JSON format:

```json
{
  "benchmark": "arc-agi",
  "timestamp": "2024-01-15T10:30:00",
  "metrics": {
    "total_tasks": 100,
    "successful_tasks": 72,
    "accuracy": 0.72,
    "avg_time_per_task": 5.3
  },
  "task_results": [...]
}
```

## Leaderboard Comparison

For comparison with published results, see:
- ARC-AGI: https://lab42.global/arc/
- SWE-bench: https://www.swebench.com/
- GAIA: https://huggingface.co/spaces/gaia-benchmark/leaderboard
- MMMU: https://mmmu-benchmark.github.io/

## Architecture Overview

```
mia/
├── benchmarks/
│   ├── __init__.py          # Module exports
│   ├── base.py              # Base classes
│   ├── arc_agi.py           # ARC-AGI benchmark
│   ├── gaia.py              # GAIA benchmark
│   ├── swe_bench.py         # SWE-bench benchmark
│   ├── gpqa.py              # GPQA benchmark
│   ├── webvoyager.py        # WebVoyager benchmark
│   ├── osworld.py           # OSWorld benchmark
│   ├── mmmu.py              # MMMU benchmark
│   └── runner.py            # Unified runner
├── reasoning/
│   ├── __init__.py          # Reasoning strategies
│   ├── solvers.py           # Domain-specific solvers
│   └── metacognition.py     # Meta-cognitive layer
```

## Contributing

When adding new benchmarks:
1. Extend `BaseBenchmark` class
2. Implement `_load_tasks()` and `_evaluate_task()`
3. Add to `BENCHMARK_REGISTRY` in runner.py
4. Update this documentation

## License

Part of the M.I.A project. See main LICENSE file.
