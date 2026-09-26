Tool-Calling Inference Project : Make speculative decoding faster on tool-calling workloads.

Constraints:  You have 1x H100 for one day.

Target model:  is fixed and frozen at Qwen2.5-Coder-14B-Instruct

Draft Model : Need to decide between  Qwen2.5 0.5 B and 1.5B - based on baseline results. Might have to compare both. 

Method: On policy Distillation. (Not going for industry standard Eagle-3 or newer methods like DFlash due to GPU and time constraints.)

DataSets: ToolBench - Use a 10k subset of tools- to reduce dataset size. Format the tool options in prompts to match Qwens tool formating and chat template. Also need to format multi-turn conversations as single turn with chat history. Train on ToolBench, but also evaluate on a second, differently-formatted set - xLAM. 

Evaluation Metrics :
 1. τ (mean accepted length) and per-token acceptance rate α. 
 2. wall-clock speedup and tokens/sec, vs. both plain autoregressive decoding and the untuned draft. Measure at batch size 1 and one larger batch, because speedup is very batch-sensitive: one reproduction of EAGLE-3 found a 2.3x speedup at batch size 4 dropping to roughly break-even at batch size 32. 
E2E Networks
3. Diagnostic: acceptance by draft position (n-α), to show whether gains hold deeper into the draft.
4. Tool-calling-specific: acceptance split by output region (tool-call JSON vs. free-form text). This is your differentiator. One agentic-serving paper found acceptance is highly bimodal, with structured tool-call regions near 100% and free-form prose sometimes below 10%.

5. Sweeps: draft length k (e.g., 3, 5, 7) and temperature (greedy and T=1), matching DistillSpec's setup.

 6. n-gram / prompt-lookup baseline. It needs no draft model and copies text from the prompt, which is exactly what tool calls do with function names and argument keys from the schema. On code-heavy workloads, suffix decoding reached a 1.45x speedup over baseline and plain n-gram matching 1.10x. 

7. Measurement rigor: verify that speculative outputs are token-identical to target-only greedy decoding and that output lengths match, not just that acceptance went up

Steps: 

1. Create data cleaning and formatting pipeline. Also create code for doing speculative decoding so you can test the two candidate draft models when you connect to GPU. 
2. Rent H100 GPU- Create SSH instance and connect your repo load datasets , Load qwen 14B,  0.5B, 1.5B 
3. Benchmark Qwen 14B and try speculative decoding with O.5B and 1.5B- record appropriate metrics. 
4. Generate training data using prompts from tool Bench.
5. While generating data create training pipelines 
6. Recording metrics , creating visualisations etc - finalizing repo 