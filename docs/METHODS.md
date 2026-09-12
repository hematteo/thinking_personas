# Methods

This project compares thinking and answer activations in Qwen3-32B. All reported measurements come from decoder block 32, counted from zero, after its MLP. We use bf16 model weights, generate with vLLM, and replay the exact generated token sequences to measure activations.

## Thinking and answers under default instructions

We retained 40 advice prompts from WildChat and 40 math prompts from GSM8K. Advice inputs were screened before generation. Each prompt has a response with thinking enabled and another with thinking disabled.

The headline comparison uses the thinking block and final answer from the same thinking-enabled response. The response with thinking disabled provides an additional baseline. We exclude incomplete or malformed responses, require at least 200 thinking tokens, and omit the first five answer tokens when calculating the headline answer mean.

For each token, we take the dot product of its activation with the published unit Assistant Axis. We average these scores separately within thinking and the answer, then subtract answer from thinking. The original analysis also calculated centered cosine similarity, which subtracts a calibration mean and divides by activation norm. The headline raw-projection analysis was added after the first experiment.

We compare the Assistant Axis with directions derived from the published Skeptic and Judge roles, a direction made from randomly split calibration responses, and a random direction. Before the main measurements, we check that default answers score above roleplay answers on the Assistant Axis.

A separate comparison replays identical text under thinking and answer prefixes. This measures how context changes activations while the text is held fixed; it does not by itself identify a persona change.

## Role and style instructions

The follow-up uses nine roles, three style instructions that preserve AI identity, and a default instruction. Each is applied to 12 fixed questions with two generation seeds, giving 312 responses.

We compare each instruction with the default on the same question and seed, separately for thinking and answers. We then compare the two shifts. Raw projections are the main measurement; centered cosine and omission of the first five answer tokens are additional checks.

The questions cover identity, social situations, and reasoning, with four questions per group. Seeds are averaged within each question before calculating uncertainty. The planned comparisons use an adjustment for testing multiple effects.

## Role geometry

For each response, we retain an average activation vector for thinking and another for the answer. We compare role averages using PCA plots and distances across all 5,120 dimensions. We also repeat the distance comparison after removing the Assistant-axis component.

These analyses describe how the roles are arranged and how much they differ. A tighter group of thinking vectors does not prove that thinking has a distinct persona or an unchanging internal state.

## Uncertainty and limits

Confidence intervals resample prompts or questions, not tokens. The first study uses one generation seed; the follow-up uses two. The geometry analysis is exploratory.

All findings apply to one model, one layer, and the selected prompts and roles. Role and style instructions can produce similar effects. A near-zero projection is not a calibrated measure of persona neutrality. Establishing a causal mechanism would require further experiments, including interventions on activations.
