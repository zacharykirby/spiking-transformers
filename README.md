# Hybrid Transformer-SNN Architecture Exploration

## Overview
This repository documents an experimental journey into combining transformer architectures with spiking neural networks (SNNs). The goal is to explore whether these fundamentally different approaches to neural computation can complement each other, potentially leading to more efficient AI systems that run on consumer hardware.

## Motivation
Current large language models are incredibly capable but computationally expensive. Transformers excel at parallel processing and pattern matching through dense attention mechanisms, but they require massive compute resources. Meanwhile, spiking neural networks offer energy-efficient, event-driven computation inspired by biological brains - but struggle with the dense transformations that make transformers powerful.

What if we could combine the strengths of both?

## Background
**Transformers:**
- Global attention mechanisms (O(n²) complexity)
- Dense matrix operations (GPU-friendly)
- Parallel sequence processing
- Excellent at pattern matching and reasoning

**Spiking Neural Networks:**
- Temporal, event-driven computation
- Sparse activation (only "fire" when needed)
- Neuromorphic hardware-friendly
- Implicit temporal attention through spike timing

## Research Questions
1. Can SNNs serve as a temporal attention filter for transformers, reducing the O(n²) attention bottleneck?
2. Is there a tractable interface between discrete spike trains and continuous transformer representations?
3. Can we achieve meaningful computation on consumer GPUs by offloading continuous state maintenance to SNNs?
4. What tasks benefit most from hybrid architectures vs. pure transformer or pure SNN approaches?

## Project Phases

### Phase 1: Fundamentals (Current)
- [ ] Implement minimal transformer from scratch (following Karpathy's approach)
- [ ] Understand attention mechanisms deeply (multi-head, self-attention, positional encoding)
- [ ] Implement basic SNN (LIF neurons, STDP learning)
- [ ] Explore spike encoding schemes (rate coding, temporal coding, population coding)

### Phase 2: Interface Exploration
- [ ] Experiment with spike train → continuous vector conversion
- [ ] Test different encoding schemes for SNN output
- [ ] Investigate training strategies (separate vs. joint training)
- [ ] Prototype simple hybrid architectures

### Phase 3: Hybrid Architectures (Speculative)
- [ ] SNN as temporal state maintenance + transformer for reasoning steps
- [ ] SNN-based attention filtering before dense transformer attention
- [ ] Hierarchical prediction with SNN base layer and transformer refinement
- [ ] Compare performance/efficiency metrics vs. pure approaches

## Wild Ideas & Long-Term Speculation

### Two-Stage Attention
Use SNNs as a fast, coarse-grained attention mechanism that continuously processes input streams, identifying salient patterns through spike timing. The transformer then performs expensive dense attention only on SNN-filtered information, potentially reducing the effective sequence length for attention computation.

### Continuous Temporal Context
SNNs maintain a rolling temporal representation (like working memory) that's always updating. The transformer samples this state periodically when it needs to do explicit reasoning, rather than processing every input token through full attention.

### Neuromorphic-GPU Hybrid Systems
Different components run on different hardware: SNNs on neuromorphic chips (Intel Loihi) for continuous temporal processing, transformers on GPUs for periodic dense reasoning. Explores heterogeneous computing for AI.

### Predictive Temporal Coding
Drawing from predictive coding in neuroscience: SNNs generate predictions about upcoming inputs, spikes represent prediction errors. Transformer layers use these prediction errors as attention signals, focusing compute on surprising/salient information.

## Technical Challenges

### The Spike-to-Continuous Interface
Spikes are discrete events; transformers need continuous vectors. Current approaches:
- **Rate coding:** Count spikes over time windows (simple but loses temporal precision)
- **Temporal coding:** Encode spike timing as continuous values (information-rich but complex)
- **Population coding:** Many neurons per value (brain-like but parameter-heavy)

### Training Through Both Architectures
- Spikes aren't differentiable (step functions)
- Surrogate gradients work for SNNs alone, but backprop through the interface?
- May require separate training pipelines or alternating optimization

### Hardware Constraints
- SNNs don't benefit from GPUs (sparse, asynchronous vs. dense, parallel)
- Neuromorphic hardware is specialized and less accessible
- Need to prove value on consumer hardware to be practically useful

## References & Inspiration
- "Attention Is All You Need" (Vaswani et al., 2017) - The transformer paper
- Andrej Karpathy's "Let's build GPT" tutorial
- PredNet: Hierarchical predictive coding with LSTMs
- Norse/snnTorch: PyTorch frameworks for SNNs
- Brain minicolumn research & cortical computation

## Why This Project?
This is a passion project exploring fundamental questions about neural architectures, inspired by:
- Graduate work on PredNET (LSTM-based motion prediction)
- Background in brain computation labs studying minicolumns and SNNs
- Reading "The Metamorphosis of Prime Intellect" and thinking about paths to more efficient AI
- Missing the hands-on research aspect of ML work

**Not trying to build AGI.** Just exploring whether there's a smarter compute architecture hiding in the intersection of these approaches.

## Current Status
🔨 Phase 1: Building fundamentals, learning transformer architecture deeply

---

*This is a long-term exploration project. Progress will be incremental and experimental. Expect dead ends, pivots, and slow iteration. That's the point.*
