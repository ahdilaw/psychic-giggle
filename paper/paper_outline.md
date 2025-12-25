# Paper Outline: From FLOPs to Compound Efficiency Measures

## New Framing: Survey + Analysis + Practical Guide
**NOT**: "We invented FRM, here's why it's great"
**YES**: "We ran 1,527 benchmarks exploring efficiency measurement—here's what we learned"

---

## Core Narrative Arc

### Act 1: The Problem Everyone Knows But Ignores
- FLOPs limitations are well-documented (ShuffleNet V2, etc.)
- Yet the field keeps using FLOPs as the primary efficiency metric
- Why? Because alternatives require expensive benchmarking

### Act 2: Our Research Journey (Honest Account)
- Started with inertial convolutions → discovered FLOPs limitations firsthand
- Tried theoretical approach (KRT) → ratio-based transfer proved challenging
- Tried compound metric (FRM) → recognized: "not sufficient as contribution alone"
- Pivoted to large-scale empirical analysis → discovered surprising transferability

### Act 3: What We Actually Learned
- FLOPs disagreements with compound metrics are systematic and quantifiable (774 cases)
- Rankings transfer better than raw metrics (surprising!)
- Architecture determines predictability more than hardware
- No universal solution—context determines best metric

### Act 4: Practical Guidance
- Decision framework for metric selection
- When to use FLOPs, when to benchmark, when to transfer
- Honest limitations and open problems

---

## SECTION 1: Introduction (1.5 pages)

### §1.1: The Efficiency Measurement Crisis
- **The $100K problem**: Comprehensive benchmarking is prohibitively expensive
- **The FLOPs trap**: We use FLOPs because it's cheap, despite knowing its limitations
- **The persistence paradox**: Why does the field keep using metrics with known issues?

### §1.2: Our Research Journey (Honest Framing)
- "This paper reports findings from an extensive exploration of efficiency measurement"
- Started with a specific architecture problem (inertial convolutions)
- Evolved through challenging theoretical approaches
- Culminated in large-scale empirical analysis

### §1.3: Contributions (Reframed)
1. **Systematic documentation** of FLOPs limitations (774 disagreement cases where FLOPs rankings disagree with compound metrics by ≥2 positions)
2. **Comparative analysis** of efficiency measurement approaches
3. **Empirical discovery** of ranking transferability across platforms
4. **Practical framework** for selecting appropriate efficiency metrics by use case

### §1.4: Paper Organization
- Section 2: The FLOPs Fallacy (what's different about current practice)
- Section 3: Alternative Approaches (survey of existing methods)
- Section 4: Our Empirical Exploration (compound metrics, transfer attempts)
- Section 5: The Transferability Discovery (surprising finding)
- Section 6: Practical Framework (guidance for researchers)
- Section 7: Discussion and Conclusion

---

## SECTION 2: The FLOPs Fallacy (1.5 pages)

### §2.1: Historical Context
**The FLOPs Assumption Era (2015-2018)**:
- VGG, ResNet papers: FLOPs as efficiency proxy
- Implicit assumption: FLOPs ≈ Runtime
- Worked reasonably well for similar architectures

**Growing Skepticism (2018-2020)**:
- ShuffleNet V2 (2018): "FLOPs is an indirect metric"
- EfficientNet (2019): Memory access patterns matter
- Dynamic Networks (2020): Conditional computation breaks FLOPs

**Current State (2020-present)**:
- Everyone acknowledges limitations
- But FLOPs remain dominant in papers
- Why? Reproducibility, comparability, cost

### §2.2: Our Inertial Convolution Case Study
**The Problem That Started Our Journey**:
- Inertial convolution: 80% FLOPs reduction vs standard conv
- Expected: Major speedup
- Measured: 10× slower on actual hardware
- Root cause: Memory access patterns, operator fusion limitations

**Key Insight**: FLOPs can show significant discrepancies from measured efficiency, not just noise

### §2.3: Systematic Empirical Evidence
**Our 1,527 Benchmark Analysis**:
- 13 models × 106 devices × multiple frameworks
- Identified 774 cases where FLOPs ranking ≠ compound metric ranking (±2 positions)
- 95.8% of device configurations show disagreements
- Inconsistency rate: ~10% of deployment decisions

**Illustrative Cases**:

| Model | FLOPs Prediction | Measured Efficiency | Rank Shift |
|-------|------------------|---------|-------|
| LeViT | Rank #2 (efficient) | Rank #7 | +5 positions |
| SqueezeNet | Rank #5 (modest) | Rank #2 | -3 positions |

**Statistical Validation**:
- Mann-Whitney U: p < 0.0001 for FLOPs vs runtime ranking differences
- Effect size: Cohen's d = 0.73 (medium-large)

### §2.4: Why FLOPs Fails (Technical Analysis)
**Memory-Bound Operations**:
- Attention mechanisms: High data movement, low arithmetic intensity
- Depthwise convolutions: Memory-bound despite low FLOPs

**Hardware-Specific Optimizations**:
- Operator fusion: Fused ops have different FLOPs but same runtime
- Quantization: INT8 on NPU vs FP32 on GPU
- Accelerator matching: Depthwise on DSP vs standard conv on GPU

**Framework Effects**:
- TFLite vs ONNX vs PyTorch can differ by 2-3×
- Same FLOPs, vastly different runtime

---

## SECTION 3: Alternative Approaches - A Survey (2 pages)

### §3.1: Single-Number Metrics

**NetScore (Li et al., 2019)**:
- Formula: NetScore = log(Accuracy^α / Params^β / FLOPs^γ)
- Pros: Simple, comparable
- Cons: Arbitrary weight parameters, still FLOPs-based

**Time-to-Accuracy (Coleman et al., 2019)**:
- Metric: Wall-clock time to reach target accuracy
- Pros: Directly measures what matters (training)
- Cons: Not applicable to inference, requires full training

**Energy-Delay Product (Horowitz, 2014)**:
- Formula: EDP = Energy × Latency
- Pros: Captures both speed and power
- Cons: Energy measurement is hard, device-specific

**Roofline Model (Williams et al., 2009)**:
- Analysis: Compute-bound vs memory-bound characterization
- Pros: Explains performance bottlenecks
- Cons: Requires detailed hardware knowledge, per-layer analysis

### §3.2: Multi-Number Frameworks

**MLPerf Inference (Mattson et al., 2020)**:
- Approach: Standardized benchmarks across devices
- Metrics: Throughput, latency (multiple percentiles), energy
- Pros: Industry standard, reproducible
- Cons: Expensive to run, limited model set

**AIoTBench (Luo et al., 2021)**:
- Focus: Edge/IoT devices
- Metrics: Latency, memory, accuracy across device classes
- Pros: Comprehensive edge coverage
- Cons: No cross-platform prediction

**AI Benchmark (Ignatov et al., 2019)**:
- Approach: Mobile device benchmarking app
- Metrics: Multiple models, CPU/GPU/NPU paths
- Pros: Large device coverage (1000+ devices)
- Cons: Limited to mobile, closed methodology

### §3.3: Pareto Frontier Approaches

**Multi-Objective NAS**:
- Approach: Compute accuracy-efficiency Pareto frontiers
- Used in: NSGA-Net, ProxylessNAS, FBNet
- Limitation: Computed independently per device

**The Device-Specificity Problem**:
- Different devices → Different Pareto frontiers
- Comprehensive evaluation requires O(models × devices) benchmarks
- Our question: Can frontiers transfer across devices?

### §3.4: Compound Metrics (Including Our FRM Exploration)

**Motivation for Compound Metrics**:
- Balance multiple efficiency dimensions
- Single number enables ranking
- Device normalization captures hardware interactions

**Our FRM Case Study**:
```
FRM = (ratio_flops × ratio_latency × ratio_memory)^(1/3)
```

**Why Geometric Mean?**:
- Balanced weighting (no domination)
- Multiplicative relationships (efficiency compounds)
- Outlier robustness

**Important Context (Advisor Feedback)**:
> "FRM as a solution in itself is not a good measure. There needs to be an application that gets enabled by this."

**Our Realization**: FRM isn't the contribution—it's a tool for exploring transferability

---

## SECTION 4: Our Empirical Exploration (2 pages)

### §4.1: Research Evolution (Honest Account)

**Phase 1: The Inertial Convolution Problem**
- Discovered FLOPs failure firsthand
- Motivation: Find better efficiency prediction

**Phase 2: Theoretical Approach (KRT)**
- Kernel Ratio Transfer: Predict runtime from operation ratios
- Hypothesis: If model A is 2× faster than B on GPU, also 2× on Edge
- Result: **Didn't work well** (r² < 0.5 on unseen devices)
- Lesson: Hardware-software interactions too complex for simple transfer

**Phase 3: Compound Metric Exploration (FRM)**
- Tried combining FLOPs + latency + memory
- Achieved good stability (ρ=0.956)
- But: Advisor correctly identified this as insufficient

**Phase 4: Large-Scale Empirical Analysis**
- Pivoted to systematic benchmarking
- 1,527 runs across 106 devices
- Discovered surprising transferability patterns

### §4.2: Experimental Setup

**Scale**:
| Dimension | Count | Details |
|-----------|-------|---------|
| Models | 13 | CNN, Transformer, Hybrid |
| Devices | 106 | GPU (7), CPU (4), Edge (93) |
| Frameworks | 3 | ONNX, PyTorch, TFLite |
| Total Runs | 1,527 | Comprehensive coverage |

**Models Evaluated**:
- Standard CNNs: ResNet18/50, DenseNet121
- Mobile CNNs: MobileNetV2/V3, MNASNet, SqueezeNet
- Efficient: EfficientNet-B0, Inception-V3
- Transformers: LeViT-128S, DeiT-Tiny
- Hybrid: ConvNeXt-Tiny, MobileViT-XXS

**Hardware Diversity**:
- Datacenter GPUs: A100, H100, H200, L40S, RTX series
- Cloud CPUs: Azure D-series, E-series
- Edge Devices: Pixel 2-9, Galaxy S21-S24, OnePlus, Xiaomi, 70+ more

### §4.3: What We Measured

**Primary Metrics**:
- Latency: Median of 100 inference runs
- Memory: Peak allocation during inference
- FLOPs: Computed from architecture
- Accuracy: Top-1 on ImageNet

**Derived Metrics**:
- FRM: Geometric mean of normalized ratios
- Device-specific rankings
- Cross-platform transfer correlations

### §4.4: Compound Metric Stability Analysis

**Within-Platform Stability**:
- FRM rank correlation: ρ = 0.956 ± 0.047
- Latency rank correlation: ρ = 0.746 ± 0.190
- FRM is 4.8× more stable (CV: 0.29 vs 1.40)

**Framework Invariance**:
- ONNX ↔ PyTorch: ρ = 0.984
- Cross-framework mean: ρ = 0.947

**Interpretation**: Compound metrics can provide stable rankings, but stability alone isn't a contribution

---

## SECTION 5: The Transferability Discovery (2 pages)

### §5.1: Our Failed Theoretical Approach (KRT)

**The Kernel Ratio Transfer Hypothesis**:
- Theory: Efficiency ratios between models are device-invariant
- Formulation: ratio(A/B, Device1) ≈ ratio(A/B, Device2)
- Expected: Build predictor from ratios

**Why It Failed**:
- Assumption too strong
- Hardware-model interactions are complex
- Different accelerators favor different operations
- Quantization effects are non-linear

**Lesson Learned**: Pure theoretical approaches insufficient for cross-platform prediction

### §5.2: The Empirical Surprise

**What We Actually Found**:
- Rankings (not ratios) transfer remarkably well
- GPU → Edge: ρ = 0.961 ± 0.026
- Overall cross-tier: ρ = 0.907

**Table: Cross-Platform Transfer Correlations**
| Transfer | FRM ρ | Latency ρ | Improvement |
|----------|-------|-----------|-------------|
| GPU → Edge | 0.961 | 0.582 | +65% |
| CPU → Edge | 0.887 | 0.503 | +76% |
| Edge → GPU | 0.968 | 0.620 | +56% |
| Overall | 0.907 | 0.582 | +56% |

**Why Rankings Transfer Better Than Metrics**:
- Architectural properties dominate (compute patterns, memory access)
- Platform effects are multiplicative (scale rankings, don't reorder)
- Geometric mean smooths platform-specific noise

### §5.3: Architecture-Dependent Predictability

**Our Key Finding**: Architecture family predicts transferability

**Transfer Success by Architecture**:
| Family | Prediction Error | Interpretation |
|--------|------------------|----------------|
| Standard CNN | 5.2% | Very predictable |
| Mobile CNN | 7.8% | Mostly predictable |
| Efficient CNN | 6.4% | Predictable |
| Transformer | 23.4% | **Unpredictable** |
| Hybrid | 15.6% | Mixed |

**Why Transformers Fail**:
- Attention efficiency is highly platform-dependent
- Flash Attention on GPU: 2-4× speedup
- No equivalent on edge NPUs
- Memory bandwidth bottleneck varies dramatically

**Practical Implication**: CNN rankings transfer; validate transformers on target device

### §5.4: Pareto Frontier Transferability

**Application**: Using transferability for practical deployment

**Method**:
1. Benchmark models on source device (e.g., RTX 4090)
2. Compute accuracy-efficiency Pareto frontier
3. Predict same frontier applies to target devices

**Results**:
| Source | Target | Precision | Recall | F1 |
|--------|--------|-----------|--------|-----|
| RTX 4090 | Edge (93) | 0.92 | 0.89 | 0.90 |
| Overall | | 0.89 | 0.86 | 0.87 |

**Cost Savings**: 1,378 runs → 13 runs (99.5% reduction) with 87% accuracy

---

## SECTION 6: A Practical Framework (1.5 pages)

### §6.1: Decision Framework for Metric Selection

**Figure: Metric Selection Decision Tree**

```
What's your use case?
├── Early architecture exploration
│   └── Use FLOPs (cheap, sufficient for rough filtering)
├── Research paper comparison
│   ├── Single device class → Latency on reference device
│   └── Cross-platform claim → Multi-device benchmark or state caveat
├── Multi-platform deployment
│   ├── CNN-dominated models → Transfer from reference device
│   └── Transformer-heavy → Benchmark on representative targets
├── Production deployment
│   └── Always benchmark on actual target device
└── Neural Architecture Search
    ├── Search phase → FLOPs or proxy
    └── Final selection → Target device benchmark
```

### §6.2: Recommendations by Context

**For Research Papers**:
- **Minimum**: Report FLOPs + latency on one reference device
- **Better**: Include memory footprint
- **Best**: Multi-device benchmarks or explicit single-device caveat
- **Avoid**: FLOPs-only efficiency claims for novel architectures

**For Multi-Platform Deployment**:
- **If CNNs only**: Transfer from accessible device (expect ρ > 0.9)
- **If transformers**: Budget for target device validation
- **If cost-constrained**: Prioritize edge device benchmarks (most variable)

**For NAS and AutoML**:
- **Search phase**: FLOPs proxy acceptable for efficiency term
- **Candidate selection**: Move to hardware-in-the-loop
- **Final validation**: Always on target device

**For Production Systems**:
- **No shortcuts**: Benchmark on actual deployment hardware
- **Include variance**: Report p50, p95, p99 latencies
- **Monitor**: Production metrics often differ from benchmarks

### §6.3: What We Recommend Based on Our Experience

**Do**:
- Use compound metrics for cross-platform comparison
- Expect CNN rankings to transfer across hardware tiers
- Validate transformer models on target devices
- Report measurement methodology for reproducibility

**Don't**:
- Trust FLOPs alone for novel architectures
- Assume GPU benchmarks perfectly predict edge performance
- Ignore memory footprint (critical for edge deployment)
- Over-engineer: Sometimes FLOPs is good enough

### §6.4: Honest Limitations of Our Recommendations

**What We Don't Know**:
- Novel architectures (Mamba, RWKV, etc.): No transfer data
- Very large models (LLMs): Different efficiency dynamics
- Training efficiency: Our analysis is inference-focused
- Dynamic batching: All results at batch=1

**Where Our Framework May Fail**:
- Rapidly evolving hardware (new accelerator generations)
- Novel operation types (custom CUDA kernels)
- Extreme quantization (binary/ternary networks)

---

## SECTION 7: Discussion and Conclusion (1 page)

### §7.1: What We Learned

**About FLOPs**:
- Failures are systematic, not random noise
- 10% error rate in deployment decisions is significant
- Transformer architectures are particularly susceptible

**About Alternatives**:
- Compound metrics provide stability but aren't magic
- Rankings transfer better than absolute metrics
- Architecture family is the key predictability factor

**About Our Research Process**:
- Theoretical approaches (KRT) didn't pan out
- Large-scale empirical analysis revealed unexpected patterns
- Advisor feedback was crucial in reframing contribution

### §7.2: The State of Efficiency Measurement

**Current Reality**:
- No silver bullet metric exists
- Context determines appropriate approach
- Comprehensive benchmarking remains expensive

**Progress Made**:
- Better understanding of FLOPs limitations
- Quantified transferability boundaries
- Practical decision framework available

**Open Challenges**:
- Energy measurement standardization
- Novel architecture generalization
- Automated metric selection tools

### §7.3: Recommendations for the Field

**For Researchers**:
- Be explicit about efficiency metric limitations
- Consider multi-device evaluation for deployment claims
- Report methodology for reproducibility

**For Practitioners**:
- Use our transfer findings to reduce benchmarking cost
- Validate transformers on target hardware
- Don't over-trust any single metric

**For Tool Builders**:
- Standardize efficiency measurement APIs
- Build cross-platform benchmark databases
- Develop automated metric selection tools

### §7.4: Conclusion

**Summary**:
We conducted a large-scale empirical exploration of efficiency measurement approaches, spanning 1,527 benchmark evaluations across 106 devices. Our journey—from discovering FLOPs failures firsthand, through failed theoretical approaches, to surprising empirical findings—provides practical guidance for the research community.

**Key Takeaways**:
1. FLOPs fails systematically (774 documented cases, 10% error rate)
2. Rankings transfer better than metrics (ρ=0.907 cross-tier)
3. Architecture determines predictability (CNNs: 5%, Transformers: 23% error)
4. No silver bullet—use our decision framework to choose appropriate metrics

**Final Thought**:
The efficiency measurement problem won't be solved by a single metric. What the field needs is better understanding of when each approach works, which is what we've tried to provide.

---

## FIGURES AND TABLES PLAN

### Main Figures (5)
1. **Figure 1**: FLOPs vs Reality disagreement visualization (LeViT, SqueezeNet cases)
2. **Figure 2**: Survey of efficiency metrics (taxonomy diagram)
3. **Figure 3**: Our research journey (timeline: inertial conv → KRT → FRM → empirical)
4. **Figure 4**: Cross-platform transfer heatmap
5. **Figure 5**: Decision framework flowchart

### Main Tables (5)
1. **Table 1**: FLOPs disagreement summary (774 cases)
2. **Table 2**: Survey of efficiency metrics (comparison matrix)
3. **Table 3**: Experimental setup (1,527 runs details)
4. **Table 4**: Cross-platform transfer correlations
5. **Table 5**: Architecture-dependent predictability

---

## KEY NUMBERS (Reframed)

| Finding | Value | New Framing |
|---------|-------|-------------|
| Disagreement cases | 774 | Evidence of systematic FLOPs failure |
| Error rate | 10% | Significant for deployment decisions |
| Rank stability | ρ=0.956 | Compound metrics provide stability |
| Cross-tier transfer | ρ=0.907 | Rankings transfer surprisingly well |
| CNN error | 5.2% | Highly predictable architecture |
| Transformer error | 23.4% | Architecture-dependent limitation |
| Cost reduction | 99.5% | Practical benefit of transfer |

---

## TONE EXAMPLES

**Before (FRM-centric)**:
> "We introduce FRM, a novel efficiency metric that achieves superior cross-platform stability."

**After (Survey/Analysis)**:
> "Through extensive benchmarking, we find that compound metrics like FRM provide more stable rankings, though they are not a universal solution."

**Before**:
> "Our key contribution is the FRM metric."

**After**:
> "Our key contributions are: (1) systematic documentation of FLOPs limitations, (2) comparative analysis of alternatives, and (3) practical guidance for metric selection."

**Before**:
> "FRM outperforms latency with 56% higher transfer correlation."

**After**:
> "We observe that efficiency rankings transfer better than raw metrics—a finding that enables practical cost reduction in cross-platform evaluation."

---

## WRITING CHECKLIST

- [ ] FRM demoted to "case study," not "contribution"
- [ ] KRT failure included with lessons learned
- [ ] Advisor feedback acknowledged (tastefully)
- [ ] Survey of existing approaches included
- [ ] Decision framework is practical and honest
- [ ] Limitations discussed throughout
- [ ] Tone is humble expertise, not metric promotion
- [ ] Research journey narrative is clear
- [ ] All claims framed as empirical findings, not superiority claims
