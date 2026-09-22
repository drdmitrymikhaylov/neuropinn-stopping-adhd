# Reaction times, ADHD, and a solver that refuses to flatter itself

**A physics-informed solver for the decision equation, applied to 79 children, and a plain account of what it does not buy you.**

![Validation](figures/01_validation.png)

---

> ### Source code is not public
>
> The repository holding the code is private. **The source is available for
> technical review under NDA.** Contact me through the links at the end of
> this page.

---

## In one paragraph

A reaction time is the first time a noisy accumulation of evidence reaches a
decision boundary. That first-passage time obeys a differential equation. This
repository solves that equation with a neural network, fits the solution to
every child in an open dataset, and then asks the question most papers of this
kind skip. **Does any of it classify better than counting correct answers?**
The answer, on these data, is no. That result is the point of the repository,
not a footnote in it.

## The model

Evidence accumulates with drift *v* until it reaches one of two boundaries
separated by *a*. The probability of having reached the correct boundary by
time *t* satisfies the backward Kolmogorov equation

    ∂U/∂t = v ∂U/∂x + ½ ∂²U/∂x²,   U(0,t) = 0,  U(a,t) = 1,  U(x,0) = 0.

Substituting *x = a·u* and *t = a²·τ* turns it into

    ∂U/∂τ = (v·a) ∂U/∂u + ½ ∂²U/∂u².

The two parameters collapse into one, **w = v·a**. A single network
*N(u, τ, w)* is therefore the solution for every drift rate and every boundary
separation at once. It is trained once, then evaluated for any child in
microseconds, with gradients.

The reaction-time density is ∂U/∂τ, the same derivative the residual already
contains. The likelihood a child is fitted with comes out of the differential
equation itself.

### Checks on the solver

Two checks, in order. The exact series (Navarro & Fuss, 2009) is verified
against a brute-force simulation of the stochastic process. Mean absolute error
is 0.014 on densities that peak near 2, and the share of correct responses is
0.862 simulated against 0.858 from the series. Then the network is checked
against that series across its whole parameter range: **median relative error
5.2 % at the density peak, worst case 6.4 %.**

![Recovery](figures/02_recovery.png)

Parameter recovery on data generated with known values, fitted exactly as a
child is fitted:

| Trials | drift *v* | boundary *a* | non-decision *t₀* |
|---|---|---|---|
| 50 | 0.97 | 0.93 | 0.96 |
| 200 | 0.98 | 0.98 | 0.99 |
| 800 | 0.99 | 0.99 | 1.00 |

Fifty trials are enough. None of the 79 fits ended against a parameter bound.

## Numbers

Open data: 79 children, 35 with an ADHD diagnosis, n-back tasks, a median of
639 usable correct trials each.

![Groups](figures/03_groups.png)

| Measure | ADHD | Control | Cohen's *d* | *p* |
|---|---|---|---|---|
| drift rate *v* | 0.92 ± 0.60 | 1.48 ± 0.68 | **−0.87** | <0.001 |
| boundary *a* | 1.46 ± 0.16 | 1.66 ± 0.29 | **−0.82** | 0.001 |
| non-decision *t₀* | 0.184 ± 0.058 | 0.216 ± 0.060 | −0.55 | 0.013 |
| mean reaction time | 0.623 ± 0.086 | 0.641 ± 0.063 | −0.24 | 0.194 |
| reaction-time variability | 0.300 ± 0.055 | 0.269 ± 0.042 | +0.64 | 0.008 |
| slow tail | 0.085 ± 0.044 | 0.083 ± 0.034 | +0.03 | 0.577 |

Mean reaction time separates nothing. The model parameters separate strongly,
and they say what kind of difference it is. Evidence accumulates more slowly,
and less of it is demanded before answering.

That is the result a paper would report. Here is the part a paper usually
leaves out.

### Accuracy alone classifies better

Leave-one-out cross-validated AUC for telling the two groups apart:

| Features | AUC |
|---|---|
| **accuracy alone** | **0.729** |
| accuracy + reaction-time variability | 0.712 |
| four classical summary measures | 0.712 |
| three model parameters | 0.703 |
| model parameters + accuracy | 0.713 |
| mean reaction time alone | 0.492 |

The drift rate correlates **0.95** with plain accuracy. It is very nearly a
restatement of it. Solving a partial differential equation with a neural
network, fitting three parameters per child and recovering them at r ≥ 0.93
produces a classifier that is *worse* than the proportion of correct answers.

This is worth stating in full because the literature on decision models in
ADHD reports reduced drift rate as a finding in its own right. On these data it
carries no information that accuracy did not already carry.

What the model does add is separation of causes rather than discrimination.
The boundary *a* correlates with reaction-time variability (−0.55) and not
primarily with accuracy. It is lower in the ADHD group too. Accuracy alone
cannot tell you whether a child is slower at accumulating evidence or simply
demands less of it before committing. The model can, at the cost of assuming
the model.

## Caveats

![Example fits](figures/04_example_fit.png)

The fitted density peaks earlier and decays more slowly than the data.
Quantified as a Kolmogorov-Smirnov distance between the empirical and model
distributions, the median child sits at **0.204**. That is a poor fit, not a
marginal one.

The obvious suspect was that each child's trials pool eight task conditions of
different difficulty, and that a mixture of eight decision processes is not a
decision process. That was tested by refitting the largest single condition per
child. **The misfit got worse, not better.** The median went 0.204 → 0.235, and
the fit improved for only 30 % of children. The explanation was wrong. The
extra parameters of a fuller model, across-trial variability in drift and
starting point, which Ratcliff's complete formulation includes and this one
omits, are the more likely cause.

The parameters above are therefore fitted with a model that demonstrably does
not describe these reaction-time distributions well. They should be read as a
three-number summary of distribution shape, not as measurements of a child's
cognition.

### What would make this worth more

- Across-trial variability in drift and starting point. This is the standard
  extension, and the one the misfit points at.
- Error reaction times, which this dataset has and this analysis discards. They
  identify the starting point, which is fixed at the midpoint here.
- A dataset where the model beats accuracy. That would need a task on which
  accuracy is near ceiling and all the information is in the timing.

## Page history

This page replaced an earlier description that framed the same work as games
for children with ADHD. It is not a game. It is a solver that reads
ADHD-relevant parameters out of reaction times, and I wanted the page to say
so.

Two errors of my own are recorded in the private repository and belong here
too. The first fit used correct-response times only. That left the boundary
unidentified, and every child ended up pinned at the upper bound of the
boundary parameter. The fix was to put the error count into the likelihood:
the probability of a correct response, sigmoid(v·a), pins the drift-boundary
product.

The second error was optimising the drift rate directly. That let the product
*w* run well beyond the range the solver was trained on, into a region where no
solution exists. The fix was to fit (*w*, *a*) and derive *v = w/a*. Parameter
recovery rose to the values in the table above only after that change.

The refuted explanation for the poor fit, pooled task conditions, is kept in
the repository as well and is described under Caveats.

## Sources

Data: OpenNeuro **ds002424**, working memory and reward in children with and
without ADHD. 79 children, trial-level onset, accuracy and response time.
Only the behavioural event files are used.

Model: Ratcliff (1978) for the two-boundary accumulation model; Navarro & Fuss
(2009) for the exact first-passage density used as the reference the solver is
checked against. The probability of a correct response from the midpoint,
sigmoid(v·a), is the standard gambler's-ruin result and is what lets accuracy
identify the drift-boundary product.

## Licence and contact

Documentation, figures and result files in this repository: CC BY 4.0. Source
code is held in a private repository, all rights reserved, and is available
under NDA.

**Prof. Dr. Dmitry Mikhaylov**, Abu Dhabi, UAE

[LinkedIn](https://www.linkedin.com/in/dmitry-mikhaylov) ·
[ORCID](https://orcid.org/0009-0009-2108-6820) ·
[Substack](https://dmitrymikhaylov.substack.com)
