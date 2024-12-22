# Rederiving Elastic Weight Consolidation (EWC) 

I first did this rederivation when trying to understand [EWC](https://arxiv.org/abs/1612.00796) in 2019, when experimenting with continual learning at SDSUG AIML lab with Davit Soselia and others. No guarantees that everything is correct, but it mostly seems to check out.



## Definitions

- Data-generating distribution: The mixture of the distributions of the dataset from the first task and the dataset from the second task.

- **$W = w$**: The event that our current parameters $w$ are optimal for our model's ability to model the data-generating distribution.

- **$x$**: The event that some data point or collection of data points $x$ has been generated from the data-generating distribution.

- **$D_a$**: The event that the data in some array $D_a$ has been generated from the data-generating distribution.

- **$D_b$**: The event that the data in some array $D_b$ has been generated from the data-generating distribution.

- **$D$**: The event that the data in some array $D$ (which is the union of the arrays $D_a$ and $D_b$) has been generated from the data-generating distribution. This can be expressed as:
```math
P(D) = P(D_a \cap D_b)
```

## Presuppositions

1. We will use a flat prior for $P(W=w)$. This means that when we have seen no data, we assume that any value of $w$ is just as likely to be the optimal weights for our data-generating distribution (DGD) as any other.

2. We will assume that the non-diagonal elements of the Hessian of the model's negative log-likelihood on the data are negligible. In other words, we assume that the variation in the loss with respect to any given parameter $w_i$ does not significantly depend on any other parameter $w_j$. Mathematically, this means that:
```math
H_{ij} \approx 0 \quad \text{for} \quad i \neq j
```
   Intuitively, this assumption means that how the loss varies with any given parameter $w_i$ does not significantly change due to the influence of other parameters.

3. $D_a$ and $D_b$ are conditionally independent given $W=w$. This can be written as:
```math
P(D_a \cap D_b \mid W=w) = P(D_a \mid W=w) \cdot P(D_b \mid W=w)
```
   Intuitively, this assumption makes sense because the contents of $D_a$ and $D_b$ are dissimilar. Thus, knowing how likely it is for our model to have assigned a high probability to an item in $D_a$ gives us no information about how likely it is for our model to assign a high probability to an item in $D_b$.

   - In supervised learning terms, if we imagine the datasets as key-value pairs (with samples as keys and labels as values), $D_a$ and $D_b$ will not share keys or have similar keys to each other.

4. All gradients and Hessians are with respect to $w$. 

5. $N$ is the number of samples $x$ in $D_a$.

6. $M$ is the dimension of $w$.


## 1. Bayesian Analysis of Training on Two Tasks Sequentially

Let us consider the probability of $W = w$ given $D$ , which represents the probability that our current weights are optimal, assuming that the data in $D$ came from the distribution we’re trying to model.

According to Bayes' Theorem:
```math
P(W = w \mid D) = \frac{P(D \mid W = w) \cdot P(W = w)}{P(D)}
```

Expanding the definition of $P(D)$, we get:
```math
P(W = w \mid D) = \frac{P(D \mid W = w) \cdot P(W = w)}{P(D_a \cap D_b)}
```

From our presupposition, we know that $D_a$ and $D_b$ are conditionally independent given $W = w$. Therefore, 
```math
P(D_a \cap D_b \mid W = w) = P(D_a \mid W = w) \cdot P(D_b \mid W = w)
```

Substituting this into the previous equation:
```math
P(W = w \mid D) = \frac{P(D_a \mid W = w) \cdot P(D_b \mid W = w) \cdot P(W = w)}{P(D_a \cap D_b)}
```

Rearranging the terms:
```math
P(W = w \mid D) = \frac{P(D_a \mid W = w) \cdot P(W = w) \cdot P(D_b \mid W = w)}{P(D_a \cap D_b)}
```

We know that:
```math
P(D_a \mid W = w) \cdot P(W = w) = P(D_a \cap W = w)
```
Thus, we can rewrite the equation as:
```math
P(W = w \mid D) = \frac{P(D_a \cap W = w) \cdot P(D_b \mid W = w)}{P(D_a \cap D_b)}
```

From Bayes' Theorem, we also know that:
```math
P(D_a \cap D_b) = P(D_b \mid D_a) \cdot P(D_a)
```
Therefore, we get:
```math
P(W = w \mid D) = \frac{P(D_a \cap W = w) \cdot P(D_b \mid W = w)}{P(D_b \mid D_a) \cdot P(D_a)}
```

Next, we know that:
```math
\frac{P(D_a \cap W = w)}{P(D_a)} = P(W = w \mid D_a)
```
Thus, we can simplify it to:
```math
P(W = w \mid D) = P(W = w \mid D_a) \cdot \frac{P(D_b \mid W = w)}{P(D_b \mid D_a)}
```

For visibility, let’s take the logarithm of both sides:
```math
\log P(W = w \mid D) = \log P(W = w \mid D_a) + \log P(D_b \mid W = w) - \log P(D_b \mid D_a)
```

Rearranging the terms:
```math
\log P(W = w \mid D) = \log P(D_b \mid W = w) + \log P(W = w \mid D_a) - \log P(D_b \mid D_a)
```

### Analysis

- $\log P(D_b \mid W = w)$ is the log-likelihood of our model on dataset $D_b$. Thus, the probability of $W$ being optimal on $D$ depends on the log-likelihood of our model on dataset $D_b$.
- $\log P(W = w \mid D_a)$ is an intractable posterior term that represents the probability of $W$ being optimal given that $D_a$ has been generated from the data-generating distribution (DGD).
- Finally, we don't care about the $\log P(D_b \mid D_a)$ term because it is independent of and not conditioned on the probability of our weights being optimal, so it's a constant regardless of what parameters we have.

In other words, to maximize $P(W = w \mid D)$, we need to:
1. Maximize $P(D_b \mid W = w)$
2. Maximize $P(W = w \mid D_a)$

To achieve the first, we need to train the model with a negative log-likelihood loss on dataset $D_b$.

The second term, $P(W = w \mid D_a)$, is intractable. However, we can approximate it and instead maximize the approximation.


## 2. Approximating $P(W = w \mid D_a)$

### 2.1 Lemmas

Let’s suppose that we have randomly initialized our model and then trained it with a negative log-likelihood loss on $D_a$. As a result, we have obtained weights $w^*$ such that:

```math
\nabla (-P(D_a \mid W = w^*)) = \nabla (P(D_a \mid W = w^*)) = 0
```

Thus, we have:

```math
\nabla \log P(D_a \mid W = w^*) = 0 2 \qquad \textnormal { (Lemma 1)}
```


Furthermore, we know that the array $D_a$ consists of some independent and identically distributed (iid) points $x$. So, we can express $P(D_a \mid W = w^*)$ as:

```math
P(D_a \mid W = w^*) = \prod_{x} P(x \mid W = w^*)
```

Taking the logarithm of both sides, we get:

```math
\log P(D_a \mid W = w^*) = \sum_{x} \log P(x \mid W = w^*)
```

The gradient of the sum can be written as:

```math
\nabla \left( \sum_{x} \log P(x \mid W = w^*) \right) = \sum_{x} \nabla \log P(x \mid W = w^*) = 0
```

Dividing by $N$, the number of samples, we get:

```math
\frac{1}{N} \sum_{x} \nabla \log P(x \mid W = w^*) = \mathbb{E}[\nabla \log P(x \mid W = w^*)] = 0
```

Thus, we have:

```math
\mathbb{E}[\nabla \log P(x \mid W = w^*)] = 0 \qquad \textnormal { (Lemma 2)}
```

Next, we know that:

```math
P(W = w \mid D_a) = \frac{P(W = w \cap D_a)}{P(D_a)}
```

Thus, the negative log of $P(W = w \mid D_a)$ is:

```math
-\log P(W = w \mid D_a) = -\log P(W = w \cap D_a) + \log P(D_a)
```

Taking the gradient of both sides:

```math
\nabla \left( -\log P(W = w \mid D_a) \right) = \nabla \left( -\log P(W = w \cap D_a) + \log P(D_a) \right) = \nabla \left( -\log P(W = w \cap D_a) \right) + \nabla \log P(D_a)
```

Since $\nabla \log P(D_a) = 0$, we get:

```math
\nabla \left( -\log P(W = w \mid D_a) \right) = \nabla \left( -\log P(W = w \cap D_a) \right)
```

We also know that:

```math
P(D_a \mid W = w) = \frac{P(W = w \cap D_a)}{P(W = w)}
```

Thus, the negative log of $P(D_a \mid W = w)$ is:

```math
-\log P(D_a \mid W = w) = -\log P(W = w \cap D_a) + \log P(W = w)
```

Taking the gradient of both sides:

```math
\nabla \left( -\log P(D_a \mid W = w) \right) = \nabla \left( -\log P(W = w \cap D_a) + \log P(W = w) \right) = \nabla \left( -\log P(W = w \cap D_a) \right) + \nabla \log P(W = w)
```

Since we are using a flat prior for $P(W = w)$, $\nabla \log P(W = w) = 0$. Thus, we get:

```math
\nabla \left( -\log P(D_a \mid W = w) \right) = \nabla \left( -\log P(W = w \cap D_a) \right)
```

Therefore, we conclude that:

```math
\nabla \left( -\log P(W = w \mid D_a) \right) = \nabla \left( -\log P(D_a \mid W = w) \right)
```

As such, we have:

```math
\nabla P(W = w \mid D_a) = \nabla P(D_a \mid W = w) \qquad \textnormal { (Lemma 3)}
```

Similarly, for the Hessian:

```math
H(P(W = w \mid D_a)) = H(P(D_a \mid W = w)) \qquad \textnormal { (Lemma 4)}
```

From Lemma 3 and Lemma 1, we get:

```math
\nabla \log P(W = w^* \mid D_a) = 0 \qquad \textnormal { (Lemma 5)}
```

From Lemma 4, we get:

```math
H(\log P(W = w \mid D_a)) = H(\log P(D_a \mid W = w)) \qquad \textnormal { (Lemma 6)}
```





### 2.2 Laplace Approximation

The Laplace approximation procedure here follows the one outlined in chapter 4.4 of Pattern Recognition and Machine Learning (Bishop). 

Let’s approximate $\log P(W = w \mid D_a)$ using a second-order Taylor series expansion.

```math
\log P(W = w \mid D_a) \approx \log P(W = w^* \mid D_a) + (w - w^*)^T \nabla \log P(W = w^* \mid D_a) + \frac{1}{2} (w - w^*)^T H(\log P(W = w^* \mid D_a)) (w - w^*)
```

From Lemma 5, we know that $\nabla \log P(W = w^* \mid D_a) = 0$. Thus, the above expression simplifies to:

```math
\log P(W = w \mid D_a) \approx \log P(W = w^* \mid D_a) + \frac{1}{2} (w - w^*)^T H(\log P(W = w^* \mid D_a)) (w - w^*)
```

Let $A = -H(\log P(W = w^* \mid D_a))$. Then we can write:

```math
\log P(W = w \mid D_a) \approx \log P(W = w^* \mid D_a) - \frac{1}{2} (w - w^*)^T A (w - w^*)
```

Taking the exponential of both sides, we get:

```math
P(W = w \mid D_a) \approx P(W = w^* \mid D_a) \cdot \exp \left( -\frac{1}{2} (w - w^*)^T A (w - w^*) \right)
```

Next, we approximate $P(W = w \mid D_a)$ as a Gaussian distribution:

```math
P(W = w \mid D_a) \approx \frac{\sqrt{\det(A)}}{(2\pi)^{M/2}} \cdot \exp \left( -\frac{1}{2} (w - w^*)^T A (w - w^*) \right) = \mathcal{N}(w \mid w^*, A^{-1})
```

Substituting back $A$, we get:

```math
P(W = w \mid D_a) \approx \mathcal{N}(w \mid w^*, \left( -H(\log P(W = w^* \mid D_a)) \right)^{-1})
```

From Lemma 6, we know that:

```math
- H(\log P(W = w^* \mid D_a)) = - H(\log P(D_a \mid W = w^*)) = \sum_{x} H(-\log P(x \mid W = w^*)) = N \cdot \frac{1}{N} \sum_{x} H(-\log P(x \mid W = w^*)) = N \cdot \mathbb{E} \left[ H(-\log P(x \mid W = w^*)) \right]
```

Thus:

```math
- H(\log P(D_a \mid W = w^*)) = N \cdot \mathbb{E} \left[ H(-\log P(x \mid W = w^*)) \right]
```

We know that $`H(-\log P(x \mid W = w^*))`$ is the observed Fisher information on $x$. Its expected value is the covariance of $\nabla (-\log P(x \mid W = w^*))$. Therefore:

```math
- H(\log P(D_a \mid W = w^*)) = N \cdot \text{cov} \left( \nabla (-\log P(x \mid W = w^*)) \right)
```

Assuming that the non-diagonal elements of this covariance matrix are zero, we get:

```math
- H(\log P(D_a \mid W = w^*)) = N \cdot \text{diag} \left( \text{var} \left( \nabla (-\log P(x \mid W = w^*)) \right) \right)
```

The variance of $\nabla (-\log P(x \mid W = w^*))$ is given by:

```math
\text{var} \left( \nabla (-\log P(x \mid W = w^*)) \right) = \mathbb{E} \left[ \left( \nabla (-\log P(x \mid W = w^*)) - \mathbb{E} \left[ \nabla (-\log P(x \mid W = w^*)) \right] \right)^2 \right]
```

From Lemma 2, we know that:

```math
\mathbb{E} \left[ \nabla (\log P(x \mid W = w^*)) \right] = 0
```

Thus, the variance simplifies to:

```math
\text{var} \left( \nabla (-\log P(x \mid W = w^*)) \right) = \mathbb{E} \left[ \nabla (-\log P(x \mid W = w^*))^2 \right] = \frac{1}{N} \sum_{x} \nabla (-\log P(x \mid W = w^*))^2
```

Therefore, we get:

```math
- H(\log P(D_a \mid W = w^*)) = N \cdot \text{diag} \left( \frac{1}{N} \sum_{x} \nabla (-\log P(x \mid W = w^*))^2 \right)
```

Thus, the Laplace approximation for $P(W = w \mid D_a)$ becomes:

```math
P(W = w \mid D_a) \approx \mathcal{N} \left( w \mid w^*, \text{diag} \left( \sum_{x} \nabla (-\log P(x \mid W = w^*))^2 \right)^{-1} \right)
```

Let $H^* = \text{diag} \left( \sum_{x} \nabla (-\log P(x \mid W = w^*))^2 \right)$. Then, we have:

```math
P(W = w \mid D_a) \approx \frac{\sqrt{\det(H^*)}}{(2\pi)^{M/2}} \cdot \exp \left( -\frac{1}{2} (w - w^*)^T H^* (w - w^*) \right)
```

Let $C = \frac{\sqrt{\det(H^*)}}{(2\pi)^{M/2}}$. We can express the approximation as:

```math
P(W = w \mid D_a) \approx C \cdot \exp \left( -\frac{1}{2} (w - w^*)^T H^* (w - w^*) \right)
```

Taking the logarithm of both sides, we get:

```math
\log P(W = w \mid D_a) \approx \log C - \frac{1}{2} (w - w^*)^T H^* (w - w^*)
```

Since $\log C$ is a constant, we can ignore it in the optimization process. Thus, to maximize $\log P(W = w \mid D_a)$, we need to minimize:

```math
\frac{1}{2} (w - w^*)^T H^* (w - w^*)
```


## 3. EWC Loss

Returning to our expression for $\log P(W = w \mid D)$, the full expression is:

```math
\log P(W = w \mid D) \approx \log P(D_b \mid W = w) + \log \left( \frac{\sqrt{\det(H^*)}}{(2\pi)^{M/2}} \right) - \frac{1}{2} (w - w^*)^T \text{diag} \left( \sum_{x \in D_a} \nabla (-\log P(x \mid W = w^*))^2 \right) (w - w^*) + \log P(D_b \mid D_a)
```

This expression is maximized by minimizing the following loss function:

```math
J(w) = - \log P(D_b \mid W = w) + \frac{1}{2} (w - w^*)^T \text{diag} \left( \sum_{x \in D_a} \nabla (-\log P(x \mid W = w^*))^2 \right) (w - w^*)
```

The second term can be viewed as a quadratic regularization term. To control its influence on the training process, we introduce a regularization parameter $\lambda$:

```math
J(w) = - \log P(D_b \mid W = w) + \frac{\lambda}{2} \sum_{i=1}^M \text{diag} \left( \sum_{x \in D_a} \nabla (-\log P(x \mid W = w^*))^2 \right)_{i,i} (w_i - w_i^*)^2
```

(Note that in the original paper, $\lambda$ is introduced slightly differently. The diagonal precision matrix is divided by $N$, and our derivation with $\lambda = 1$ can be reobtained by setting $\lambda$ in the original expression to $N$).

To add L2 weight decay, we can include the decay term:

```math
J(w) = - \log P(D_b \mid W = w) + \frac{\lambda}{2} \sum_{i=1}^M \left( \text{diag} \left( \sum_{x \in D_a} \nabla (-\log P(x \mid W = w^*))^2 \right)_{i,i} + \ell_2 \right) (w_i - w_i^*)^2
```


## 4. New Tasks

### 4.1 Taylor Approximation

Let us introduce a new task $D_c$, and redefine $D$ as the union of arrays $D_a$, $D_b$, and $D_c$:

```math
\log P(W = w \mid D) = \log P(D_c \mid W = w) + \log P(W = w \mid D_a \cap D_b) + \log P(D_c \mid D_a \cap D_b)
```

We have already derived an approximation for $\log P(W = w \mid D_a \cap D_b)$. Let $`w_a^*`$ be the optimal point found after training on $D_a$ (the same value as $`w^*`$ in the previous section).

```math
\log P(W = w \mid D_a \cap D_b) \approx \log P(D_b \mid W = w) + \log \left( \frac{\sqrt{\det(H^*)}}{(2\pi)^{M/2}} \right) - \frac{1}{2} (w - w_a^*)^T \text{diag} \left( \sum_{x \in D_a} \nabla (-\log P(x \mid W = w_a^*))^2 \right) (w - w_a^*) + \log P(D_b \mid D_a)
```

This simplifies to:

```math
\log P(W = w \mid D_a \cap D_b) \approx \log P(D_b \mid W = w) - \frac{1}{2} (w - w_a^*)^T \text{diag} \left( \sum_{x \in D_a} \nabla (-\log P(x \mid W = w_a^*))^2 \right) (w - w_a^*) + \text{(constant part with 0 grad)}
```

After training on $D_a$ and $D_b$, the gradient of the above expression is zero. Let $w_b^*$ denote the weights at this stage. Now, we compute the second-order Taylor approximation of the above expression, term by term.

- **0th degree term**:

```math
\log P(D_b \mid W = w_b^*) - \frac{1}{2} (w_b^* - w_a^*)^T \text{diag} \left( \sum_{x \in D_a} \nabla (-\log P(x \mid W = w_a^*))^2 \right) (w_b^* - w_a^*) + \text{(constant part with 0 grad)}
```

- **1st degree term**:

This term is zero because the gradient of the expression is zero at $w_b^*$.

- **2nd degree term**:

  - The Hessian of the constant part with 0 gradient is zero.
  - The Hessian of $`\frac{1}{2} (w - w_a^*)^T \text{diag} \left( \sum_{x \in D_a} \nabla (-\log P(x \mid W = w_a^*))^2 \right) (w - w_a^*)`$ is $`\text{diag} \left( \sum_{x \in D_a} \nabla (-\log P(x \mid W = w_a^*))^2 \right)`$.
  - The Hessian of $\log P(D_b \mid W = w)$ is the same as the Hessian of $\log P(W = w \mid D_b)$ (Lemma 4). Thus, it can be approximated as $\text{diag} \left( \sum_{x \in D_b} \nabla (-\log P(x \mid W = w_b^*))^2 \right)$.

The complete second-degree term is:

```math
\frac{1}{2} (w - w_b^*)^T \left( \text{diag} \left( \sum_{x \in D_a} \nabla (-\log P(x \mid W = w_a^*))^2 \right) + \text{diag} \left( \sum_{x \in D_b} \nabla (-\log P(x \mid W = w_b^*))^2 \right) \right) (w - w_b^*)
```

Thus, the full second-degree Taylor approximation becomes:

```math
\log P(W = w \mid D_a \cap D_b) \approx \frac{1}{2} (w - w_b^*)^T \left( \text{diag} \left( \sum_{x \in D_a} \nabla (-\log P(x \mid W = w_a^*))^2 \right) + \text{diag} \left( \sum_{x \in D_b} \nabla (-\log P(x \mid W = w_b^*))^2 \right) \right) (w - w_b^*) + \text{(constant part with 0 grad)}
```

Therefore:

```math
\log P(W = w \mid D) = \log P(D_c \mid W = w) + \frac{1}{2} (w - w_b^*)^T \left( \text{diag} \left( \sum_{x \in D_a} \nabla (-\log P(x \mid W = w_a^*))^2 \right) + \text{diag} \left( \sum_{x \in D_b} \nabla (-\log P(x \mid W = w_b^*))^2 \right) \right) (w - w_b^*) + \text{(constant part with 0 grad)}
```

### 4.2 Multiple-Task Loss

Generalizing, let $D$ be the union of arrays $D_1, D_2, \dots, D_k$, and let $w_t^*$ be the optimum found after training on task $t$.

The loss becomes:

```math
\log P(W = w \mid D) \approx \log P(D_k \mid W = w) + \frac{1}{2} (w - w_{k-1}^*)^T \left( \sum_{t=1}^{k-1} \text{diag} \left( \sum_{x \in D_t} \nabla (-\log P(x \mid W = w_t^*))^2 \right) \right) (w - w_{k-1}^*) + \text{(constant part with 0 grad)}
```

The EWC loss becomes:

```math
J(w) = -\log P(D_k \mid W = w) - \frac{1}{2} (w - w_{k-1}^*)^T \left( \sum_{t=1}^{k-1} \text{diag} \left( \sum_{x \in D_t} \nabla (-\log P(x \mid W = w_t^*))^2 \right) \right) (w - w_{k-1}^*)
```

Adding per-task $\lambda$'s and $\ell_2$ decay, the loss becomes:

```math
J(w) = -\log P(D_k \mid W = w) - \frac{1}{2} \sum_{i=1}^M \left( \ell_2 + \sum_{t=1}^{k-1} \lambda_t \cdot \text{diag} \left( \sum_{x \in D_t} \nabla (-\log P(x \mid W = w_t^*))^2 \right)_{i,i} \right) (w_i - w_{k-1,i}^*)^2
```

[As outlined by Ferenc Huszár](https://www.pnas.org/doi/10.1073/pnas.1717042115), this is different from the loss recommended by the original paper for multiple tasks. In the original paper, a new quadratic penalty is added for each task:

```math
J_{\text{original}}(w) = -\log P(D_k \mid W = w) - \frac{1}{2} \sum_{t=1}^{k-1} \sum_{i=1}^M \left( \ell_2 + \lambda_t \cdot \text{diag} \left( \sum_{x \in D_t} \nabla (-\log P(x \mid W = w_t^*))^2 \right)_{i,i} \right) (w_i - w_{t,i}^*)^2
```

$J(w)$ is more theoretically sound, but $J_{\text{original}}(w)$ is empirically informed. At the same time, $J(w)$ only requires storing the running sum of the Hessian approximations and the previous-task optimum, whereas $J_{\text{original}}(w)$ requires storing the Hessian and the optimum for each task.
