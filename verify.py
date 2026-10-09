#!/usr/bin/env python3
"""Verify the numerical examples in README.md; all project data are simulated."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import platform

import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy
from scipy.integrate import quad
from scipy.stats import beta as beta_distribution

FloatArray = NDArray[np.float64]


def probability_vector(value: ArrayLike, name: str) -> FloatArray:
    """Validate a finite, one-dimensional probability distribution."""
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must contain finite, nonnegative values")
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must sum to 1")
    return result


def cross_entropy(p: ArrayLike, q: ArrayLike) -> float:
    """Cross-entropy in nats, preserving the exact zero-probability rules."""
    target = probability_vector(p, "p")
    prediction = probability_vector(q, "q")
    if target.shape != prediction.shape:
        raise ValueError("p and q must have identical shapes")
    support = target > 0
    if np.any(prediction[support] == 0):
        return float("inf")
    return float(-np.dot(target[support], np.log(prediction[support])))


def softmax_cross_entropy(
    logits: ArrayLike, target: ArrayLike
) -> tuple[float, FloatArray, FloatArray]:
    """Return stable cross-entropy, probabilities, and gradient w.r.t. logits."""
    z = np.asarray(logits, dtype=np.float64)
    p = probability_vector(target, "target")
    if z.shape != p.shape or not np.all(np.isfinite(z)):
        raise ValueError("logits must be finite and have the same shape as target")
    shifted = z - z.max()
    log_probabilities = shifted - np.log(np.exp(shifted).sum())
    q = np.exp(log_probabilities)
    loss = float(-np.dot(p, log_probabilities))
    return loss, q, q - p


def binary_cross_entropy_logits(labels: ArrayLike, logits: ArrayLike) -> FloatArray:
    """Per-sample BCE using a stable expression; labels may be hard or soft."""
    y = np.asarray(labels, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    if y.shape != z.shape or y.size == 0:
        raise ValueError("labels and logits must have identical nonempty shapes")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(z)):
        raise ValueError("labels and logits must be finite")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("labels must lie in [0, 1]")
    return np.maximum(z, 0.0) - y * z + np.log1p(np.exp(-np.abs(z)))


@dataclass(frozen=True)
class BetaBernoulli:
    """Beta prior/posterior for the probability of a verified fault."""
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or not np.isfinite(self.beta):
            raise ValueError("alpha and beta must be finite")
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be positive")

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, observations: int, faults: int) -> BetaBernoulli:
        if any(isinstance(v, bool) or not isinstance(v, Integral)
               for v in (observations, faults)):
            raise ValueError("observations and faults must be integers")
        if not 0 <= faults <= observations:
            raise ValueError("require 0 <= faults <= observations")
        return BetaBernoulli(
            self.alpha + faults, self.beta + observations - faults
        )

    def credible_interval(self, level: float = 0.95) -> FloatArray:
        if not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must lie strictly between 0 and 1")
        tail = (1.0 - level) / 2.0
        return beta_distribution.ppf(
            [tail, 1.0 - tail], self.alpha, self.beta
        )


def probability_after_alarm(base_rate: float, tpr: float, fpr: float) -> float:
    """P(fault | alarm), treating sensitivity and false-positive rate as fixed."""
    rates = np.asarray([base_rate, tpr, fpr], dtype=np.float64)
    if not np.all(np.isfinite(rates)) or np.any((rates < 0) | (rates > 1)):
        raise ValueError("all rates must lie in [0, 1]")
    numerator = tpr * base_rate
    denominator = numerator + fpr * (1.0 - base_rate)
    if denominator <= 0:
        raise ValueError("cannot condition on an event with zero probability")
    return numerator / denominator


def verify_svd() -> list[str]:
    A = np.array([[3.0, 1.0], [1.0, 3.0], [1.0, -1.0]])
    U = np.column_stack((
        np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        np.array([1.0, -1.0, 1.0]) / np.sqrt(3),
        np.array([1.0, -1.0, -2.0]) / np.sqrt(6),
    ))
    Sigma = np.array([[4.0, 0.0], [0.0, np.sqrt(6)], [0.0, 0.0]])
    Vt = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    np.testing.assert_allclose(A.T @ A, [[11, 5], [5, 11]], atol=1e-12)
    np.testing.assert_allclose(U.T @ U, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(Vt @ Vt.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(U @ Sigma @ Vt, A, atol=1e-12)
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    np.testing.assert_allclose(s, [4.0, np.sqrt(6)], atol=1e-12)
    np.testing.assert_allclose((u * s) @ vt, A, atol=1e-12)
    rank_one = 4.0 * np.outer(U[:, 0], Vt[0, :])
    np.testing.assert_allclose(rank_one, [[2, 2], [2, 2], [0, 0]], atol=1e-12)
    error = float(np.linalg.norm(A - rank_one, ord="fro"))
    np.testing.assert_allclose(error, np.sqrt(6), atol=1e-12)
    return [
        "## 1. SVD", "",
        f"奇异值：{s[0]:.8f}, {s[1]:.8f}。",
        f"手算分解重构误差：{np.linalg.norm(A - U @ Sigma @ Vt):.3e}。",
        f"秩 1 近似的 Frobenius 误差：{error:.8f}。",
        f"保留的平方 Frobenius 范数比例：{16 / 22:.8%}。", "",
    ]


def verify_bayes() -> list[str]:
    prior = BetaBernoulli(2.0, 98.0)
    posterior = prior.update(observations=100, faults=8)
    ci = posterior.credible_interval()
    np.testing.assert_allclose([posterior.alpha, posterior.beta], [10, 190])
    np.testing.assert_allclose(posterior.mean, 0.05)
    np.testing.assert_allclose(
        beta_distribution.cdf(ci, 10, 190), [0.025, 0.975], atol=1e-12
    )
    risk = probability_after_alarm(posterior.mean, 0.9, 0.05)
    np.testing.assert_allclose(risk, 18 / 37, atol=1e-12)
    density = lambda theta: beta_distribution.pdf(theta, 10, 190)
    joint_probability = quad(lambda theta: 0.9 * theta * density(theta), 0, 1)[0]
    alarm_probability = quad(
        lambda theta: (0.9 * theta + 0.05 * (1 - theta)) * density(theta), 0, 1
    )[0]
    np.testing.assert_allclose(joint_probability / alarm_probability, risk, atol=1e-10)
    online = posterior.update(observations=50, faults=2)
    batch = prior.update(observations=150, faults=10)
    if online != batch:
        raise AssertionError("sequential and batch updates differ")
    map_estimate = (posterior.alpha - 1) / (posterior.alpha + posterior.beta - 2)
    return [
        "## 2. 贝叶斯案例（模拟数据）", "",
        f"先验均值：{prior.mean:.4%}。",
        "后验分布：Beta(10, 190)。",
        f"后验均值 / 无告警信息时的预测概率：{posterior.mean:.4%}。",
        f"95% 等尾可信区间：[{ci[0]:.8f}, {ci[1]:.8f}]。",
        f"MLE：{8 / 100:.4%}；MAP：{map_estimate:.4%}。",
        f"告警后的故障概率：{risk:.8%}。",
        f"复检决策阈值：{100 / 1100:.8%}。",
        f"复检的期望相对损失：{100 * (1 - risk):.8f}。",
        f"不复检的期望相对损失：{1000 * risk:.8f}。",
        f"下一批更新后：Beta({online.alpha:g}, {online.beta:g})，均值 {online.mean:.4%}。", "",
    ]


def verify_cross_entropy() -> list[str]:
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.6, 0.3, 0.1])
    entropy = cross_entropy(p, p)
    ce = cross_entropy(p, q)
    kl = float(np.dot(p, np.log(p / q)))
    np.testing.assert_allclose(ce, entropy + kl, atol=1e-12)
    np.testing.assert_allclose(cross_entropy([1, 0], [1, 0]), 0)
    if not np.isinf(cross_entropy([1, 0], [0, 1])):
        raise AssertionError("positive target mass at zero prediction must give infinity")
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    probabilities = np.array([0.8, 0.3, 0.6, 0.1])
    binary_logits = np.log(probabilities) - np.log1p(-probabilities)
    bce_values = binary_cross_entropy_logits(labels, binary_logits)
    np.testing.assert_allclose(bce_values.mean(), 0.2990011586691898, atol=1e-12)
    np.testing.assert_allclose(
        binary_cross_entropy_logits([1, 0], [-1000, 1000]), [1000, 1000]
    )
    z = np.array([2.0, 1.0, 0.1])
    target = np.array([1.0, 0.0, 0.0])
    loss, softmax, gradient = softmax_cross_entropy(z, target)
    np.testing.assert_allclose(loss, cross_entropy(target, softmax), atol=1e-12)
    epsilon = 1e-6
    numerical_gradient = np.zeros_like(z)
    for k in range(z.size):
        offset = np.zeros_like(z)
        offset[k] = epsilon
        positive = softmax_cross_entropy(z + offset, target)[0]
        negative = softmax_cross_entropy(z - offset, target)[0]
        numerical_gradient[k] = (positive - negative) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numerical_gradient, rtol=1e-6, atol=1e-9)
    next_loss = softmax_cross_entropy(z - 0.1 * gradient, target)[0]
    if next_loss >= loss:
        raise AssertionError("the demonstrated gradient step did not decrease loss")
    stable_loss = softmax_cross_entropy(z + 1000, target)[0]
    np.testing.assert_allclose(stable_loss, loss, atol=1e-12)
    return [
        "## 3. 交叉熵", "",
        f"H(P) = {entropy:.9f} nats。",
        f"H(P,Q) = {ce:.9f} nats。",
        f"KL(P||Q) = {kl:.9f} nats。",
        f"四个样本的平均 BCE：{bce_values.mean():.9f}。",
        f"Softmax 概率：{np.array2string(softmax, precision=8)}。",
        f"多分类交叉熵：{loss:.9f}。",
        f"对 logits 的梯度：{np.array2string(gradient, precision=8)}。",
        f"学习率 0.1、一次梯度更新后的损失：{next_loss:.9f}。", "",
    ]


def main() -> None:
    # Complete all checks before printing the success report.
    sections = verify_svd() + verify_bayes() + verify_cross_entropy()
    lines = [
        "# 数值验证结果", "",
        "> 本文件由 `python scripts/verify.py` 生成。项目数据均为教学模拟数据。", "",
        f"环境：Python {platform.python_version()}，NumPy {np.__version__}，SciPy {scipy.__version__}。", "",
        "全部检查通过：正交性、矩阵重构、后验更新、分位数、概率积分、交叉熵恒等式、梯度和数值稳定性。", "",
    ] + sections
    print("\n".join(lines))


if __name__ == "__main__":
    main()
