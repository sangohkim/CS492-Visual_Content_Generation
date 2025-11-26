import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, lognorm

# timestep range
T = 1000
t = np.linspace(0, 1, 2000)  # PDF는 0~1 구간에서 정의 후 0~1000로 스케일링

# ---- 1) Beta distribution ----
alpha, beta_param = 0.7, 2.0
pdf_beta = beta.pdf(t, alpha, beta_param)

# ---- 2) Power-law distribution (u^k) ----
k = 1.8
pdf_power = k * (t ** (k - 1))  # u^k 변환의 유도된 PDF

# ---- 3) Wide Log-normal ----
mu, sigma = -0.3, 1.6
# lognormal 정의: scipy는 shape parameter = sigma
pdf_lognorm = lognorm.pdf(t + 1e-6, sigma, scale=np.exp(mu))  # 0 문제 방지

# ---- 4) Uniform for comparison ----
pdf_uniform = np.ones_like(t)


# ---- Plotting ----
plt.figure(figsize=(10, 6))
plt.plot(t * T, pdf_beta, label=f"Beta(α={alpha}, β={beta_param})", linewidth=2)
plt.plot(t * T, pdf_power, label=f"Power-law(k={k})", linewidth=2)
plt.plot(t * T, pdf_lognorm, label=f"LogNormal(μ={mu}, σ={sigma})", linewidth=2)
plt.plot(t * T, pdf_uniform, label="Uniform", linestyle="--", alpha=0.6)

plt.title("PDF Comparison for Timesteps Distribution (0 ~ 1000)")
plt.xlabel("timestep (0 ~ 1000)")
plt.ylabel("probability density")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(0, 2)
plt.savefig("timestep_pdf_comparison.png", dpi=300)
