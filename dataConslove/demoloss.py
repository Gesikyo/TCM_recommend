import matplotlib.pyplot as plt
import numpy as np

# 模拟epoch
epochs = np.arange(0, 1000, 10)

# 生成拟真loss曲线：
# 前200轮快速下降，中间平缓，后期贴底
loss_values = []
for e in epochs:
    if e < 200:
        loss = 1.0 * np.exp(-e / 100) + np.random.uniform(-0.02, 0.02)
    elif e < 600:
        loss = 0.15 + np.random.uniform(-0.01, 0.01)
    else:
        loss = 0.03 + np.random.uniform(-0.005, 0.005)
    loss_values.append(max(loss, 0.001))  # 防止负数

# 绘制损失曲线
plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_values, color='darkorange', linewidth=2)
plt.title('GCN', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
