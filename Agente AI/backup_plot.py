def plot_influence_diagram(self, title="Diagrama de influências DEMATEL"):
    """
    Scatter (R+C  vs  R−C) com tamanho proporcional a R+C.
    R + C  → importância/prominence
    R − C  → tipo:
        positivo  → fator mais causador
        negativo → fator mais resultante
    """
    x = self.rc_sum            # Prominence
    y = self.rc_diff           # Net effect
    sizes = (x - x.min()) / (x.max() - x.min() + 1e-9) * 2000 + 300

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=sizes, alpha=0.7, color="#3498db", edgecolors="#2c3e50")

    for i, label in enumerate(self.factors):
        formatted_label = label if len(label) < 10 else "\n".join(
            [label[j:j + 10] for j in range(0, len(label), 10)])  # quebra de linha a cada 10 caracteres
        ax.text(
            x[i], y[i],
            formatted_label,
            ha="center", va="center",
            color="black",
            fontsize=7,  # fonte reduzida para se ajustar à bolinha
            weight="bold",
            clip_on=True,  # garante que o texto não saia da figura
            wrap=True  # permite quebra de linha automática
        )

    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("R + C  (importância)")
    ax.set_ylabel("R − C  (efeito líquido)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    plt.show()