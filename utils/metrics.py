# ==================== COMPOSITE EFFICIENCY SCORE ====================
def composite_efficiency_score(P, R, S=0.9, w=[0.5, 0.3, 0.2]):
    # P = performance (AUC), R = resource cost, S = stability (placeholder 0.9)
    return w[0] * P + w[1] * (1 - R) + w[2] * S