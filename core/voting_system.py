"""
Voting System for AgentSaham.

Aggregates BUY/SELL/HOLD signals from multiple agents using weighted voting
to produce a final consensus signal with confidence level.
"""

from loguru import logger


# Map raw signals to numeric scores
_SIGNAL_SCORES = {
    "BUY": 1.0,
    "HOLD": 0.0,
    "SELL": -1.0,
}

# Map weighted score to final signal thresholds
_FINAL_SIGNAL_THRESHOLDS = [
    (0.6, "STRONG_BUY"),
    (0.2, "BUY"),
    (-0.2, "HOLD"),
    (-0.6, "SELL"),
]


class VotingSystem:
    """
    Weighted voting system that aggregates agent signals.

    Default weights:
    - Technical:    35%
    - Fundamental:  40%
    - Transaction:  25%

    Final signal categories:
    - STRONG_BUY  (weighted score ≥ 0.6)
    - BUY         (0.2 ≤ score < 0.6)
    - HOLD        (-0.2 < score < 0.2)
    - SELL        (-0.6 < score ≤ -0.2)
    - STRONG_SELL (score ≤ -0.6)

    Attributes:
        weight_technical: Weight for the TechnicalAgent vote.
        weight_fundamental: Weight for the FundamentalAgent vote.
        weight_transaction: Weight for the TransactionAgent vote.
    """

    def __init__(
        self,
        weight_technical: float = 0.35,
        weight_fundamental: float = 0.40,
        weight_transaction: float = 0.25,
    ) -> None:
        """
        Initialize VotingSystem with voting weights.

        Args:
            weight_technical: Weight for technical analysis (default 0.35).
            weight_fundamental: Weight for fundamental analysis (default 0.40).
            weight_transaction: Weight for transaction analysis (default 0.25).

        Raises:
            ValueError: If weights do not sum to approximately 1.0.
        """
        total = weight_technical + weight_fundamental + weight_transaction
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Voting weights must sum to 1.0, got {total:.3f}"
            )
        self.weight_technical = weight_technical
        self.weight_fundamental = weight_fundamental
        self.weight_transaction = weight_transaction

    def vote(self, agent_results: dict) -> dict:
        """
        Compute a weighted consensus from all agent results.

        Args:
            agent_results: Dict with keys "technical", "fundamental",
                           "transaction". Each value is an agent result dict
                           containing at least "signal" and "confidence".

        Returns:
            Dict with keys:
                - final_signal: One of STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
                - weighted_score: float -1 to +1
                - confidence: float 0–1
                - reasoning: str summary
                - breakdown: per-agent contribution details
                - has_conflict: bool, True if agents strongly disagree
        """
        breakdown = {}
        weighted_score = 0.0

        weights = {
            "technical": self.weight_technical,
            "fundamental": self.weight_fundamental,
            "transaction": self.weight_transaction,
        }

        raw_signals: list[str] = []

        for agent_name, weight in weights.items():
            result = agent_results.get(agent_name, {})
            signal = result.get("signal", "HOLD").upper()
            confidence = float(result.get("confidence", 0.5))

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

            # Weighted score: direction × confidence × weight
            direction = _SIGNAL_SCORES.get(signal, 0.0)
            contribution = direction * confidence * weight

            weighted_score += contribution
            raw_signals.append(signal)

            breakdown[agent_name] = {
                "signal": signal,
                "confidence": confidence,
                "weight": weight,
                "contribution": round(contribution, 4),
            }

        # Determine final signal
        final_signal = "STRONG_SELL"
        for threshold, label in _FINAL_SIGNAL_THRESHOLDS:
            if weighted_score >= threshold:
                final_signal = label
                break

        # Aggregate confidence (mean of agent confidences weighted)
        total_conf = sum(
            breakdown[a]["confidence"] * breakdown[a]["weight"]
            for a in breakdown
        )

        # Detect conflicts (BUY + SELL among top-weighted agents)
        has_conflict = self._detect_conflict(raw_signals)

        # Build reasoning
        reasoning = self._build_reasoning(breakdown, final_signal, weighted_score, has_conflict)

        result = {
            "final_signal": final_signal,
            "weighted_score": round(weighted_score, 4),
            "confidence": round(total_conf, 2),
            "reasoning": reasoning,
            "breakdown": breakdown,
            "has_conflict": has_conflict,
        }

        logger.info(
            f"[VotingSystem] Result: {final_signal} "
            f"(score={weighted_score:.3f}, conflict={has_conflict})"
        )
        return result

    def _detect_conflict(self, signals: list[str]) -> bool:
        """Return True if BUY and SELL signals coexist among agents."""
        has_buy = any(s in ("BUY",) for s in signals)
        has_sell = any(s in ("SELL",) for s in signals)
        return has_buy and has_sell

    def _build_reasoning(
        self,
        breakdown: dict,
        final_signal: str,
        score: float,
        has_conflict: bool,
    ) -> str:
        """Build a human-readable reasoning string."""
        parts = []
        for agent, data in breakdown.items():
            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(data["signal"], "⚪")
            parts.append(
                f"{emoji} {agent.capitalize()} → {data['signal']} "
                f"(confidence {data['confidence']:.0%}, bobot {data['weight']:.0%})"
            )

        summary = f"Skor tertimbang: {score:+.3f} → Sinyal akhir: {final_signal}"
        if has_conflict:
            summary += " ⚠️ KONFLIK terdeteksi antar agent — pertimbangkan untuk wait & see."

        parts.append(summary)
        return "\n".join(parts)
