"""
Orchestrator Agent for AgentSaham.

Coordinates all specialized agents (Technical, Fundamental, Transaction),
runs them in parallel using asyncio, collects their results, feeds them
into the voting system, and generates the final analysis report.
"""

import asyncio
from typing import Optional

from loguru import logger

from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent
from agents.transaction_agent import TransactionAgent
from core.voting_system import VotingSystem
from core.report_generator import ReportGenerator
from config.settings import settings


class Orchestrator:
    """
    Main coordinator that runs all analysis agents in parallel.

    Workflow:
        1. Optionally transcribe video input.
        2. Run TechnicalAgent, FundamentalAgent, TransactionAgent concurrently.
        3. Collect results and pass to VotingSystem.
        4. Generate final report via ReportGenerator.

    Attributes:
        technical_agent: TechnicalAgent instance.
        fundamental_agent: FundamentalAgent instance.
        transaction_agent: TransactionAgent instance.
        voting_system: VotingSystem instance.
        report_generator: ReportGenerator instance.
    """

    def __init__(
        self,
        technical_agent: Optional[TechnicalAgent] = None,
        fundamental_agent: Optional[FundamentalAgent] = None,
        transaction_agent: Optional[TransactionAgent] = None,
        voting_system: Optional[VotingSystem] = None,
        report_generator: Optional[ReportGenerator] = None,
    ) -> None:
        """
        Initialize Orchestrator with optional agent overrides.

        Args:
            technical_agent: Override for TechnicalAgent.
            fundamental_agent: Override for FundamentalAgent.
            transaction_agent: Override for TransactionAgent.
            voting_system: Override for VotingSystem.
            report_generator: Override for ReportGenerator.
        """
        self.technical_agent = technical_agent or TechnicalAgent()
        self.fundamental_agent = fundamental_agent or FundamentalAgent(
            gemini_api_key=settings.gemini_api_key
        )
        self.transaction_agent = transaction_agent or TransactionAgent()
        self.voting_system = voting_system or VotingSystem(
            weight_technical=settings.weight_technical,
            weight_fundamental=settings.weight_fundamental,
            weight_transaction=settings.weight_transaction,
        )
        self.report_generator = report_generator or ReportGenerator()

    async def analyze(
        self,
        ticker: str,
        transcript: Optional[str] = None,
    ) -> dict:
        """
        Run full multi-agent analysis for a stock ticker.

        Agents run concurrently. If any individual agent fails, its error is
        logged and a neutral HOLD result is substituted so the overall
        analysis can still complete.

        Args:
            ticker: Stock ticker symbol.
            transcript: Optional video transcript text.

        Returns:
            Dict with:
                - ticker: str
                - final_signal: STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
                - confidence: float 0–1
                - reasoning: str
                - agent_results: dict of individual agent results
                - voting_details: raw voting breakdown
                - report: str markdown report
        """
        logger.info(f"[Orchestrator] Starting analysis for {ticker}")

        # Run all agents concurrently
        tech_result, fund_result, trans_result = await asyncio.gather(
            self._safe_run(self.technical_agent.analyze(ticker, transcript), "technical"),
            self._safe_run(self.fundamental_agent.analyze(ticker, transcript), "fundamental"),
            self._safe_run(self.transaction_agent.analyze(ticker), "transaction"),
        )

        agent_results = {
            "technical": tech_result,
            "fundamental": fund_result,
            "transaction": trans_result,
        }

        # Voting
        voting_result = self.voting_system.vote(agent_results)

        # Report
        report = self.report_generator.generate(ticker, agent_results, voting_result)

        final = {
            "ticker": ticker,
            "final_signal": voting_result["final_signal"],
            "confidence": voting_result["confidence"],
            "reasoning": voting_result["reasoning"],
            "agent_results": agent_results,
            "voting_details": voting_result,
            "report": report,
        }

        logger.info(
            f"[Orchestrator] {ticker} analysis complete: "
            f"signal={final['final_signal']}, confidence={final['confidence']:.2f}"
        )
        return final

    async def _safe_run(self, coro, agent_name: str) -> dict:
        """
        Run a coroutine safely, returning a neutral HOLD result on failure.

        Args:
            coro: Coroutine to execute.
            agent_name: Name used for logging and fallback result.

        Returns:
            Agent result dict, or neutral fallback on exception.
        """
        try:
            return await coro
        except Exception as exc:
            logger.error(f"[Orchestrator] Agent '{agent_name}' failed: {exc}", exc_info=True)
            return {
                "agent": agent_name,
                "signal": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Agent gagal: {exc}",
                "error": str(exc),
            }
