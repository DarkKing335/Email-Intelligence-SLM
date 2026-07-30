"""Deterministic provider used until a fine-tuned model is available."""

from __future__ import annotations

import re

from email_data_engineering.domain.taxonomy import CanonicalLabel, Priority
from email_intelligence.schemas import (
    ActionType,
    AnalysisRequest,
    AnalysisResult,
    DraftRequest,
    DraftResult,
    ExtractedEntities,
    LinkEntity,
    ModelMetadata,
    MoneyEntity,
    RecommendedAction,
    TextEntity,
)


class MockModelProvider:
    def __init__(
        self,
        model_name: str = "epic3-mock",
        model_version: str = "development",
        summary_max_chars: int = 320,
    ) -> None:
        self._metadata = ModelMetadata(
            provider="mock",
            model_name=model_name,
            model_version=model_version,
        )
        self._summary_max_chars = summary_max_chars

    @property
    def ready(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        email = request.email
        text = f"{email.subject}\n{email.body_text}"
        lowered = text.lower()
        classification = self._classify(lowered)
        priority = self._priority(classification, lowered)
        response_required = classification in {
            CanonicalLabel.RESPOND,
            CanonicalLabel.SUPPORT,
        } or any(word in lowered for word in ("please reply", "let me know", "can you"))
        action = self._action(classification, response_required)
        summary = " ".join(email.body_text.split())
        if len(summary) > self._summary_max_chars:
            summary = summary[: self._summary_max_chars - 1].rstrip() + "…"

        return AnalysisResult(
            classification=classification,
            classification_confidence=0.86,
            priority=priority,
            priority_confidence=0.82,
            summary=summary,
            entities=self._extract_entities(text),
            response_required=response_required,
            recommended_actions=[
                RecommendedAction(
                    action=action,
                    confidence=0.84,
                    reason=self._action_reason(action),
                )
            ],
            model=self._metadata,
        )

    async def generate_draft(self, request: DraftRequest) -> DraftResult:
        sender_name = request.email.sender.split("<", maxsplit=1)[0].strip() or "there"
        body = (
            f"Hello {sender_name},\n\n"
            "Thank you for your email. I have received your request and will review "
            "the details. I will follow up as soon as possible.\n\n"
            "Best regards"
        )
        return DraftResult(body=body, tone=request.tone, model=self._metadata)

    @staticmethod
    def _classify(text: str) -> CanonicalLabel:
        rules = (
            (CanonicalLabel.SPAM, ("lottery", "winner", "crypto giveaway", "scam")),
            (CanonicalLabel.SECURITY, ("otp", "password reset", "security alert", "verify")),
            (CanonicalLabel.BILLING, ("invoice", "payment", "receipt", "billing")),
            (CanonicalLabel.SUPPORT, ("support", "ticket", "help desk")),
            (CanonicalLabel.UNSUBSCRIBE, ("unsubscribe", "newsletter", "promotion", "sale")),
            (CanonicalLabel.ARCHIVE, ("automated notification", "no-reply", "noreply")),
            (CanonicalLabel.RESPOND, ("please reply", "let me know", "can you", "request")),
        )
        for label, keywords in rules:
            if any(keyword in text for keyword in keywords):
                return label
        return CanonicalLabel.FYI

    @staticmethod
    def _priority(classification: CanonicalLabel, text: str) -> Priority:
        if classification in {CanonicalLabel.SECURITY, CanonicalLabel.BILLING} or any(
            word in text for word in ("urgent", "immediately", "deadline", "overdue")
        ):
            return Priority.HIGH
        if classification in {CanonicalLabel.FYI, CanonicalLabel.SUPPORT}:
            return Priority.MEDIUM
        return Priority.LOW

    @staticmethod
    def _action(
        classification: CanonicalLabel,
        response_required: bool,
    ) -> ActionType:
        if response_required:
            return ActionType.REPLY
        if classification is CanonicalLabel.UNSUBSCRIBE:
            return ActionType.UNSUBSCRIBE
        if classification is CanonicalLabel.SPAM:
            return ActionType.IGNORE
        if classification is CanonicalLabel.ARCHIVE:
            return ActionType.ARCHIVE
        return ActionType.FLAG_FOR_REVIEW

    @staticmethod
    def _action_reason(action: ActionType) -> str:
        return {
            ActionType.REPLY: "The email contains a request that appears to need a response.",
            ActionType.UNSUBSCRIBE: "The message appears to be promotional.",
            ActionType.IGNORE: "The message appears to be unsolicited or junk.",
            ActionType.ARCHIVE: "The message appears informational or automated.",
            ActionType.FLAG_FOR_REVIEW: "No consequential action should be taken automatically.",
        }[action]

    @staticmethod
    def _extract_entities(text: str) -> ExtractedEntities:
        links = [
            LinkEntity(text=url.rstrip(".,)"), url=url.rstrip(".,)"))
            for url in re.findall(r"https?://[^\s]+", text)
        ]
        amounts: list[MoneyEntity] = []
        for match in re.finditer(r"(?P<symbol>[$€£])\s?(?P<amount>\d[\d,]*(?:\.\d{2})?)", text):
            currency = {"$": "USD", "€": "EUR", "£": "GBP"}[match.group("symbol")]
            amounts.append(
                MoneyEntity(
                    text=match.group(0),
                    amount=float(match.group("amount").replace(",", "")),
                    currency=currency,
                )
            )
        dates = [
            TextEntity(text=value)
            for value in re.findall(
                r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
                r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
                r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s+\d{4})?\b",
                text,
                flags=re.IGNORECASE,
            )
        ]
        references = [
            TextEntity(text=value)
            for value in re.findall(r"\b(?:INV|REF|TICKET)[-#: ]*\d[\w-]*\b", text, re.I)
        ]
        return ExtractedEntities(
            dates=dates,
            links=links,
            monetary_amounts=amounts,
            references=references,
        )
