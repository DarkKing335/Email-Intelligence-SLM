"""Application service implementing the Epic 3 email workflow."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from email_api.inference_client import InferenceError, InferenceGateway
from email_api.models import (
    EmailAnalysisDB,
    EmailDraftDB,
    EmailRecordDB,
    ReviewAuditEventDB,
)
from email_api.schemas import DecisionRequest, EmailCreate, ReviewChanges
from email_intelligence.schemas import (
    ActionType,
    AnalysisRequest,
    AnalysisResult,
    DraftRequest,
    EmailContent,
)


class WorkflowNotFoundError(LookupError):
    pass


class WorkflowConflictError(RuntimeError):
    pass


class EmailWorkflow:
    def __init__(self, session: AsyncSession, inference: InferenceGateway) -> None:
        self.session = session
        self.inference = inference

    async def create_and_analyze(self, payload: EmailCreate) -> EmailRecordDB:
        email = EmailRecordDB(
            external_id=payload.external_id,
            thread_id=payload.thread_id,
            sender=payload.sender,
            recipients=payload.recipients,
            subject=payload.subject,
            body_text=payload.body_text,
            received_at=payload.received_at,
            processing_status="processing",
        )
        self.session.add(email)
        await self.session.commit()
        await self.session.refresh(email)
        await self.analyze(email, payload.user_context)
        return email

    async def get_email(self, email_id: str) -> EmailRecordDB:
        email = await self.session.get(EmailRecordDB, email_id)
        if email is None:
            raise WorkflowNotFoundError(f"email {email_id} was not found")
        return email

    async def analyze(self, email: EmailRecordDB, user_context=None) -> EmailAnalysisDB:
        email.processing_status = "processing"
        email.analysis_error = None
        await self.session.commit()
        request = AnalysisRequest(email=self._email_content(email), user_context=user_context)
        try:
            result = await self.inference.analyze(request)
        except InferenceError as exc:
            email.processing_status = "analysis_failed"
            email.analysis_error = str(exc)
            self._audit(email.id, "analysis_failed", None, None, {"error": str(exc)})
            await self.session.commit()
            raise

        version = await self._next_analysis_version(email.id)
        analysis = EmailAnalysisDB(
            email_id=email.id,
            version=version,
            result=result.model_dump(mode="json"),
            provider=result.model.provider,
            model_name=result.model.model_name,
            model_version=result.model.model_version,
            classification_confidence=result.classification_confidence,
            priority_confidence=result.priority_confidence,
        )
        self.session.add(analysis)
        await self.session.flush()
        email.processing_status = "analyzed"
        email.review_status = "pending"
        email.current_classification = result.classification.value
        email.current_priority = result.priority.value
        email.current_action = result.recommended_actions[0].action.value
        email.analysis_error = None
        email.draft_error = None
        self._audit(
            email.id,
            "analysis_created",
            None,
            None,
            {"analysis_id": analysis.id, "version": version},
        )
        await self.session.commit()
        await self.session.refresh(analysis)

        if result.response_required and any(
            action.action is ActionType.REPLY for action in result.recommended_actions
        ):
            try:
                await self.generate_draft(email, result, analysis, "professional", user_context)
            except InferenceError as exc:
                email.draft_error = str(exc)
                self._audit(email.id, "draft_failed", None, None, {"error": str(exc)})
                await self.session.commit()
        return analysis

    async def generate_draft(
        self,
        email: EmailRecordDB,
        analysis_result: AnalysisResult,
        analysis: EmailAnalysisDB,
        tone: str,
        user_context=None,
    ) -> EmailDraftDB:
        request = DraftRequest(
            email=self._email_content(email),
            analysis=analysis_result,
            tone=tone,
            user_context=user_context,
        )
        result = await self.inference.generate_draft(request)
        version = await self._next_draft_version(email.id)
        draft = EmailDraftDB(
            email_id=email.id,
            analysis_id=analysis.id,
            version=version,
            body=result.body,
            tone=result.tone,
            source="model",
            model_metadata=result.model.model_dump(mode="json"),
        )
        self.session.add(draft)
        email.draft_error = None
        await self.session.flush()
        self._audit(
            email.id,
            "draft_created",
            None,
            None,
            {"draft_id": draft.id, "version": version},
        )
        await self.session.commit()
        await self.session.refresh(draft)
        return draft

    async def apply_review(
        self,
        email: EmailRecordDB,
        changes: ReviewChanges,
        reviewer_id: str,
        event_type: str = "edited",
    ) -> None:
        before = await self.current_review_state(email)
        if changes.classification is not None:
            email.current_classification = changes.classification.value
        if changes.priority is not None:
            email.current_priority = changes.priority.value
        if changes.recommended_action is not None:
            email.current_action = changes.recommended_action.value
        if changes.draft_body is not None:
            analysis = await self.latest_analysis(email.id)
            if analysis is None:
                raise WorkflowConflictError("cannot edit a draft without an analysis")
            previous = await self.latest_draft(email.id)
            version = await self._next_draft_version(email.id)
            self.session.add(
                EmailDraftDB(
                    email_id=email.id,
                    analysis_id=analysis.id,
                    version=version,
                    body=changes.draft_body,
                    tone=previous.tone if previous else "professional",
                    source="user",
                    model_metadata={},
                )
            )
        await self.session.flush()
        after = await self.current_review_state(email)
        self._audit(email.id, event_type, reviewer_id, before, after)
        await self.session.commit()

    async def decide(self, email: EmailRecordDB, request: DecisionRequest) -> None:
        if email.processing_status != "analyzed":
            raise WorkflowConflictError("only analyzed emails can be reviewed")
        before = await self.current_review_state(email)
        if request.changes:
            await self.apply_review(
                email,
                request.changes,
                request.reviewer_id,
                event_type="decision_edit",
            )
        email.review_status = "approved" if request.decision == "approve" else "rejected"
        await self.session.flush()
        after = await self.current_review_state(email)
        after["review_status"] = email.review_status
        self._audit(email.id, email.review_status, request.reviewer_id, before, after)
        await self.session.commit()

    async def latest_analysis(self, email_id: str) -> EmailAnalysisDB | None:
        query = (
            select(EmailAnalysisDB)
            .where(EmailAnalysisDB.email_id == email_id)
            .order_by(EmailAnalysisDB.version.desc())
            .limit(1)
        )
        return (await self.session.execute(query)).scalar_one_or_none()

    async def latest_draft(self, email_id: str) -> EmailDraftDB | None:
        query = (
            select(EmailDraftDB)
            .where(EmailDraftDB.email_id == email_id)
            .order_by(EmailDraftDB.version.desc())
            .limit(1)
        )
        return (await self.session.execute(query)).scalar_one_or_none()

    async def audit_events(self, email_id: str) -> list[ReviewAuditEventDB]:
        query = (
            select(ReviewAuditEventDB)
            .where(ReviewAuditEventDB.email_id == email_id)
            .order_by(ReviewAuditEventDB.created_at.asc())
        )
        return list((await self.session.execute(query)).scalars().all())

    async def current_review_state(self, email: EmailRecordDB) -> dict:
        draft = await self.latest_draft(email.id)
        return {
            "classification": email.current_classification,
            "priority": email.current_priority,
            "recommended_action": email.current_action,
            "draft_body": draft.body if draft else None,
            "review_status": email.review_status,
        }

    async def _next_analysis_version(self, email_id: str) -> int:
        query = select(func.max(EmailAnalysisDB.version)).where(
            EmailAnalysisDB.email_id == email_id
        )
        return ((await self.session.execute(query)).scalar_one() or 0) + 1

    async def _next_draft_version(self, email_id: str) -> int:
        query = select(func.max(EmailDraftDB.version)).where(
            EmailDraftDB.email_id == email_id
        )
        return ((await self.session.execute(query)).scalar_one() or 0) + 1

    @staticmethod
    def _email_content(email: EmailRecordDB) -> EmailContent:
        return EmailContent(
            email_id=email.id,
            external_id=email.external_id,
            thread_id=email.thread_id,
            sender=email.sender,
            recipients=email.recipients,
            subject=email.subject,
            body_text=email.body_text,
            received_at=email.received_at,
        )

    def _audit(
        self,
        email_id: str,
        event_type: str,
        reviewer_id: str | None,
        before: dict | None,
        after: dict | None,
    ) -> None:
        self.session.add(
            ReviewAuditEventDB(
                email_id=email_id,
                event_type=event_type,
                reviewer_id=reviewer_id,
                before_state=before,
                after_state=after,
            )
        )
