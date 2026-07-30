"""FastAPI application for the Epic 3 workflow."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from email_api.database import create_database, create_schema
from email_api.inference_client import (
    HttpInferenceGateway,
    InferenceError,
    InferenceGateway,
)
from email_api.models import EmailRecordDB
from email_api.schemas import (
    AuditEventView,
    DecisionRequest,
    DraftGenerateRequest,
    DraftView,
    EmailCreate,
    EmailDetail,
    ReviewedValues,
    ReviewQueueItem,
    ReviewQueuePage,
    ReviewUpdate,
)
from email_api.settings import ApiSettings
from email_api.workflow import (
    EmailWorkflow,
    WorkflowConflictError,
    WorkflowNotFoundError,
)
from email_data_engineering.domain.taxonomy import CanonicalLabel, Priority
from email_intelligence.schemas import AnalysisResult


def create_app(
    settings: ApiSettings | None = None,
    inference: InferenceGateway | None = None,
) -> FastAPI:
    resolved_settings = settings or ApiSettings()
    engine, session_factory = create_database(resolved_settings.database_url)
    resolved_inference = inference or HttpInferenceGateway(
        resolved_settings.inference_url,
        resolved_settings.inference_timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if resolved_settings.auto_create_schema:
            await create_schema(engine)
        yield
        await resolved_inference.close()
        await engine.dispose()

    app = FastAPI(title="Email Intelligence API", version="1.0.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.inference = resolved_inference

    async def get_session():
        async with session_factory() as session:
            yield session

    async def get_workflow(
        session: AsyncSession = Depends(get_session),
    ) -> EmailWorkflow:
        return EmailWorkflow(session, resolved_inference)

    @app.get("/health")
    async def health() -> dict[str, object]:
        database_ready = True
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except SQLAlchemyError:
            database_ready = False
        inference_ready = await resolved_inference.health()
        return {
            "status": "ok" if database_ready and inference_ready else "degraded",
            "database": "ok" if database_ready else "unavailable",
            "inference": "ok" if inference_ready else "unavailable",
        }

    @app.post("/v1/emails", response_model=EmailDetail, status_code=status.HTTP_201_CREATED)
    async def create_email(
        payload: EmailCreate,
        workflow: EmailWorkflow = Depends(get_workflow),
    ) -> EmailDetail:
        try:
            email = await workflow.create_and_analyze(payload)
        except IntegrityError as exc:
            await workflow.session.rollback()
            raise HTTPException(status_code=409, detail="external_id already exists") from exc
        except InferenceError as exc:
            raise HTTPException(
                status_code=502,
                detail={"message": str(exc), "retryable": True},
            ) from exc
        return await _detail(workflow, email)

    @app.get("/v1/emails/{email_id}", response_model=EmailDetail)
    async def get_email(
        email_id: str,
        workflow: EmailWorkflow = Depends(get_workflow),
    ) -> EmailDetail:
        try:
            return await _detail(workflow, await workflow.get_email(email_id))
        except WorkflowNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/emails/{email_id}/reanalyze", response_model=EmailDetail)
    async def reanalyze(
        email_id: str,
        workflow: EmailWorkflow = Depends(get_workflow),
    ) -> EmailDetail:
        try:
            email = await workflow.get_email(email_id)
            await workflow.analyze(email)
            return await _detail(workflow, email)
        except WorkflowNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InferenceError as exc:
            raise HTTPException(
                status_code=502,
                detail={"message": str(exc), "email_id": email_id, "retryable": True},
            ) from exc

    @app.post("/v1/emails/{email_id}/draft", response_model=EmailDetail)
    async def generate_draft(
        email_id: str,
        payload: DraftGenerateRequest,
        workflow: EmailWorkflow = Depends(get_workflow),
    ) -> EmailDetail:
        try:
            email = await workflow.get_email(email_id)
            analysis = await workflow.latest_analysis(email_id)
            if analysis is None:
                raise WorkflowConflictError("cannot generate a draft without an analysis")
            result = AnalysisResult.model_validate(analysis.result)
            await workflow.generate_draft(
                email,
                result,
                analysis,
                payload.tone,
                payload.user_context,
            )
            return await _detail(workflow, email)
        except WorkflowNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (WorkflowConflictError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InferenceError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.patch("/v1/emails/{email_id}/review", response_model=EmailDetail)
    async def update_review(
        email_id: str,
        payload: ReviewUpdate,
        workflow: EmailWorkflow = Depends(get_workflow),
    ) -> EmailDetail:
        try:
            email = await workflow.get_email(email_id)
            await workflow.apply_review(email, payload, payload.reviewer_id)
            return await _detail(workflow, email)
        except WorkflowNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except WorkflowConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/emails/{email_id}/decision", response_model=EmailDetail)
    async def decide(
        email_id: str,
        payload: DecisionRequest,
        workflow: EmailWorkflow = Depends(get_workflow),
    ) -> EmailDetail:
        try:
            email = await workflow.get_email(email_id)
            await workflow.decide(email, payload)
            return await _detail(workflow, email)
        except WorkflowNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except WorkflowConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/review-queue", response_model=ReviewQueuePage)
    async def review_queue(
        review_status: str | None = None,
        processing_status: str | None = None,
        classification: CanonicalLabel | None = None,
        priority: Priority | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        session: AsyncSession = Depends(get_session),
    ) -> ReviewQueuePage:
        filters = []
        if review_status:
            filters.append(EmailRecordDB.review_status == review_status)
        if processing_status:
            filters.append(EmailRecordDB.processing_status == processing_status)
        if classification:
            filters.append(EmailRecordDB.current_classification == classification.value)
        if priority:
            filters.append(EmailRecordDB.current_priority == priority.value)
        total = (
            await session.execute(select(func.count(EmailRecordDB.id)).where(*filters))
        ).scalar_one()
        query = (
            select(EmailRecordDB)
            .where(*filters)
            .order_by(
                EmailRecordDB.current_priority.asc(),
                EmailRecordDB.created_at.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        records = list((await session.execute(query)).scalars().all())
        items = []
        for email in records:
            workflow = EmailWorkflow(session, resolved_inference)
            analysis = await workflow.latest_analysis(email.id)
            result = AnalysisResult.model_validate(analysis.result) if analysis else None
            items.append(
                ReviewQueueItem(
                    id=email.id,
                    sender=email.sender,
                    subject=email.subject,
                    processing_status=email.processing_status,
                    review_status=email.review_status,
                    classification=email.current_classification,
                    priority=email.current_priority,
                    recommended_action=email.current_action,
                    summary=result.summary if result else None,
                    classification_confidence=(
                        result.classification_confidence if result else None
                    ),
                    priority_confidence=result.priority_confidence if result else None,
                    created_at=email.created_at,
                )
            )
        return ReviewQueuePage(items=items, total=total, limit=limit, offset=offset)

    return app


async def _detail(workflow: EmailWorkflow, email: EmailRecordDB) -> EmailDetail:
    analysis = await workflow.latest_analysis(email.id)
    draft = await workflow.latest_draft(email.id)
    events = await workflow.audit_events(email.id)
    return EmailDetail(
        id=email.id,
        external_id=email.external_id,
        thread_id=email.thread_id,
        sender=email.sender,
        recipients=email.recipients,
        subject=email.subject,
        body_text=email.body_text,
        received_at=email.received_at,
        processing_status=email.processing_status,
        review_status=email.review_status,
        analysis_error=email.analysis_error,
        draft_error=email.draft_error,
        reviewed=ReviewedValues(
            classification=email.current_classification,
            priority=email.current_priority,
            recommended_action=email.current_action,
        ),
        analysis_version=analysis.version if analysis else None,
        analysis=AnalysisResult.model_validate(analysis.result) if analysis else None,
        draft=(
            DraftView(
                id=draft.id,
                version=draft.version,
                body=draft.body,
                tone=draft.tone,
                source=draft.source,
                model_metadata=draft.model_metadata,
                created_at=draft.created_at,
            )
            if draft
            else None
        ),
        audit_events=[
            AuditEventView(
                id=event.id,
                event_type=event.event_type,
                reviewer_id=event.reviewer_id,
                before_state=event.before_state,
                after_state=event.after_state,
                created_at=event.created_at,
            )
            for event in events
        ],
        created_at=email.created_at,
        updated_at=email.updated_at,
    )


app = create_app()
