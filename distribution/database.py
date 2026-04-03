#DECLARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/ 04/12/25 - 22/03/26
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, JSON, String, Text, create_engine, desc, select
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


LOCAL_SQLITE_PATH = Path(__file__).resolve().parents[1] / "interactive_dashboard.db"


def utc_now() -> datetime:
	return datetime.now(UTC)


class Base(DeclarativeBase):
	pass


class LearnerProfile(Base):
	__tablename__ = "learner_profiles"

	learner_id: Mapped[str] = mapped_column(String(128), primary_key=True)
	display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
	created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
	updated_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True),
		default=utc_now,
		onupdate=utc_now,
	)

	progress_entries: Mapped[list["LevelProgress"]] = relationship(back_populates="learner")
	experiment_runs: Mapped[list["ExperimentRun"]] = relationship(back_populates="learner")


class LevelProgress(Base):
	__tablename__ = "level_progress"

	id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
	learner_id: Mapped[str] = mapped_column(ForeignKey("learner_profiles.learner_id"), index=True)
	level_key: Mapped[str] = mapped_column(String(64), index=True)
	status: Mapped[str] = mapped_column(String(32), default="not_started")
	score: Mapped[Optional[float]] = mapped_column(nullable=True)
	completed: Mapped[bool] = mapped_column(Boolean, default=False)
	completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
	details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

	learner: Mapped[LearnerProfile] = relationship(back_populates="progress_entries")


class ExperimentRun(Base):
	__tablename__ = "experiment_runs"

	id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
	learner_id: Mapped[str] = mapped_column(ForeignKey("learner_profiles.learner_id"), index=True)
	level_key: Mapped[str] = mapped_column(String(64), index=True)
	dataset_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
	model_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
	parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
	metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
	notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
	created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

	learner: Mapped[LearnerProfile] = relationship(back_populates="experiment_runs")


@dataclass(frozen=True)
class DatabaseConfig:
	database_url: str
	source: str

	@classmethod
	def from_env(cls) -> "DatabaseConfig":
		explicit_url = os.getenv("DATABASE_URL")
		if explicit_url:
			return cls(database_url=explicit_url, source="DATABASE_URL")

		cloud_sql_connection_name = os.getenv("CLOUD_SQL_CONNECTION_NAME")
		db_name = os.getenv("DB_NAME")
		db_user = os.getenv("DB_USER")
		db_password = os.getenv("DB_PASSWORD")

		if cloud_sql_connection_name and db_name and db_user and db_password:
			socket_path = f"/cloudsql/{cloud_sql_connection_name}/.s.PGSQL.5432"
			cloud_sql_url = URL.create(
				drivername="postgresql+pg8000",
				username=db_user,
				password=db_password,
				database=db_name,
				query={"unix_sock": socket_path},
			)
			return cls(
				database_url=cloud_sql_url.render_as_string(hide_password=False),
				source="cloud-sql-socket",
			)

		sqlite_url = URL.create(drivername="sqlite", database=str(LOCAL_SQLITE_PATH))
		return cls(
			database_url=sqlite_url.render_as_string(hide_password=False),
			source="local-sqlite",
		)


def create_database_engine(database_url: Optional[str] = None, *, echo: bool = False) -> Engine:
	resolved_url = database_url or DatabaseConfig.from_env().database_url
	connect_args = {"check_same_thread": False} if resolved_url.startswith("sqlite") else {}
	return create_engine(resolved_url, echo=echo, future=True, connect_args=connect_args)


def create_session_factory(engine: Optional[Engine] = None) -> sessionmaker[Session]:
	bound_engine = engine or create_database_engine()
	return sessionmaker(bind=bound_engine, autoflush=False, autocommit=False, future=True)


@lru_cache(maxsize=1)
def get_database_engine() -> Engine:
	return create_database_engine()


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
	return create_session_factory(get_database_engine())


@contextmanager
def session_scope(session_factory: Optional[sessionmaker[Session]] = None) -> Iterator[Session]:
	factory = session_factory or get_session_factory()
	session = factory()
	try:
		yield session
		session.commit()
	except Exception:
		session.rollback()
		raise
	finally:
		session.close()


def _normalize_user_display_name(learner_id: str, display_name: Optional[str]) -> str:
	if display_name:
		return display_name
	return f"Learner {learner_id.split('-')[0][:8]}"


def _ensure_learner_profile(session: Session, learner_id: str, display_name: Optional[str] = None) -> LearnerProfile:
	learner = session.get(LearnerProfile, learner_id)
	resolved_display_name = _normalize_user_display_name(learner_id, display_name)

	if learner is None:
		learner = LearnerProfile(
			learner_id=learner_id,
			display_name=resolved_display_name,
		)
		session.add(learner)
	else:
		learner.display_name = resolved_display_name

	return learner


def _make_level2_run_payload(params: dict[str, Any]) -> dict[str, Any]:
	return {
		"record_type": "level2-model-snapshot",
		"params": params,
	}


def _serialize_level2_run(experiment_run: ExperimentRun) -> dict[str, Any]:
	parameters = experiment_run.parameters or {}
	params = parameters.get("params", {})
	meta = params.get("meta", {})
	return {
		"id": experiment_run.id,
		"learner_id": experiment_run.learner_id,
		"level_key": experiment_run.level_key,
		"dataset_name": experiment_run.dataset_name,
		"model_name": experiment_run.model_name,
		"metrics": experiment_run.metrics or {},
		"params": params,
		"created_at": experiment_run.created_at.isoformat() if experiment_run.created_at else None,
		"display_name": experiment_run.learner.display_name if experiment_run.learner else None,
		"epoch": int(params.get("epoch", 0)),
		"history_length": len((params.get("history") or {}).get("epochs", [])),
		"replay_frame_count": len(params.get("replay_frames", [])),
		"activation": meta.get("activation"),
		"hidden_layer_sizes": meta.get("hidden_layer_sizes", []),
	}


def save_level2_model_run(
	learner_id: str,
	model_name: str,
	params: dict[str, Any],
	metrics: dict[str, Any],
	*,
	display_name: Optional[str] = None,
	session_factory: Optional[sessionmaker[Session]] = None,
) -> dict[str, Any]:
	with session_scope(session_factory) as session:
		learner = _ensure_learner_profile(session, learner_id, display_name=display_name)
		meta = params.get("meta", {})
		experiment_run = ExperimentRun(
			learner_id=learner.learner_id,
			level_key="level2",
			dataset_name=meta.get("dataset"),
			model_name=model_name,
			parameters=_make_level2_run_payload(params),
			metrics=metrics,
		)
		session.add(experiment_run)
		session.flush()
		session.refresh(experiment_run)
		return _serialize_level2_run(experiment_run)


def list_level2_model_runs(
	learner_id: str,
	*,
	limit: int = 12,
	session_factory: Optional[sessionmaker[Session]] = None,
) -> list[dict[str, Any]]:
	with session_scope(session_factory) as session:
		statement = (
			select(ExperimentRun)
			.where(ExperimentRun.learner_id == learner_id, ExperimentRun.level_key == "level2")
			.order_by(desc(ExperimentRun.created_at), desc(ExperimentRun.id))
			.limit(limit)
		)
		runs = session.execute(statement).scalars().all()
		return [_serialize_level2_run(experiment_run) for experiment_run in runs]


def get_level2_model_run(
	learner_id: str,
	run_id: int,
	*,
	session_factory: Optional[sessionmaker[Session]] = None,
) -> Optional[dict[str, Any]]:
	with session_scope(session_factory) as session:
		statement = select(ExperimentRun).where(
			ExperimentRun.id == int(run_id),
			ExperimentRun.learner_id == learner_id,
			ExperimentRun.level_key == "level2",
		)
		experiment_run = session.execute(statement).scalar_one_or_none()
		if experiment_run is None:
			return None
		return _serialize_level2_run(experiment_run)


def init_database(engine: Optional[Engine] = None) -> Engine:
	bound_engine = engine or get_database_engine()
	Base.metadata.create_all(bound_engine)
	return bound_engine


__all__ = [
	"Base",
	"DatabaseConfig",
	"ExperimentRun",
	"LearnerProfile",
	"LevelProgress",
	"LOCAL_SQLITE_PATH",
	"create_database_engine",
	"create_session_factory",
	"get_database_engine",
	"get_level2_model_run",
	"get_session_factory",
	"init_database",
	"list_level2_model_runs",
	"save_level2_model_run",
	"session_scope",
]