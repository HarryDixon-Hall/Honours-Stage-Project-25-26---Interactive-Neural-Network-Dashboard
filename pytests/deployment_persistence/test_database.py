from sqlalchemy import inspect

from distribution.database import (
    DatabaseConfig,
    create_database_engine,
    create_session_factory,
    get_level2_model_run,
    init_database,
    list_level2_model_runs,
    save_level2_model_run,
)


def test_database_config_prefers_explicit_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///custom.db")
    monkeypatch.delenv("CLOUD_SQL_CONNECTION_NAME", raising=False)
    monkeypatch.delenv("DB_NAME", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)

    config = DatabaseConfig.from_env()

    assert config.source == "DATABASE_URL"
    assert config.database_url == "sqlite+pysqlite:///custom.db"


def test_database_config_builds_cloud_sql_socket_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("CLOUD_SQL_CONNECTION_NAME", "demo-project:europe-west2:dashboard-db")
    monkeypatch.setenv("DB_NAME", "interactive_dashboard")
    monkeypatch.setenv("DB_USER", "dashboard_user")
    monkeypatch.setenv("DB_PASSWORD", "super-secret")

    config = DatabaseConfig.from_env()

    assert config.source == "cloud-sql-socket"
    assert config.database_url.startswith("postgresql+pg8000://dashboard_user:super-secret@/")
    assert "interactive_dashboard" in config.database_url
    assert "unix_sock=%2Fcloudsql%2Fdemo-project%3Aeurope-west2%3Adashboard-db%2F.s.PGSQL.5432" in config.database_url


def test_database_config_defaults_to_local_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("CLOUD_SQL_CONNECTION_NAME", raising=False)
    monkeypatch.delenv("DB_NAME", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)

    config = DatabaseConfig.from_env()

    assert config.source == "local-sqlite"
    assert config.database_url.startswith("sqlite:///")


def test_init_database_creates_expected_tables():
    engine = create_database_engine("sqlite+pysqlite:///:memory:")

    init_database(engine)

    table_names = set(inspect(engine).get_table_names())
    assert {"learner_profiles", "level_progress", "experiment_runs"}.issubset(table_names)


def test_level2_model_runs_are_saved_and_listed_per_user():
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    init_database(engine)
    session_factory = create_session_factory(engine)
    params = {
        "epoch": 4,
        "weights": [[[0.1, 0.2]]],
        "biases": [[[0.0]]],
        "history": {"epochs": [0, 1, 2, 3, 4], "train_loss": [0.9, 0.7, 0.5, 0.4, 0.3]},
        "replay_frames": [
            {"epoch": 0, "weights": [[[0.0, 0.0]]], "biases": [[[0.0]]], "history": {"epochs": [0]}, "meta": {"dataset": "moons", "activation": "tanh"}},
            {"epoch": 4, "weights": [[[0.1, 0.2]]], "biases": [[[0.0]]], "history": {"epochs": [0, 1, 2, 3, 4]}, "meta": {"dataset": "moons", "activation": "tanh"}},
        ],
        "meta": {"dataset": "moons", "activation": "tanh", "hidden_layer_sizes": [6, 6]},
    }
    metrics = {"train_accuracy": 0.95, "test_accuracy": 0.91, "epoch": 4}

    saved_run = save_level2_model_run(
        "learner-1",
        "Moons run",
        params,
        metrics,
        display_name="Learner One",
        session_factory=session_factory,
    )

    saved_runs = list_level2_model_runs("learner-1", session_factory=session_factory)

    assert saved_run["id"] == saved_runs[0]["id"]
    assert saved_runs[0]["model_name"] == "Moons run"
    assert saved_runs[0]["dataset_name"] == "moons"
    assert saved_runs[0]["replay_frame_count"] == 2


def test_level2_model_run_lookup_is_scoped_to_the_owner():
    engine = create_database_engine("sqlite+pysqlite:///:memory:")
    init_database(engine)
    session_factory = create_session_factory(engine)
    params = {
        "epoch": 2,
        "weights": [[[0.1, 0.2]]],
        "biases": [[[0.0]]],
        "history": {"epochs": [0, 1, 2]},
        "replay_frames": [{"epoch": 0, "weights": [[[0.0, 0.0]]], "biases": [[[0.0]]], "history": {"epochs": [0]}, "meta": {"dataset": "linear"}}],
        "meta": {"dataset": "linear", "activation": "relu"},
    }
    metrics = {"train_accuracy": 0.81, "test_accuracy": 0.78, "epoch": 2}

    saved_run = save_level2_model_run(
        "learner-a",
        "Linear run",
        params,
        metrics,
        session_factory=session_factory,
    )

    owner_result = get_level2_model_run("learner-a", saved_run["id"], session_factory=session_factory)
    other_user_result = get_level2_model_run("learner-b", saved_run["id"], session_factory=session_factory)

    assert owner_result is not None
    assert owner_result["model_name"] == "Linear run"
    assert other_user_result is None