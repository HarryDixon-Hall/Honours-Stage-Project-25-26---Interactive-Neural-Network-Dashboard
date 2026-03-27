# Pytest Structure

This test suite is organised around the project report categories so each folder can group related functional-requirement tests.

- `learning_interface/`: page layouts, staged navigation, and learner-facing controls.
- `model_training/`: model construction, parameter updates, and training history.
- `data_handling/`: dataset loading, metadata, and feature preparation.
- `adaptive_learning/`: skill-tree state and Level 3 progression helpers.
- `secure_execution/`: sandbox validation, execution, and editor wiring.
- `deployment_persistence/`: project test setup, packaging, and persistence scaffolding.

Keep tests focused on one requirement at a time so the report can trace each test back to a specific feature claim.