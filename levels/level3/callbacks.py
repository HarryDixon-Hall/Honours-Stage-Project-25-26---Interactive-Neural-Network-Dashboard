from importlib import import_module

import dash

try:
    from dash import Input, Output, State, html
except ImportError:
    dash_dependencies = import_module('dash.dependencies')
    Input = dash_dependencies.Input
    Output = dash_dependencies.Output
    State = dash_dependencies.State
    from dash import html

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from levels.level2 import (
    level2_evaluate_metrics,
    level2_forward_pass,
    make_decision_boundary_figure,
    make_level2_summary_panel,
    make_level2_training_curves_figure,
    make_network_diagram_figure,
    train_level2_model,
)
from levels.level3.methods import (
    level3_activation_heatmap_figure,
    level3_build_meta,
    level3_confusion_matrix_figure,
    level3_dataset_preview_figure,
    level3_dataset_summary_children,
    level3_deserialize_split,
    level3_forward_summary_children,
    level3_hidden_space_figure,
    level3_initialize_model,
    level3_initialize_store,
    level3_metrics_summary_children,
    level3_misclassified_figure,
    level3_model_matches,
    level3_notebook_status_children,
    level3_training_log_children,
    make_level3_placeholder_figure,
)


def register_level3_callbacks(app):
    @app.callback(
        Output('level3-params-store', 'data'),
        Input('level3-load-data-btn', 'n_clicks'),
        Input('level3-define-model-btn', 'n_clicks'),
        Input('level3-forward-btn', 'n_clicks'),
        Input('level3-train-btn', 'n_clicks'),
        Input('level3-inspect-btn', 'n_clicks'),
        Input('level3-evaluate-btn', 'n_clicks'),
        State('level3-dataset-dropdown', 'value'),
        State('level3-depth-slider', 'value'),
        State('level3-width-slider', 'value'),
        State('level3-activation-dropdown', 'value'),
        State('level3-epochs-slider', 'value'),
        State('level3-params-store', 'data'),
    )
    def update_level3_params(
        n_load,
        n_define,
        n_forward,
        n_train,
        n_inspect,
        n_evaluate,
        dataset,
        depth,
        width,
        activation,
        epochs,
        store,
    ):
        _ = (n_load, n_define, n_forward, n_train, n_inspect, n_evaluate)
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        meta = level3_build_meta(dataset, depth, width, activation, epochs)

        if store is None or store.get('meta', {}).get('dataset') != dataset:
            store = level3_initialize_store(meta)
        else:
            store['meta'] = meta

        if trigger in (None, 'level3-load-data-btn'):
            store = level3_initialize_store(meta)
            store['cell_runs']['load_dataset'] = True
            return store

        store['cell_runs']['load_dataset'] = True

        if trigger == 'level3-define-model-btn':
            store = level3_initialize_model(store, meta)
            store['cell_runs']['define_model'] = True
            return store

        if store.get('model') is None or not level3_model_matches(store, meta):
            store = level3_initialize_model(store, meta)
        store['cell_runs']['define_model'] = True

        X_train, X_test, y_train, y_test, _, _ = level3_deserialize_split(store['data'])

        if trigger == 'level3-forward-btn':
            batch_X = X_train[:24]
            batch_y = y_train[:24]
            predictions, cache = level2_forward_pass(batch_X, store['model'], meta['activation'])
            hidden_shapes = [list(activation_matrix.shape) for activation_matrix in cache['activations'][1:-1]]
            store['forward_summary'] = {
                'batch_shape': list(batch_X.shape),
                'hidden_shapes': hidden_shapes,
                'output_shape': list(predictions.shape),
                'examples': [
                    {
                        'x1': batch_X[index, 0],
                        'x2': batch_X[index, 1],
                        'target': int(batch_y[index]),
                        'probability': float(predictions[index, 0]),
                    }
                    for index in range(min(5, batch_X.shape[0]))
                ],
            }
            store['cell_runs']['forward_pass'] = True

        elif trigger == 'level3-train-btn':
            store['model'] = train_level2_model(
                X_train,
                y_train,
                store['model'],
                activation=meta['activation'],
                epochs=meta['epochs'],
                lr=0.08,
                l2=1e-4,
            )
            train_metrics = level2_evaluate_metrics(X_train, y_train, store['model'], meta['activation'], l2=1e-4)
            test_metrics = level2_evaluate_metrics(X_test, y_test, store['model'], meta['activation'], l2=1e-4)
            store['training_logs'].append({
                'run_number': len(store['training_logs']) + 1,
                'epochs': meta['epochs'],
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'test_loss': test_metrics['loss'],
                'test_accuracy': test_metrics['accuracy'],
            })
            store['cell_runs']['train_model'] = True

        elif trigger == 'level3-inspect-btn':
            store['inspect_ran'] = True
            store['cell_runs']['inspect'] = True

        elif trigger == 'level3-evaluate-btn':
            predictions, _ = level2_forward_pass(X_test, store['model'], meta['activation'])
            pred_labels = (predictions.flatten() >= 0.5).astype(np.int32)
            confusion_values = confusion_matrix(y_test, pred_labels, labels=[0, 1])
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test,
                pred_labels,
                labels=[0, 1],
                zero_division=0,
            )
            metrics = level2_evaluate_metrics(X_test, y_test, store['model'], meta['activation'], l2=1e-4)
            store['evaluation'] = {
                'metrics': metrics,
                'confusion_matrix': confusion_values.tolist(),
                'pred_labels': pred_labels.tolist(),
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'f1': f1.tolist(),
                'support': support.tolist(),
                'misclassified_count': int(np.sum(pred_labels != y_test)),
                'sample_count': int(y_test.shape[0]),
            }
            store['cell_runs']['evaluate'] = True

        return store

    @app.callback(
        Output('level3-boundary-graph', 'figure'),
        Output('level3-loss-graph', 'figure'),
        Output('level3-activations-graph', 'figure'),
        Output('level3-dataset-preview-graph', 'figure'),
        Output('level3-dataset-summary', 'children'),
        Output('level3-network-diagram-graph', 'figure'),
        Output('level3-arch-summary', 'children'),
        Output('level3-forward-output', 'children'),
        Output('level3-training-log', 'children'),
        Output('level3-hidden-space-graph', 'figure'),
        Output('level3-confusion-matrix-graph', 'figure'),
        Output('level3-misclassified-graph', 'figure'),
        Output('level3-metrics-summary', 'children'),
        Output('level3-notebook-status', 'children'),
        Input('level3-params-store', 'data'),
    )
    def update_level3_views(store):
        boundary_placeholder = make_level3_placeholder_figure(
            'Decision boundary / prediction surface',
            'Run Cell 2 to define a classifier and render its decision surface.',
        )
        loss_placeholder = make_level3_placeholder_figure(
            'Loss curve',
            'Run Cell 4 to train the model and record optimisation history.',
        )
        activation_placeholder = make_level3_placeholder_figure(
            'Per-layer activations',
            'Run Cell 5 to inspect hidden activations after defining the model.',
        )
        dataset_placeholder = make_level3_placeholder_figure(
            'Dataset preview',
            'Run Cell 1 to load a toy classification dataset.',
        )
        model_placeholder = make_level3_placeholder_figure(
            'Architecture diagram',
            'Run Cell 2 to define the hidden stack and output layer.',
        )
        hidden_placeholder = make_level3_placeholder_figure(
            'Hidden-space projection',
            'Run Cell 5 to inspect hidden representations.',
        )
        eval_placeholder = make_level3_placeholder_figure(
            'Evaluation',
            'Run Cell 6 to compute confusion and evaluation diagnostics.',
        )

        if store is None:
            return (
                boundary_placeholder,
                loss_placeholder,
                activation_placeholder,
                dataset_placeholder,
                html.Div('Run Cell 1 to create the dataset preview.', style={'color': '#64748b'}),
                model_placeholder,
                html.Div('Run Cell 2 to generate the architecture summary.', style={'color': '#64748b'}),
                level3_forward_summary_children(None),
                level3_training_log_children([]),
                hidden_placeholder,
                eval_placeholder,
                eval_placeholder,
                level3_metrics_summary_children(None),
                level3_notebook_status_children(None),
            )

        meta = store['meta']
        X_train, X_test, y_train, y_test, X_full, y_full = level3_deserialize_split(store['data'])
        dataset_preview = level3_dataset_preview_figure(X_train, X_test, y_train, y_test, meta['dataset'])
        dataset_summary = level3_dataset_summary_children(X_train, X_test, y_train, y_test, meta)
        notebook_status = level3_notebook_status_children(store)

        boundary_fig = boundary_placeholder
        loss_fig = loss_placeholder
        activations_fig = activation_placeholder
        network_fig = model_placeholder
        arch_summary = html.Div('Run Cell 2 to generate the architecture summary.', style={'color': '#64748b'})
        hidden_space_fig = hidden_placeholder
        confusion_fig = eval_placeholder
        misclassified_fig = eval_placeholder

        model = store.get('model')
        if model is not None:
            boundary_fig = make_decision_boundary_figure(X_full, y_full, model, meta['activation'])
            if store['cell_runs']['train_model']:
                boundary_fig.update_layout(title=f"Trained decision boundary after {model.get('epoch', 0)} epochs")
            else:
                boundary_fig.update_layout(title='Initial decision surface from the current model definition')

            history = model.get('history', {})
            if history.get('loss'):
                loss_fig = make_level2_training_curves_figure(history)
                loss_fig.update_layout(title='Training loss and accuracy from Cell 4')

            network_fig = make_network_diagram_figure(
                input_dim=2,
                hidden_layers=meta['hidden_layer_sizes'],
                output_dim=1,
                params=model,
                activation=meta['activation'],
            )
            arch_summary = make_level2_summary_panel(model, meta['activation'])

            if store['inspect_ran']:
                activations_fig = level3_activation_heatmap_figure(model, X_test, meta['activation'])
                hidden_space_fig = level3_hidden_space_figure(model, X_test, y_test, meta['activation'])

        forward_output = level3_forward_summary_children(store.get('forward_summary'))
        training_log = level3_training_log_children(store.get('training_logs', []))

        evaluation = store.get('evaluation')
        if evaluation is not None:
            confusion_fig = level3_confusion_matrix_figure(evaluation['confusion_matrix'])
            pred_labels = np.array(evaluation['pred_labels'], dtype=np.int32)
            misclassified_fig = level3_misclassified_figure(X_test, y_test, pred_labels)
        metrics_summary = level3_metrics_summary_children(evaluation)

        return (
            boundary_fig,
            loss_fig,
            activations_fig,
            dataset_preview,
            dataset_summary,
            network_fig,
            arch_summary,
            forward_output,
            training_log,
            hidden_space_fig,
            confusion_fig,
            misclassified_fig,
            metrics_summary,
            notebook_status,
        )
