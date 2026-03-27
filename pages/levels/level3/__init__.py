from importlib import import_module

__all__ = [
	'level3_layout',
	'register_level3_callbacks',
	'build_level3_execution_environment',
	'make_level3_placeholder_figure',
	'level3_extract_meta_from_code',
	'level3_build_meta',
	'level3_build_dataset',
	'level3_initialise_store',
	'level3_model_matches',
	'level3_initialise_model',
	'level3_dataset_preview_figure',
	'level3_activation_heatmap_figure',
	'level3_hidden_space_figure',
	'level3_confusion_matrix_figure',
	'level3_misclassified_figure',
	'level3_forward_summary_children',
	'level3_training_log_children',
	'level3_metrics_summary_children',
	'level3_dataset_summary_children',
	'level3_notebook_status_children',
	'level3_model_status_children',
	'level3_execution_live_children',
]


_EXPORT_TO_MODULE = {
	'level3_layout': 'pages.levels.level3.layout',
	'register_level3_callbacks': 'pages.levels.level3.callbacks',
	'build_level3_execution_environment': 'pages.levels.level3.methods',
	'make_level3_placeholder_figure': 'pages.levels.level3.methods',
	'level3_extract_meta_from_code': 'pages.levels.level3.methods',
	'level3_build_meta': 'pages.levels.level3.methods',
	'level3_build_dataset': 'pages.levels.level3.methods',
	'level3_initialise_store': 'pages.levels.level3.methods',
	'level3_model_matches': 'pages.levels.level3.methods',
	'level3_initialise_model': 'pages.levels.level3.methods',
	'level3_dataset_preview_figure': 'pages.levels.level3.methods',
	'level3_activation_heatmap_figure': 'pages.levels.level3.methods',
	'level3_hidden_space_figure': 'pages.levels.level3.methods',
	'level3_confusion_matrix_figure': 'pages.levels.level3.methods',
	'level3_misclassified_figure': 'pages.levels.level3.methods',
	'level3_forward_summary_children': 'pages.levels.level3.methods',
	'level3_training_log_children': 'pages.levels.level3.methods',
	'level3_metrics_summary_children': 'pages.levels.level3.methods',
	'level3_dataset_summary_children': 'pages.levels.level3.methods',
	'level3_notebook_status_children': 'pages.levels.level3.methods',
	'level3_model_status_children': 'pages.levels.level3.methods',
	'level3_execution_live_children': 'pages.levels.level3.methods',
}


def __getattr__(name):
	if name not in _EXPORT_TO_MODULE:
		raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

	module = import_module(_EXPORT_TO_MODULE[name])
	value = getattr(module, name)
	globals()[name] = value
	return value