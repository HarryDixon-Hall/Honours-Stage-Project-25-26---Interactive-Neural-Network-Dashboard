from importlib import import_module

__all__ = [
	'level2_layout',
	'register_level2_callbacks',
	'build_level2_dataset',
	'load_toy_dataset',
	'init_level2_mlp',
	'level2_forward_pass',
	'level2_evaluate_metrics',
	'level2_set_baseline_history',
	'train_level2_model',
	'make_decision_boundary_figure',
	'make_activation_figure',
	'make_network_diagram_figure',
	'make_level2_training_curves_figure',
	'make_level2_output_panel',
	'make_level2_summary_panel',
]


_EXPORT_TO_MODULE = {
	'level2_layout': 'pages.levels.level2.layout',
	'register_level2_callbacks': 'pages.levels.level2.callbacks',
	'build_level2_dataset': 'pages.levels.level2.methods',
	'load_toy_dataset': 'pages.levels.level2.methods',
	'init_level2_mlp': 'pages.levels.level2.methods',
	'level2_forward_pass': 'pages.levels.level2.methods',
	'level2_evaluate_metrics': 'pages.levels.level2.methods',
	'level2_set_baseline_history': 'pages.levels.level2.methods',
	'train_level2_model': 'pages.levels.level2.methods',
	'make_decision_boundary_figure': 'pages.levels.level2.methods',
	'make_activation_figure': 'pages.levels.level2.methods',
	'make_network_diagram_figure': 'pages.levels.level2.methods',
	'make_level2_training_curves_figure': 'pages.levels.level2.methods',
	'make_level2_output_panel': 'pages.levels.level2.methods',
	'make_level2_summary_panel': 'pages.levels.level2.methods',
}


def __getattr__(name):
	if name not in _EXPORT_TO_MODULE:
		raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

	module = import_module(_EXPORT_TO_MODULE[name])
	value = getattr(module, name)
	globals()[name] = value
	return value