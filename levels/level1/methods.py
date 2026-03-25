import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.decomposition import PCA


def build_architecture_diagram(model, hidden_size):
	fig = go.Figure()

	input_nodes = [index / 5 for index in range(4)]
	hidden_nodes = [index / 16 for index in range(hidden_size)]
	output_nodes = [index / 3 for index in range(3)]

	fig.add_trace(go.Scatter(
		x=[0] * 4,
		y=input_nodes,
		mode='markers+text',
		marker=dict(size=20, color='blue'),
		text=[f'I{index + 1}' for index in range(4)],
		name='Input',
	))
	fig.add_trace(go.Scatter(
		x=[1] * hidden_size,
		y=hidden_nodes,
		mode='markers+text',
		marker=dict(size=15, color='orange'),
		text=[f'H{index + 1}' for index in range(hidden_size)],
		name='Hidden',
	))
	fig.add_trace(go.Scatter(
		x=[2] * 3,
		y=output_nodes,
		mode='markers+text',
		marker=dict(size=20, color='green'),
		text=['O1', 'O2', 'O3'],
		name='Output',
	))

	for input_index in range(4):
		for hidden_index in range(hidden_size):
			fig.add_trace(go.Scatter(
				x=[0, 1],
				y=[input_nodes[input_index], hidden_nodes[hidden_index]],
				mode='lines',
				line=dict(width=1, color='gray'),
				showlegend=False,
				hoverinfo='skip',
			))

	fig.update_layout(
		title=f'Live Architecture: 4 -> {hidden_size} -> 3',
		xaxis=dict(showgrid=False, range=[-0.2, 2.2]),
		yaxis=dict(showgrid=False, range=[-0.2, 1.2]),
		height=400,
		showlegend=False,
	)
	return fig


def plot_decision_boundary(model, Xtrain_sample):
	pca = PCA(n_components=2)
	X_2d = pca.fit_transform(Xtrain_sample[:100])

	step = 0.02
	x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
	y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
	xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

	grid_input = np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], 2))]
	predictions = model.forward(grid_input)
	decision = np.argmax(predictions, axis=1).reshape(xx.shape)

	fig = go.Figure(data=go.Heatmap(
		z=decision,
		x=xx[0],
		y=yy[:, 0],
		colorscale='RdYlBu',
		hoverongaps=False,
	))
	fig.update_layout(
		title='Live Decision Boundary (PCA Projection)',
		xaxis_title='PC1',
		yaxis_title='PC2',
		height=350,
	)
	return fig


def generate_data(dataset):
	if dataset == 'linear':
		X, y = make_blobs(
			n_samples=300,
			centers=[(-2, -2), (2, 2)],
			cluster_std=0.8,
			random_state=42,
		)
	elif dataset == 'moons':
		X, y = make_moons(
			n_samples=300,
			noise=0.2,
			random_state=42,
		)
	elif dataset == 'circles':
		X, y = make_circles(
			n_samples=300,
			noise=0.1,
			factor=0.4,
			random_state=42,
		)
	else:
		X, y = make_blobs(
			n_samples=300,
			centers=[(-2, -2), (2, 2)],
			cluster_std=0.8,
			random_state=42,
		)

	return X, y
