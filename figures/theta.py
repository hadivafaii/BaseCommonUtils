from .fighelper import *
from matplotlib.gridspec import GridSpec


def plot_theta(
		df: pd.DataFrame,
		key: str = 'prior_rate',
		display: bool = True,
		**kwargs, ):
	defaults = {
		'dpi': 100,
		'figsize': (5, 5.5),
		'xtick_spacing': 45,
		'width_ratios': [4, 1],
		'height_ratios': [1.5, 4],
		'hspace': 0.0,
		'wspace': 0.07,
		'ylim_lower': -0.02,
		'ylim_upper': 0.71,
		'hue': 'label',
		'style': None,
		'palette': {
			'cardinal': 'dimgrey',
			'oblique': 'tomato'},
		'hist_bin_width': 3.0,
		'hist_element': 'step',
		'hist_fill': True,
		'scatter_s': 20,
		'scatter_alpha': 0.9,
		'strip_alpha': 0.3,
		'strip_size': 4,
		'point_lw': 2,
		'point_capsize': 0.5,
		'point_color': 'k',
	}
	kwargs = setup_kwargs(defaults, kwargs)

	fig = plt.figure(
		figsize=kwargs['figsize'],
		dpi=kwargs['dpi'],
	)
	gs = GridSpec(
		nrows=2, ncols=2,
		width_ratios=kwargs['width_ratios'],
		height_ratios=kwargs['height_ratios'],
		hspace=kwargs['hspace'],
		wspace=kwargs['wspace'],
	)
	ax1 = fig.add_subplot(gs[0, 0])
	ax2 = fig.add_subplot(gs[1, 0])
	ax3 = fig.add_subplot(gs[1, 1])

	n = np.ceil(180 / kwargs['xtick_spacing'])
	xticks = [
		kwargs['xtick_spacing'] * i
		for i in range(int(n) + 1)
	]
	ylim = (
		kwargs['ylim_lower'],
		kwargs['ylim_upper'],
	)
	ax1.set(xticks=[], yticks=[])
	ax2.set(ylim=ylim, xticks=xticks)
	ax3.set(ylim=ylim, xticklabels=[], yticklabels=[])

	# histplot
	bins = np.linspace(
		start=0,
		stop=180 + kwargs['hist_bin_width'],
		num=int(np.ceil(180 / kwargs['hist_bin_width'])) + 2,
	) - kwargs['hist_bin_width'] / 2
	for lbl, color in kwargs['palette'].items():
		x2p = df.loc[df['label'] == lbl, 'theta_deg'].values
		sns.histplot(
			data=x2p,
			bins=bins,
			color=color,
			element=kwargs['hist_element'],
			fill=kwargs['hist_fill'],
			ax=ax1,
		)
	remove_ticks(ax1)

	# scatterplot
	sns.scatterplot(
		data=df,
		x='theta_deg',
		y=key,
		hue=kwargs['hue'],
		style=kwargs['style'],
		palette=kwargs['palette'],
		alpha=kwargs['scatter_alpha'],
		s=kwargs['scatter_s'],
		ax=ax2,
	)
	ax2.set(xlabel='', ylabel='')
	move_legend(ax2)  # , (1.4, 1.3))
	# ax2.set_xlabel(r"$\theta$", fontsize=14)
	# ax2.set_ylabel('Prior firing rate', fontsize=14)

	# stripplot
	axtw = ax3.twiny()

	sns.stripplot(
		data=df,
		x='label',
		y=key,
		hue=kwargs['hue'],
		palette=kwargs['palette'],
		size=kwargs['strip_size'],
		alpha=kwargs['strip_alpha'],
		order=['cardinal', 'oblique'],
		ax=ax3,
	)
	# noinspection PyArgumentList
	sns.pointplot(
		data=df,
		x='label',
		y=key,
		markers='o',
		legend=False,
		errorbar=('ci', 95),
		color=kwargs['point_color'],
		capsize=kwargs['point_capsize'],
		err_kws={'lw': kwargs['point_lw']},
		order=['cardinal', 'oblique'],
		ax=axtw,
	)
	ax3.set(xlabel='', ylabel='')
	axtw.set(xlabel='', xticks=[])
	move_legend(ax3)

	if display:
		plt.show()
	else:
		plt.close()
	return fig, (ax1, ax2, ax3, axtw)
