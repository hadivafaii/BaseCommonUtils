from .plotting import *
from IPython.display import HTML
from matplotlib.animation import FuncAnimation


def show_movie(
		x: np.ndarray,
		max_n: int = 100,
		**kwargs, ):
	defaults = dict(
		cmap='Greys_r',
		figsize=(6.5, 5.0),
		interval=50,  # 50ms between frames
		blit=True,
	)
	kwargs = setup_kwargs(defaults, kwargs)

	fig, ax = plt.subplots(
		figsize=kwargs['figsize'])

	_x = tonp(x[:max_n])

	# Show first frame and set colormap
	im = ax.imshow(
		_x[0],
		vmin=np.nanmin(_x),
		vmax=np.nanmax(_x),
		cmap=kwargs['cmap'],
	)
	plt.colorbar(im)
	remove_ticks(ax, False)

	def _update(i):
		im.set_array(_x[i])
		return [im]

	anim = FuncAnimation(
		fig=fig,
		func=_update,
		frames=len(_x),
		interval=kwargs['interval'],
		blit=kwargs['blit'],
	)

	# Convert animation to HTML5 video
	html_video = anim.to_jshtml()
	plt.close()
	return HTML(html_video)
