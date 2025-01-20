from utils.generic import *


def balls_save_dataset(
		root: str,
		npix: int = 64,
		trn_size: int = int(8e4),
		tst_size: int = int(5e4),
		bounds: Tuple[float, float] = (0, 5), ):
	# sample latents
	z, z_tst = balls_sample_latents(
		trn_size, tst_size, bounds)
	# create save_dir
	save_dir = pjoin(root, f'npix-{npix}')
	os.makedirs(save_dir, exist_ok=True)
	# save
	save = {
		# trn + vld
		'z': z,
		'x': render_imgs(z, npix, bounds),
		# tst
		'z_tst': z_tst,
		'x_tst': render_imgs(z_tst, npix, bounds),
	}
	for name, obj in save.items():
		save_obj(
			obj=obj,
			file_name=name,
			save_dir=save_dir,
			verbose=True,
			mode='npy',
		)
	return


def render_imgs(
		z: np.ndarray,
		npix: int = 64,
		bounds: Tuple[float, float] = (0, 5), ):
	xx, yy = np.meshgrid(
		np.linspace(*bounds, npix),
		np.linspace(*bounds, npix),
	)
	x1, y1, x2, y2 = z.T
	b1 = bump2d(x1, y1, xx, yy)[:, np.newaxis, ...]
	b2 = bump2d(x2, y2, xx, yy)[:, np.newaxis, ...]
	return b1 + b2


def bump2d(x, y, xx, yy, t=1):
	x = np.reshape(x, (-1, 1, 1))
	y = np.reshape(y, (-1, 1, 1))

	r2 = (x - xx) ** 2 + (y - yy) ** 2
	r2 = np.clip(r2, None, 0.999)

	cond = np.sqrt(r2) < 1
	val = np.exp(t - t / (1 - r2))

	return np.where(cond, val, 0)


def balls_sample_latents(
		trn_size: int,
		tst_size: int,
		bounds: Tuple[float, float] = (0, 5),
		bounds_tst: Tuple[float, float] = None,
		seed: int = 0, ):
	rng = get_rng(seed)

	if bounds_tst is None:
		lower = 0.5 * (max(bounds) - min(bounds))
		bounds_tst = (lower, max(bounds))
	y1_tst = rng.uniform(*bounds_tst, size=tst_size)
	y2_tst = rng.uniform(*bounds_tst, size=tst_size)

	ratio = (
		(max(bounds) - min(bounds)) /
		(max(bounds_tst) - min(bounds_tst))
	)
	ratio = ratio ** 2 / (ratio ** 2 - 1) + 0.1

	y1_trn = rng.uniform(*bounds, size=int(np.ceil(ratio * trn_size)))
	y2_trn = rng.uniform(*bounds, size=int(np.ceil(ratio * trn_size)))

	# get l-shaped subset
	m1 = (
		(y1_trn < min(bounds_tst)) |
		(y1_trn >= max(bounds_tst))
	)
	m2 = (
		(y2_trn < min(bounds_tst)) |
		(y2_trn >= max(bounds_tst))
	)
	inds_trn = np.where(m1 | m2)[0]
	inds_trn = inds_trn[:trn_size]
	y1_trn = y1_trn[inds_trn]
	y2_trn = y2_trn[inds_trn]

	# ensure trn / tst are disjoint
	leakage1 = (
		(y1_trn >= min(bounds_tst)) &
		(y1_trn < max(bounds_tst))
	)
	leakage2 = (
		(y2_trn >= min(bounds_tst)) &
		(y2_trn < max(bounds_tst))
	)
	leakge = leakage1 & leakage2
	assert not leakge.sum()

	x1 = 0.25 * max(bounds)
	x2 = 0.75 * max(bounds)

	z_trn = np.stack([
		np.ones(trn_size) * x1,
		y1_trn,
		np.ones(trn_size) * x2,
		y2_trn,
	]).T
	z_tst = np.stack([
		np.ones(tst_size) * x1,
		y1_tst,
		np.ones(tst_size) * x2,
		y2_tst,
	]).T
	return z_trn, z_tst
