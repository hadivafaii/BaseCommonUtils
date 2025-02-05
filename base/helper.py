from utils.plotting import *
from sklearn import metrics as sk_metric
from sklearn import neighbors as sk_neigh
from sklearn import inspection as sk_inspect
from sklearn import linear_model as sk_linear
from sklearn import decomposition as sk_decomp
from sklearn import model_selection as sk_modselect
from statsmodels.stats.multitest import multipletests


def rotate_img(
		x: torch.Tensor,
		angle: float,
		interpolation: str = 'bilinear',
		hist_match: bool = True, ):
	if angle % 360 == 0:
		return x
	interpolation = getattr(
		F_vis.InterpolationMode,
		interpolation.upper(),
	)
	x_rotated = F_vis.rotate(
		inpt=x,
		angle=angle,
		interpolation=interpolation,
	)
	if hist_match:
		x_rotated = match_histograms(
			src=x_rotated, ref=x)
	return x_rotated


def match_histograms(
		src: torch.Tensor,
		ref: torch.Tensor, ) -> torch.Tensor:
	orig_dims = src.ndim
	while src.ndim < 4:
		src = src.unsqueeze(0)
		ref = ref.unsqueeze(0)
	result = match_histograms_batch(
		src=src, ref=ref)
	while result.ndim > orig_dims:
		result = result.squeeze(0)
	return result


def match_histograms_batch(
		src: torch.Tensor,
		ref: torch.Tensor, ) -> torch.Tensor:
	b, c, h, w = src.shape
	s = src.view(b, c, h * w)
	r = ref.view(b, c, h * w)
	s_sorted, s_indices = torch.sort(s, dim=2)
	r_sorted, _ = torch.sort(r, dim=2)
	s_matched = torch.empty_like(s)
	s_matched.scatter_(2, s_indices, r_sorted)
	return s_matched.view(b, c, h, w)


def translate_img(
		x: torch.Tensor,
		translate_x: float,
		translate_y: float,
		interpolation: str = 'bilinear',
		hist_match: bool = True, ):
	if translate_x == translate_y == 0:
		return x
	interpolation = getattr(
		F_vis.InterpolationMode,
		interpolation.upper(),
	)
	x_translated = F_vis.affine(
		inpt=x,
		angle=0.0,
		scale=1.0,
		shear=[0.0],
		translate=[translate_x, translate_y],
		interpolation=interpolation,
	)
	if hist_match:
		x_translated = match_histograms(
			tonp(x_translated), tonp(x))
	return x_translated


def job_runner_script(
		device: int,
		dataset: str,
		model: str,
		archi: str,
		seed: int = 0,
		args: str = None,
		bash_script: str = 'fit_vae.sh',
		relative_path: str = '.', ):
	s = ' '.join([
		f"'{device}'",
		f"'{dataset}'",
		f"'{model}'",
		f"'{archi}'",
	])
	s = f"{relative_path}/{bash_script} {s}"
	s = f"{s} --seed {seed}"
	if args is not None:
		s = f"{s} {args}"
	return s


def skew(x: np.ndarray, axis: int = 0):
	x1 = np.expand_dims(np.expand_dims(np.take(
		x, 0, axis=axis), axis=axis), axis=axis)
	x2 = np.expand_dims(np.expand_dims(np.take(
		x, 1, axis=axis), axis=axis), axis=axis)
	x3 = np.expand_dims(np.expand_dims(np.take(
		x, 2, axis=axis), axis=axis), axis=axis)
	s1 = np.concatenate([np.zeros_like(x1), -x3, x2], axis=axis+1)
	s2 = np.concatenate([x3, np.zeros_like(x2), -x1], axis=axis+1)
	s3 = np.concatenate([-x2, x1, np.zeros_like(x3)], axis=axis+1)
	s = np.concatenate([s1, s2, s3], axis=axis)
	return s


def compute_sta(
		n_lags: int,
		stim: torch.Tensor,
		spks: torch.Tensor,
		good: torch.Tensor = None,
		zscore: bool = False,
		nanzero: bool = True,
		verbose: bool = False, ):
	assert n_lags >= 0
	shape = stim.shape
	nc = spks.shape[-1]
	sta_shape = (nc, n_lags + 1, *shape[1:])
	shape = (nc,) + (1,) * len(shape)
	if zscore:
		# TODO:
		stim = sp_stats.zscore(stim)
	if good is None:
		inds = torch.arange(len(stim))
	else:
		inds = good.copy()
	inds = inds[inds > n_lags]
	sta = torch.zeros(sta_shape, device=stim.device)
	for t in tqdm(inds, disable=not verbose):
		# zero n_lags allowed:
		x = stim[t - n_lags: t + 1]
		for i in range(nc):
			y = spks[t, i]
			if y > 0:
				sta[i] += x * y
	n = spks[inds].sum(0)
	n = n.reshape(shape)
	sta /= n
	if nanzero:
		nans = torch.isnan(sta)
		sta[nans] = 0.0
		if nans.sum() and verbose:
			warnings.warn(
				"NaN in STA",
				RuntimeWarning,
			)
	return sta
