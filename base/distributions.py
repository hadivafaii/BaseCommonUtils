from base.utils_model import *
dists.Distribution.set_default_validate_args(False)


class Poisson:
	def __init__(
			self,
			log_rate: torch.Tensor,
			temp: float = 0.0,
			n_exp: int | str = 'infer',
			clamp: float | None = None,
	):
		assert temp >= 0.0, f"must be non-neg: {temp}"
		self.temp = temp
		self.clamp = clamp
		# setup rate & exp dist
		if clamp is not None:
			log_rate = softclamp_upper(
				log_rate, clamp)
		eps = torch.finfo(torch.float32).eps
		self.rate = torch.exp(log_rate) + eps
		self.exp = dists.Exponential(self.rate)
		# compute n_exp
		if n_exp == 'infer':
			max_rate = self.rate.max().item()
			n_exp = compute_n_exp(max_rate, 1e-6)
		self.n_exp = int(n_exp)

	@property
	def mean(self):
		return self.rate

	@property
	def variance(self):
		return self.rate

	def rsample(self, hard: bool = False):
		# noinspection PyTypeChecker
		x = self.exp.rsample((self.n_exp,))  # inter-event times
		times = torch.cumsum(x, dim=0)  # arrival times of events

		indicator = times < 1.0
		z_hard = indicator.sum(0).float()

		if self.temp > 0:
			indicator = torch.sigmoid(
				(1.0 - times) / self.temp)
			z = indicator.sum(0).float()
		else:
			z = z_hard

		if hard:
			return z + (z_hard - z).detach()
		return z

	def sample(self):
		return torch.poisson(self.rate).float()

	def log_prob(self, samples: torch.Tensor):
		return (
			- self.rate
			- torch.lgamma(samples + 1)
			+ samples * torch.log(self.rate)
		)


class PoissonQuantile:
	def __init__(
			self,
			log_rate: torch.Tensor,
			temp: float = 0.0,
			q: List[float] = None,
			clamp: float = 5.3,
	):
		assert temp >= 0.0, f"must be non-neg: {temp}"
		self.t = temp
		self.c = clamp
		self._init_rates(log_rate)
		self._init_quantiles(q)
		self._init_masks()
		self._init_n_exp()

	@property
	def mean(self):
		return self.rate

	@property
	def variance(self):
		return self.rate

	def rsample(self, hard: bool = False):
		samples = torch.zeros_like(
			self.rate.ravel())
		index = torch.zeros(
			self.rate.numel(),
			dtype=torch.long,
			device=self.rate.device,
		)
		z = []
		offset = 0
		for i, m in self.masks.items():
			z.append(self._rsample_per_group(
				rates=self.rate.ravel()[m],
				n_exp=self.n_exp[i],
				hard=hard,
			))
			index[m] = torch.arange(
				int(m.sum()),
				device=m.device,
			) + offset
			offset += m.sum()
		samples.scatter_(0, index, torch.cat(z))
		samples = samples.view(self.rate.shape)
		return samples

	# noinspection PyTypeChecker
	def _rsample_per_group(
			self,
			rates: torch.Tensor,
			n_exp: int,
			hard: bool = False, ):
		exp = dists.Exponential(rates)
		x = exp.rsample((n_exp,))
		times = torch.cumsum(x, dim=0)

		indicator = times < 1.0
		z_hard = indicator.sum(0).float()

		if self.t > 0:
			indicator = torch.sigmoid(
				(1.0 - times) / self.t)
			z = indicator.sum(0).float()
		else:
			z = z_hard

		if hard:
			return z + (z_hard - z).detach()
		return z

	def sample(self):
		return torch.poisson(self.rate).float()

	def log_p(self, samples: torch.Tensor):
		return (
			- self.rate
			- torch.lgamma(samples + 1)
			+ samples * torch.log(self.rate)
		)

	def _init_rates(self, log_rate):
		eps = torch.finfo(torch.float32).eps
		log_rate = softclamp_upper(log_rate, self.c)
		self.rate = torch.exp(log_rate) + eps
		return

	@torch.no_grad()
	def _init_quantiles(self, quantiles):
		quantiles = quantiles or [0.5, 0.75, 0.95]
		self.q = sorted(set(quantiles))
		return

	@torch.no_grad()
	def _init_masks(self):
		masks = {}
		rate_flat = self.rate.ravel()
		for i in range(len(self.q) + 1):
			if i == 0:
				higher = torch.quantile(
					rate_flat, self.q[i])
				m = rate_flat < higher
			elif i == len(self.q):
				lower = torch.quantile(
					rate_flat, self.q[i - 1])
				m = rate_flat >= lower
			else:
				lower = torch.quantile(
					rate_flat, self.q[i - 1])
				higher = torch.quantile(
					rate_flat, self.q[i])
				m = (
					(lower <= rate_flat) &
					(rate_flat < higher)
				)
			masks[i] = m
		self.masks = masks
		return

	@torch.no_grad()
	def _init_n_exp(self):
		r_max = {
			i: self.rate.ravel()[m].max().item()
			for i, m in self.masks.items()
		}
		self.n_exp = {
			i: compute_n_exp(r, 1e-6)
			for i, r in r_max.items()
		}
		return


# noinspection PyAbstractClass
class Normal(dists.Normal):
	def __init__(
			self,
			loc: torch.Tensor,
			log_scale: torch.Tensor,
			temp: float = 1.0,
			clamp: float | None = None,
			seed: int = None,
			**kwargs,
	):
		if clamp is not None:
			log_scale = softclamp_sym(
				log_scale, clamp)
		super(Normal, self).__init__(
			loc=loc, scale=torch.exp(log_scale), **kwargs)

		assert temp >= 0
		if temp != 1.0:
			self.scale *= temp
		self.temp = temp
		self.clamp = clamp
		self._init_rng(seed, loc.device)

	def kl(self, p: dists.Normal = None):
		if p is None:
			term1 = self.mean
			term2 = self.scale
		else:
			term1 = (self.mean - p.mean) / p.scale
			term2 = self.scale / p.scale
		kl = 0.5 * (
			term1.pow(2) + term2.pow(2) +
			torch.log(term2).mul(-2) - 1
		)
		return kl

	def retrieve_noise(self, samples):
		return (samples - self.loc).div(self.scale)

	@torch.no_grad()
	def sample(self, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		samples = torch.normal(
			mean=self.loc.expand(shape),
			std=self.scale.expand(shape),
			generator=self.rng,
		)
		return samples

	def _init_rng(self, seed, device):
		if seed is not None:
			self.rng = torch.Generator(device)
			self.rng.manual_seed(seed)
		else:
			self.rng = None
		return


# noinspection PyAbstractClass
class Laplace(dists.Laplace):
	def __init__(
			self,
			loc: torch.Tensor,
			log_scale: torch.Tensor,
			temp: float = 1.0,
			clamp: float = 5.3,
			**kwargs,
	):
		if clamp is not None:
			log_scale = softclamp_sym(log_scale, clamp)
		super(Laplace, self).__init__(
			loc=loc, scale=torch.exp(log_scale), **kwargs)

		assert temp >= 0
		if temp != 1.0:
			self.scale *= temp
		self.t = temp
		self.c = clamp

	def kl(self, p: dists.Laplace = None):
		if p is not None:
			mean, scale = p.mean, p.scale
		else:
			mean, scale = 0, 1

		delta_m = torch.abs(self.mean - mean)
		delta_b = self.scale / scale
		term1 = delta_m / self.scale
		term2 = delta_m / scale

		kl = (
			delta_b * torch.exp(-term1) +
			term2 - torch.log(delta_b) - 1
		)
		return kl


# noinspection PyAbstractClass
class Categorical(dists.RelaxedOneHotCategorical):
	def __init__(
			self,
			logits: torch.Tensor,
			temp: float = 1.0,
			**kwargs,
	):
		self._categorical = None
		temp = max(temp, torch.finfo(torch.float).eps)
		super(Categorical, self).__init__(
			logits=logits, temperature=temp, **kwargs)

	@property
	def t(self):
		return self.temperature

	@property
	def mean(self):
		return self.probs

	@property
	def variance(self):
		return self.probs * (1 - self.probs)

	def kl(self, p: dists.Categorical = None):
		if p is None:
			probs = torch.full(
				size=self.probs.size(),
				fill_value=1 / self.probs.size(-1),
			)
			p = dists.Categorical(probs=probs)
		q = dists.Categorical(probs=self.probs)
		return dists.kl.kl_divergence(q, p)


# noinspection PyAbstractClass,PyTypeChecker
class PoissonTorch(dists.Poisson):
	def __init__(
			self,
			log_rate: torch.Tensor,
			temp: float = 0.1,
			clamp: float = 5.3,
			n_exp: int = 263,
			**kwargs,
	):
		if clamp is not None:
			log_rate = softclamp_upper(log_rate, clamp)
		eps = torch.finfo(torch.float32).eps
		rate = torch.exp(log_rate) + eps
		super(PoissonTorch, self).__init__(
			rate=rate, **kwargs)

		assert temp >= 0
		self.t = temp
		self.c = clamp
		self.n = n_exp
		self.exp = dists.Exponential(self.rate)

	def rsample(self, **kwargs):
		x = self.exp.rsample((self.n,))  # inter-event times
		times = torch.cumsum(x, dim=0)  # arrival times of events

		if self.t == 0:
			indicator = times < 1.0
		else:
			indicator = torch.sigmoid(
				(1.0 - times) / self.t)

		z = indicator.sum(0)  # event counts
		return z.float()

	def log_p(self, samples: torch.Tensor):
		return (
			- self.rate
			- torch.lgamma(samples + 1)
			+ samples * torch.log(self.rate)
		)


class GaBe:
	def __init__(
			self,
			mu: torch.Tensor,
			logsig: torch.Tensor,
			temp: float = 1.0,
			clamp: float = 5.3,
			seed: int = None,
			device: torch.device = None,
	):
		self.mu = mu
		logsig = softclamp_sym(logsig, clamp)
		self.sigma = torch.exp(logsig)
		assert temp >= 0
		self.temp = temp
		if temp != 1.0:
			self.sigma *= temp
		if seed is not None:
			self.rng = torch.Generator(device)
			self.rng.manual_seed(seed)
		else:
			self.rng = None

	def rsample(self):
		if self.rng is None:
			u = sample_normal_jit(self.mu, self.sigma)
		else:
			u = sample_normal(self.mu, self.sigma, self.rng)
		p = phi(u)
		# TODO: z = sample_bernoulli(p)
		#  return z
		return

	def log_p(self, samples: torch.Tensor):
		zscored = (samples - self.mu) / self.sigma
		log_p = (
			- 0.5 * zscored.pow(2)
			- 0.5 * np.log(2*np.pi)
			- torch.log(self.sigma)
		)
		return log_p

	def kl(self, p=None):
		if p is None:
			term1 = self.mu
			term2 = self.sigma
		else:
			term1 = (self.mu - p.mu) / p.sigma
			term2 = self.sigma / p.sigma
		kl = 0.5 * (
			term1.pow(2) + term2.pow(2) +
			torch.log(term2).mul(-2) - 1
		)
		return kl


def compute_n_exp(rate: float, p: float = 1e-6):
	assert rate > 0.0, f"must be positive, got: {rate}"
	pois = sp_stats.poisson(rate)
	n_exp = pois.ppf(1.0 - p)
	return int(n_exp)


def phi(x):
	sqrt2 = torch.sqrt(torch.tensor(2.0))
	x = torch.special.erf(x.div(sqrt2))
	x = (1 + x).mul(0.5)
	return x


def softclamp(x: torch.Tensor, upper: float, lower: float = 0.0):
	return lower + F.softplus(x - lower) - F.softplus(x - upper)


def softclamp_sym(x: torch.Tensor, c: float):
	return x.div(c).tanh_().mul(c)


def softclamp_sym_inv(y: torch.Tensor, c: float) -> torch.Tensor:
	y = y.clone().detach()

	if not torch.all((y > -c) & (y < c)):
		msg = "must: all(-c < y < c)"
		raise ValueError(msg)

	x = y.div(c).atanh_().mul(c)
	return x


def softclamp_upper(x: torch.Tensor, c: float):
	return c - F.softplus(c - x)


def softclamp_upper_inv(y: torch.Tensor, c: float):
	y = y.clone().detach()

	if not torch.all(y < c):
		msg = "must: all(y < c)"
		raise ValueError(msg)

	a = (c - y).float()
	log_term = torch.log1p(-torch.exp(-a))
	log_expm1 = a + log_term
	x = c - log_expm1
	return x


@torch.jit.script
def sample_normal_jit(
		mu: torch.Tensor,
		sigma: torch.Tensor, ):
	eps = torch.empty_like(mu).normal_()
	return sigma * eps + mu


def sample_normal(
		mu: torch.Tensor,
		sigma: torch.Tensor,
		rng: torch.Generator = None, ):
	eps = torch.empty_like(mu).normal_(
		mean=0., std=1., generator=rng)
	return sigma * eps + mu


@torch.jit.script
def residual_kl(
		delta_mu: torch.Tensor,
		delta_sig: torch.Tensor,
		sigma: torch.Tensor, ):
	return 0.5 * (
		delta_sig.pow(2) - 1 +
		(delta_mu / sigma).pow(2)
	) - torch.log(delta_sig)
