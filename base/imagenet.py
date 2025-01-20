from utils.generic import *


def process_imagenet(path: str = 'Datasets/ImageNet32'):
	path = add_home(path)
	files = sort_batches(os.listdir(path))

	trn_files = [f for f in files if 'train_data' in f]
	vld_files = [f for f in files if 'val_data' in f]

	trn_files = [pjoin(path, f) for f in trn_files]
	vld_files = [pjoin(path, f) for f in vld_files]

	x_trn, y_trn = load_cat(trn_files)
	x_vld, y_vld = load_cat(vld_files)

	# reshape, shift, rescale
	x_trn = transform(x_trn)
	x_vld = transform(x_vld)

	# shift: 0, ..., 999
	y_trn -= 1
	y_vld -= 1

	# save
	save_dir = pjoin(path, 'processed')
	os.makedirs(save_dir, exist_ok=True)
	_save = {
		'x_trn': x_trn,
		'y_trn': y_trn,
		'x_vld': x_vld,
		'y_vld': y_vld,
	}
	for name, obj in _save.items():
		save_obj(
			obj=obj,
			file_name=name,
			save_dir=save_dir,
			verbose=True,
			mode='npy',
		)
	return


def transform(x, mean=(0.5,) * 3, std=(0.5,) * 3):
	x = x.reshape(-1, 3, 32, 32)
	x = x.astype(np.float32) / 255.0

	mean = np.array(mean).reshape(1, 3, 1, 1)
	std = np.array(std).reshape(1, 3, 1, 1)

	x = (x - mean) / std

	return x


def load_cat(files: List[str]):
	x, y = [], []
	for f in files:
		data_dict = unpickle(f)
		y.append(data_dict['labels'])
		x.append(data_dict['data'])
	x, y = cat_map([x, y])
	assert len(x) == len(y)
	return x, y


def sort_batches(files: List[str]):
	def _extract_number(name):
		try:
			i = int(name.split('_')[-1])
		except ValueError:
			i = -1
		return i

	return sorted(files, key=_extract_number)


def unpickle(file: str):
	with open(file, 'rb') as fo:
		data_dict = pickle.load(fo)
	return data_dict
