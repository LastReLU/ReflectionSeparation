import cv2
import numpy as np
from tqdm.auto import tqdm


def filter_images(paths, limit=None):
    ret = []
    for path in tqdm(paths):
        try:
            img = cv2.imread(path)
            w, h, ch = img.shape
            assert w > 128 and h > 128 and ch == 3
            ret.append(path)
            if limit and len(ret) >= limit:
                break
        except:
            pass
    return ret


def random_crop(img, w=None, h=None):
    # slightly freaky, but let it bee
    img_w, img_h, ch = img.shape
    if w is None and h is None:
        w = h = min(img_w, img_h)
    assert (img_w >= w) and (img_h >= h) and (ch == 3), "Bad image shape: {}".format(img.shape)
    x = np.random.randint(0, img_w - w) if img_w - w > 0 else 0
    y = np.random.randint(0, img_h - h) if img_h - h > 0 else 0
    return img[x: x + w, y: y + h, :]


class DummyDataset:
    def __init__(self, paths, transform_fn=None, reflection_size=5, blur=5):
        self.paths = paths
        self.transform_fn = transform_fn
        # self.reflects = reflects
        self.reflection_size = reflection_size
        self.blur = blur

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        img = cv2.imread(path)
        img = random_crop(img).astype(np.float32) / 255.0  # we need squared initial image
        resized = cv2.resize(img, (128, 128))  # this is transmission image
        random_cropped = random_crop(img, 128, 128)  # for reflected image
        alpha = np.float32(np.random.uniform(0.75, 0.8))
        # let's generate kernel for double reflections with bluuuuuring
        kernel = np.zeros((self.reflection_size, self.reflection_size))
        x1, x2, y1, y2 = np.random.randint(0, self.reflection_size, size=4)
        kernel[x1, y1] = 1.0 - np.sqrt(alpha)
        kernel[x2, y2] = np.sqrt(alpha) - alpha
        if self.blur > 0:
            kernel = cv2.GaussianBlur(kernel, (self.blur, self.blur), 0)
        reflected = cv2.filter2D(random_cropped, -1, kernel)

        return dict(
            img=np.transpose(resized, (2, 0, 1)),  # transmitted image
            reflected=np.transpose(reflected, (2, 0, 1)),  # reflected image
            alpha=alpha,  # alpha for this reflected images
        )


def all_transform(a, b):
    """
    :param a: batch of images from one domain
    :param b: batch of images from another domain
    :param device: 'cuda' or 'cpu'
    """
    # there are many options about combining images, let's make it simple first:
    alpha_transmitted = b['alpha'][:, None, None, None] * a['img']
    reflected = b['reflected']
    synthetic = alpha_transmitted + reflected

    return dict(
        synthetic=synthetic,
        alpha_transmitted=alpha_transmitted,
        reflected=reflected,
    )


