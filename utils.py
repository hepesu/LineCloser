import numpy as np
import cv2


def generate_random_gap(imgs, gap_configs, seed=None):
    bg = np.full(imgs[0].shape, 1., np.float32)

    imgs_with_gaps = []
    masks = []

    if seed is not None:
        np.random.seed(seed)

    for img in imgs:
        img_height, img_width = img.shape[:2]
        mask = np.zeros_like(img, np.float32)
        print(mask.shape)
        for gap_config in gap_configs:
            nb_min, nb_max, r_min, r_max, b_min, b_max = gap_config
            _mask = np.zeros_like(img, np.float32)

            for _ in range(np.random.randint(nb_min, nb_max)):
                center = (np.random.randint(img_width), np.random.randint(img_height))
                radius = np.random.randint(r_min, r_max)
                cv2.circle(_mask, center, radius, 1., -1)

                blur_radius = np.random.randint(b_min, b_max) * 2 + 1
                _mask = cv2.blur(_mask, (blur_radius, blur_radius))

                _mask = np.expand_dims(_mask, axis=-1)

            # accumulate masks
            mask = mask + _mask

        mask = np.clip(mask, 0., 1.)

        # composite with mix
        imgs_with_gaps.append(img * (1. - mask) + bg * mask)
        masks.append(mask * (1. - img))

    return np.array(imgs_with_gaps, np.float32), np.array(masks, np.float32)


if __name__ == "__main__":
    y = cv2.imread('./input/0.png', cv2.IMREAD_GRAYSCALE)
    y = np.expand_dims(y, -1) / 255

    gap_configs352 = [
        [50, 600, 2, 8, 0, 1],
        [50, 600, 2, 10, 0, 2],
        [1, 2, 5, 15, 0, 3]
    ]

    x, m = generate_random_gap([y], gap_configs352, 1)

    cv2.imwrite('./gap_x_check.png', x[0] * 255)
    cv2.imwrite('./gap_y_check.png', y * 255)
