def load_test_data(data_dir, sz, rescale = 1):
    from PIL import Image
    import os
    import numpy as np
    from tqdm import tqdm

    num_images = len(os.listdir(data_dir))
    print("{} images found".format(num_images))
    file_name = []
    x_test = np.empty((num_images, sz,sz,3), dtype=np.float32)

    c = 0
    for f in tqdm(os.listdir(data_dir)):
        img = Image.open(os.path.join(data_dir, f))
        img = img.convert(mode = "RGB")
        x_test[c,...] = img.resize((sz,sz))
        file_name.append(f)
        c += 1

    return x_test * rescale, file_name