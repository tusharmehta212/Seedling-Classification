def load_data(data_dir, sz, train_split = 1, rescale = 1):
    """
        Arguments : 
                    data_dir : directory containing seedling dataset
                    sz : image height and width
                    train_split : ratio of train_split w.r.t. test_split
                    rescale : if rescaling image
                    
        Returns  (x_train, y_train, x_valid, y_valid, class_dict, class_dict_inv)              
    """
    from PIL import Image
    from keras.utils import to_categorical
    import os
    from sklearn.model_selection import train_test_split
    import numpy as np
    from tqdm import tqdm

    classes = []
    for i, j in enumerate(os.listdir(data_dir)):
        classes.append(j)
    classes.sort()

    class_dict = {}
    class_dict_inv = {}

    for i, j in enumerate(classes):
        class_dict[i] = j
        class_dict_inv[j] = i

    num_images = 0
    for i in class_dict_inv:
        num_images += len(os.listdir(os.path.join(data_dir, i)))

    print("{} images found".format(num_images))

    train_x = np.empty((num_images, sz, sz, 3), dtype=np.float32)
    train_y = np.empty((num_images,), dtype=np.int8)
    label = []

    count = 0

    for i in tqdm(class_dict_inv):
        class_path = os.path.join(data_dir, str(i))
        image_paths = os.listdir(class_path)
        for j in image_paths:
            img = Image.open(os.path.join(class_path, j))
            img = img.convert(mode="RGB")
            train_x[count, ...] = img.resize((sz, sz))
            train_y[count, ...] = class_dict_inv[i]
            label.append(j)
            count += 1
    train_x = train_x * rescale
    x_train, x_valid, y_train, y_valid = train_test_split(train_x, to_categorical(train_y, num_classes=12), shuffle=True,train_size=train_split, test_size=1 - train_split, random_state=57)

    return (x_train, y_train, x_valid, y_valid, class_dict, class_dict_inv)
