from data.utils import *


# For a path
def trainGenerator_path(
        batch_size, train_path, aug_dict, image_color_mode, mask_color_mode, target_size, classes, image_folder="image",
        mask_folder="label", image_save_prefix="image", mask_save_prefix="mask", save_to_dir=None):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    #
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,  # None'categorical'
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    #
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,  # None'categorical'
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        img, mask = adjust_data(img, mask, classes)
        yield (img, mask)


#TODO：For a file 尚未完成
def trainGenerator_file(
        batch_size, train_path, aug_dict, image_color_mode, mask_color_mode, target_size, image_folder="image",
        mask_folder="label", image_save_prefix="image", mask_save_prefix="mask", save_to_dir=None):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    #
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,  # None'categorical'
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    #
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,  # None'categorical'
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)
        yield (img, mask)


def testGenerator(imagelist, test_path, num_image):
    for i in range(num_image):
        # new
        img = np.array(imageio.imread(os.path.join(test_path, imagelist[i])))
        img = img / 255
        # img = trans.resize(img,target_size)
        img = np.reshape(img, (1,) + img.shape)  # (1,256,256,3)
        # img = np.reshape(img,img.shape+(1,)) if not flag_multi_class else img
        yield img
