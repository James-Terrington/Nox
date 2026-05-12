import sys
import pathlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable debugging logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import cv2
import skimage
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
import pickle
import gc
import csv
import time
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Add, Multiply, Concatenate, LeakyReLU, Lambda, Resizing
from tqdm.auto import tqdm
import torch
import lpips
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess

# enable dynamic memory allocation
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

# global variables
epochs = 800
n_channels = 3 # 3 for RGB or 1 for grayscale
patch_size = 512
input_size = 512  # training/eval resize
stride = 128
border = int((patch_size - stride)/2)
generator_fname = os.getcwd() + os.sep + ('generator_gray_resnet.weights.h5', 'generator_color_resnet.weights.h5')[n_channels == 3]
generator_ckpt = os.getcwd() + os.sep + ('generator_gray_resnet_ckpt', 'generator_color_resnet_ckpt')[n_channels == 3]
discriminator_fname = os.getcwd() + os.sep + ('discriminator_gray_resnet.weights.h5', 'discriminator_color_resnet.weights.h5')[n_channels == 3]
discriminator_ckpt = os.getcwd() + os.sep + ('discriminator_gray_resnet_ckpt', 'discriminator_color_resnet_ckpt')[n_channels == 3]

# training parameters
BATCH_SIZE = 1 # number of training samples to work through before the model’s parameters are updated
INFER_BATCH_SIZE = 1
ema = 0.9995 # exponential moving average: keep ema % of the existing state and (1 - ema) % of the new state
ema_val = ema
lr = 5e-6 # learning rate
patience = 2 # number of epochs with no improvement after which learning rate will be reduced
cooldown = 4 # number of epochs to wait before resuming normal operation after lr has been reduced
validation = True
save_freq = 1

# START: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
def generator():
    tf.keras.backend.clear_session()

    if n_channels != 3:
        raise ValueError("ResNet50V2 pretrained encoder requires n_channels = 3")

    inputs = tf.keras.layers.Input(
        shape=(patch_size, patch_size, n_channels),
        name="gen_input_image"
    )

    backbone = ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )
    backbone.trainable = False

    skip1 = backbone.get_layer("conv1_conv").output
    skip2 = backbone.get_layer("conv2_block3_out").output
    skip3 = backbone.get_layer("conv3_block4_out").output
    skip4 = backbone.get_layer("conv4_block6_out").output
    bottleneck = backbone.get_layer("conv5_block3_out").output

    init = tf.initializers.GlorotUniform()

    def resize_like(args):
        src, ref = args
        return tf.image.resize(src, tf.shape(ref)[1:3], method="bilinear")

    def up_block(x, skip, filters, name):
        x = tf.keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation="bilinear",
            name=f"{name}_upsample"
        )(x)
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            kernel_initializer=init,
            name=f"{name}_conv"
        )(x)
        x = tf.keras.layers.LayerNormalization(name=f"{name}_ln")(x)
        x = tf.keras.layers.ReLU(name=f"{name}_relu")(x)

        skip = tf.keras.layers.Lambda(
            resize_like,
            name=f"{name}_skip_resize"
        )([skip, x])

        x = tf.keras.layers.Concatenate(name=f"{name}_concat")([x, skip])
        return x

    x = up_block(bottleneck, skip4, 512, "up1")
    x = up_block(x,          skip3, 256, "up2")
    x = up_block(x,          skip2, 128, "up3")
    x = up_block(x,          skip1,  64, "up4")

    x = tf.keras.layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear",
        name="final_upsample"
    )(x)

    predicted_stars = tf.keras.layers.Conv2D(
        n_channels,
        kernel_size=3,
        padding="same",
        kernel_initializer=init,
        name="predicted_stars_conv"
    )(x)
    predicted_stars = tf.keras.layers.ReLU(name="predicted_stars")(predicted_stars)

    output = tf.keras.layers.Subtract(name="generator_output")([inputs, predicted_stars])

    return tf.keras.Model(inputs=inputs, outputs=output, name="generator")
# END: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License

# START: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
def discriminator():
    init = tf.initializers.GlorotUniform()

    inp = tf.keras.layers.Input(
        shape=(patch_size, patch_size, n_channels),
        name="dis_input_image"
    )
    tgt = tf.keras.layers.Input(
        shape=(patch_size, patch_size, n_channels),
        name="dis_target_image"
    )

    # Conditional discriminator: judge (input, target) pair together
    x = tf.keras.layers.Concatenate(axis=3, name="dis_concat_inputs")([inp, tgt])

    p1 = tf.keras.layers.Conv2D(
        64, kernel_size=4, strides=2, padding="same",
        kernel_initializer=init, name="dis_conv1"
    )(x)
    p1 = tf.keras.layers.LeakyReLU(alpha=0.2, name="dis_lrelu1")(p1)

    p2 = tf.keras.layers.Conv2D(
        128, kernel_size=4, strides=2, padding="same",
        kernel_initializer=init, name="dis_conv2"
    )(p1)
    p2 = tf.keras.layers.LayerNormalization(name="dis_ln2")(p2)
    p2 = tf.keras.layers.LeakyReLU(alpha=0.2, name="dis_lrelu2")(p2)

    p3 = tf.keras.layers.Conv2D(
        256, kernel_size=4, strides=2, padding="same",
        kernel_initializer=init, name="dis_conv3"
    )(p2)
    p3 = tf.keras.layers.LayerNormalization(name="dis_ln3")(p3)
    p3 = tf.keras.layers.LeakyReLU(alpha=0.2, name="dis_lrelu3")(p3)

    p4 = tf.keras.layers.Conv2D(
        512, kernel_size=4, strides=1, padding="same",
        kernel_initializer=init, name="dis_conv4"
    )(p3)
    p4 = tf.keras.layers.LayerNormalization(name="dis_ln4")(p4)
    p4 = tf.keras.layers.LeakyReLU(alpha=0.2, name="dis_lrelu4")(p4)

    p5 = tf.keras.layers.Conv2D(
        512, kernel_size=4, strides=1, padding="same",
        kernel_initializer=init, name="dis_conv5"
    )(p4)
    p5 = tf.keras.layers.LayerNormalization(name="dis_ln5")(p5)
    p5 = tf.keras.layers.LeakyReLU(alpha=0.2, name="dis_lrelu5")(p5)

    logits = tf.keras.layers.Conv2D(
        1, kernel_size=4, strides=1, padding="same",
        kernel_initializer=init, name="dis_logits"
    )(p5)
    predict = tf.keras.layers.Activation("sigmoid", name="dis_sigmoid")(logits)

    return tf.keras.Model(
        inputs=[inp, tgt],
        outputs=[p1, p2, p3, p4, p5, predict],
        name="discriminator"
    )
# END: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License

def build_vgg_loss():
    """
    Returns a frozen VGG19 feature extractor for perceptual loss.
    Extracts features from block1_conv2, block2_conv2, block3_conv4,
    block4_conv4 — a standard multi-scale perceptual loss setup.
    Inputs are expected in [-1, 1], matching your training scaling.
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet',
                                       input_shape=(None, None, 3))
    vgg.trainable = False
    layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4']
    outputs = [vgg.get_layer(n).output for n in layer_names]
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs, name='vgg_perceptual')
    model.trainable = False

    def perceptual_loss(y_true, y_pred):
        # Rescale from [-1,1] to [0,255] for VGG (expects unnormalised RGB)
        y_true_vgg = (y_true + 1.0) * 127.5
        y_pred_vgg = (y_pred + 1.0) * 127.5
        feats_true = model(y_true_vgg, training=False)
        feats_pred = model(y_pred_vgg, training=False)
        loss = tf.zeros((), dtype=tf.float32)
        # Layer weights: more weight on deeper features
        weights = [0.1, 0.2, 0.3, 0.4]
        for ft, fp, w in zip(feats_true, feats_pred, weights):
            loss += w * tf.reduce_mean(tf.abs(ft - fp))
        return loss

    return perceptual_loss
    
def get_images_paths(root_dir):
    root = pathlib.Path(root_dir)
    x_paths = sorted([str(p) for p in root.glob("x*.png")])
    y_paths = sorted([str(p) for p in root.glob("y*.png")])
    m_paths = sorted([str(p) for p in root.glob("m*.png")])

    if len(m_paths) != len(x_paths):
        m_paths = [''] * len(x_paths)

    return np.asarray(x_paths), np.asarray(y_paths), np.asarray(m_paths)
    
def up_down_flip(image, label):
    if tf.random.uniform(shape = [], minval = 0, maxval = 2, dtype = tf.int32) == 1:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
    return image, label

def left_right_flip(image, label):
    if tf.random.uniform(shape = [], minval = 0, maxval = 2, dtype = tf.int32) == 1:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    return image, label

def rotate_90(image, label):
    rand_value = tf.random.uniform(shape = [], minval = 0, maxval = 4, dtype = tf.int32)
    image = tf.image.rot90(image, rand_value)
    label = tf.image.rot90(label, rand_value)
    return image, label

# START: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
def adjust_color(image, label):
    if n_channels == 3: # RGB model only
        
        # tweak colors
        if tf.random.uniform(shape = []) < 0.70:
            ch = tf.random.uniform(shape = [], minval = 0, maxval = 3, dtype = tf.int32)
            m = tf.reduce_min((image, label))
            offset = tf.random.uniform(shape = [])*0.25 - tf.random.uniform(shape = [])*m
            image_r, image_g, image_b = tf.split(image, axis = 2, num_or_size_splits = 3)
            label_r, label_g, label_b = tf.split(label, axis = 2, num_or_size_splits = 3)
            if ch == 0: # messy but I can't get tensor_scatter_nd_update to work
                image_r = image_r + offset*(1. - image_r)
                label_r = label_r + offset*(1. - label_r)
            elif ch == 1:
                image_g = image_g + offset*(1. - image_g)
                label_g = label_g + offset*(1. - label_g)
            else:
                image_b = image_b + offset*(1. - image_b)
                label_b = label_b + offset*(1. - label_b)
            
            image = tf.concat([image_r, image_g, image_b], axis = 2)
            label = tf.concat([label_r, label_g, label_b], axis = 2)
        
        # flip channels
        if tf.random.uniform(shape = []) < 0.70:
            image_ch = tf.split(image, axis = 2, num_or_size_splits = 3)
            label_ch = tf.split(label, axis = 2, num_or_size_splits = 3)
            indices = tf.range(start = 0, limit = tf.shape(image_ch)[0], dtype = tf.int32)
            indices = tf.random.shuffle(indices)
            image = tf.gather(image, indices, axis = 2)
            label = tf.gather(label, indices, axis = 2)
        
        image = tf.clip_by_value(image, 0. , 1.)
        label = tf.clip_by_value(label, 0. , 1.)
    
    return image, label
# END: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License

# START: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
def adjust_brightness(image, label):
    if n_channels == 1: # grayscale model only
        if tf.random.uniform(shape = []) < 0.7:
            m = tf.reduce_min((image, label))
            offset = tf.random.uniform(shape = [])*0.25 - tf.random.uniform(shape = [])*m
            image = image + offset*(1. - image)
            label = label + offset*(1. - label)
            image = tf.clip_by_value(image, 0. , 1.)
            label = tf.clip_by_value(label, 0. , 1.)
    return image, label
# END: The ideas in this portion of code are Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License

def choose_channel(image, label):
    image_r, image_g, image_b = tf.split(image, axis = 2, num_or_size_splits = 3)
    label_r, label_g, label_b = tf.split(label, axis = 2, num_or_size_splits = 3)
    ch = tf.random.uniform(shape = [], minval = 0, maxval = 3, dtype = tf.int32)
    if ch == 0: return image_r, label_r
    elif ch == 1: return image_g, label_g
    else: return image_b, label_b

def to_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)
    label = tf.image.rgb_to_grayscale(label)
    return image, label

def process_path(path_original, path_starless, path_mask):
    img_original = tf.io.read_file(path_original)
    img_original = tf.io.decode_image(img_original, dtype=tf.dtypes.float32, channels=3)
    img_original.set_shape([None, None, 3])
    img_original = tf.image.resize(img_original, [input_size, input_size])

    img_starless = tf.io.read_file(path_starless)
    img_starless = tf.io.decode_image(img_starless, dtype=tf.dtypes.float32, channels=3)
    img_starless.set_shape([None, None, 3])
    img_starless = tf.image.resize(img_starless, [input_size, input_size])

    mask = tf.cond(
        tf.equal(path_mask, ''),
        lambda: tf.ones([input_size, input_size, 1], dtype=tf.float32),
        lambda: tf.image.resize(
            tf.cast(tf.io.decode_png(tf.io.read_file(path_mask), channels=1), tf.float32) / 255.0,
            [input_size, input_size]
        )
    )
    mask.set_shape([input_size, input_size, 1])

    return img_original, img_starless, mask

def random_crop(image, label):
    combined = tf.concat([image, label], axis=2)
    combined_crop = tf.image.random_crop(combined, [patch_size, patch_size, n_channels*2])
    return (combined_crop[:, :, :n_channels], combined_crop[:, :, n_channels:])

def data_generator(X, y, m, batch_size, augmentations=None):
    dataset = tf.data.Dataset.from_tensor_slices((X, y, m))
    dataset = dataset.shuffle(buffer_size=100, reshuffle_each_iteration=True)
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    if augmentations:
        for f in augmentations:
            dataset = dataset.map(
                lambda x, y, m: (lambda r: (r[0], r[1], m))(f(x, y)),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        if n_channels == 1:
            dataset = dataset.map(
                lambda x, y, m: (*choose_channel(x, y), m),
                num_parallel_calls=tf.data.AUTOTUNE
            )
    elif n_channels == 1:
        dataset = dataset.map(
            lambda x, y, m: (*to_grayscale(x, y), m),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
    
def PSNR(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val = 1.)

def SSIM(y_true, y_pred):
    return tf.image.ssim(y_pred, y_true, max_val=1.0)

def inference_single_tile(model, original_image):
    input_image = np.expand_dims(original_image, axis = 0)
    predicted_image = (model.predict(input_image*2. - 1.) + 1.)/2.
    return predicted_image[0]
    
def inference_batch_tiles(model, original_images):
    original_images = np.asarray(original_images, dtype=np.float32)
    preds = []
    for i in range(0, len(original_images), INFER_BATCH_SIZE):
        batch = original_images[i:i + INFER_BATCH_SIZE]
        with tf.device('/CPU:0'):
            pred = model(batch * 2.0 - 1.0, training=False).numpy()
        pred = (pred + 1.0) / 2.0
        preds.append(pred)
        
    return np.concatenate(preds, axis=0)

    
def infer_image(model, original_image, border=0):
    sizeY, sizeX = original_image.shape[:2]
    n_ch = original_image.shape[2] if len(original_image.shape) == 3 else 1

    pad_img = cv2.copyMakeBorder(original_image, border, border, border, border, cv2.BORDER_REFLECT)
    padY, padX = pad_img.shape[:2]

    def make_weight_tile(size):
        sigma = size / 6.0
        ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        gauss = np.exp(-0.5 * (ax / sigma) ** 2)
        tile_w = np.outer(gauss, gauss).astype(np.float32)
        tile_w = tile_w / tile_w.max()
        return tile_w[..., np.newaxis]

    weight_tile = make_weight_tile(patch_size)

    accum = np.zeros((padY, padX, n_ch), dtype=np.float32)
    weight_accum = np.zeros((padY, padX, 1), dtype=np.float32)

    def tile_positions(total, patch, step):
        positions = list(range(0, total - patch, step))
        positions.append(total - patch)
        return positions

    step = patch_size - 2 * border

    row_starts = tile_positions(padY, patch_size, step)
    col_starts = tile_positions(padX, patch_size, step)

    tiles = []
    coords = []
    for r in row_starts:
        for c in col_starts:
            tile = pad_img[r:r+patch_size, c:c+patch_size]
            tiles.append(tile)
            coords.append((r, c))

    # TTA: 8 augmentations (4 rotations x 2 flips = D4 symmetry group)
    def augment(tile, idx):
        """Apply augmentation idx (0-7) to a tile."""
        t = tile.copy()
        if idx >= 4:
            t = np.fliplr(t)
        t = np.rot90(t, k=idx % 4)
        return t

    def deaugment(tile, idx):
        """Reverse augmentation idx to restore original orientation."""
        t = tile.copy()
        t = np.rot90(t, k=-(idx % 4))
        if idx >= 4:
            t = np.fliplr(t)
        return t

    # Run all 8 augmentations and average
    tta_accum = np.zeros((padY, padX, n_ch), dtype=np.float32)
    tta_weight = np.zeros((padY, padX, 1), dtype=np.float32)


    for aug_idx in range(4):
        for idx, (tile, (r, c)) in enumerate(zip(tiles, coords)):
            aug_tile = augment(tile, aug_idx)
            pred = inference_batch_tiles(model, [aug_tile])[0]
            pred_deaug = deaugment(pred, aug_idx)
            tta_accum[r:r+patch_size, c:c+patch_size] += pred_deaug * weight_tile
            tta_weight[r:r+patch_size, c:c+patch_size] += weight_tile
#    for aug_idx in range(4):
#        aug_tiles = [augment(t, aug_idx) for t in tiles]
#        preds = inference_batch_tiles(model, aug_tiles)
#
#        for pred, (r, c) in zip(preds, coords):
#            pred_deaug = deaugment(pred, aug_idx)
#            tta_accum[r:r+patch_size, c:c+patch_size] += pred_deaug * weight_tile
#            tta_weight[r:r+patch_size, c:c+patch_size] += weight_tile

    blended = tta_accum / (tta_weight + 1e-8)
    result = blended[border:padY-border, border:padX-border]
    return np.clip(result, 0.0, 1.0)
    
#def infer(file = ''):
#    tf.keras.backend.clear_session()
#    print('building generator...')
#    model = generator()
#    # model.summary()
    
#    if os.path.exists(generator_fname):
#        print('loading weights...')
#        model.load_weights(generator_fname)
#    else: return
    
#    # process image from disk
#    if len(file) > 0 and os.path.exists(file):
#        original_image = cv2.imread(file)
#        original_image = prepare_image(original_image)
#        save_image('starry.png', original_image)
#        start_time = time.time()
#        starless_image = infer_image(model, original_image, border = border)
#        print("--- %s seconds ---" % (time.time() - start_time))
#        save_image('starless.png', starless_image)
#        
#    else: # process image from dataset
#        original_array_paths, starless_array_paths = get_images_paths(os.getcwd() + os.sep + "nox data") # training and test image paths
#        i = np.random.randint(0, original_array_paths.shape[0]) # get a random noisy image index
#        original_image = cv2.imread(original_array_paths[i])
#        original_image = prepare_image(original_image)
#        save_image('input.png', original_image)
#        gt_image = cv2.imread(starless_array_paths[i])
#        gt_image = prepare_image(gt_image)
#        save_image('gt.png', gt_image)
#        starless_image = infer_image(model, original_image, border = border)
#        save_image('output.png', starless_image)
#        psnr1 = PSNR(original_image, gt_image).numpy()
#        psnr2 = PSNR(starless_image, gt_image).numpy()
#        print('PSNR(input, gt) = %f'%psnr1)
#        print('PSNR(output, gt) = %f'%psnr2)

def infer(path_in=None, out_dir=None):
    """
    Full-resolution star removal using tiled inference.
    Works on a single image or a folder.
    """

    print("building generator...")
    G = generator()
    
    gen_weights = "generator_color_resnet.weights.h5" if n_channels == 3 else "generator_gray_resnet.weights.h5"
    if not os.path.exists(gen_weights):
        print(f"Could not find generator weights: {gen_weights}")
        return

    print("loading weights...")
    G.load_weights(gen_weights)

    # collect input files
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    if path_in is None:
        print("infer() needs a file or folder path.")
        return

    if os.path.isdir(path_in):
        files = [os.path.join(path_in, f) for f in os.listdir(path_in) if f.lower().endswith(exts)]
        files.sort()
        if not files:
            print(f"No images found in folder: {path_in}")
            return
    else:
        files = [path_in]

    # output directory
    if out_dir is None:
        out_dir = path_in if os.path.isdir(path_in) else os.path.dirname(path_in)
    if out_dir == "":
        out_dir = os.getcwd()

    os.makedirs(out_dir, exist_ok=True)
    print(f"writing outputs to: {out_dir}")

    # run tiled inference
    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        out_path = os.path.join(out_dir, base + "_starless.png")

        print(f"processing: {fp}")

        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"could not read: {fp}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0

        pred01 = infer_image(G, rgb, border=border)
        pred01 = np.clip(pred01, 0.0, 1.0)

        out_bgr = (pred01 * 255.0).astype("uint8")
        out_bgr = cv2.cvtColor(out_bgr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, out_bgr)

        print(f"saved: {out_path}")


def train():
    tf.keras.backend.clear_session()
    
    # training and test image paths
    original_array_paths, starless_array_paths, mask_array_paths = get_images_paths(os.getcwd() + os.sep + "nox data")
    
    if validation:
        original_train_paths, original_test_paths, \
        starless_train_paths, starless_test_paths, \
        mask_train_paths, mask_test_paths = train_test_split(
            original_array_paths, starless_array_paths, mask_array_paths,
            test_size=0.2, random_state=42)
    else:
        original_train_paths = original_array_paths
        starless_train_paths = starless_array_paths
        mask_train_paths = mask_array_paths
    
    # nets
    print('building generator...')
    G = generator()
    # G.summary()
    print('building discriminator...')
    D = discriminator()
    # D.summary()
    print('building VGG perceptual loss...')
    vgg_loss_fn = build_vgg_loss()
    
    # optimizers
    global lr
    on_lr = 0 # number of epochs at present lr
    gen_optimizer = tf.optimizers.Adam(learning_rate = lr)
    dis_optimizer = tf.optimizers.Adam(learning_rate = lr/4)
    
    hist_train = {}
    hist_val = {}
    
    # load weights
    e0 = -1
    
    generator_fname = "generator_color_resnet.weights.h5"
    discriminator_fname = "discriminator_color_resnet.weights.h5"

    if os.path.exists(generator_fname) \
        and os.path.exists(discriminator_fname) \
        and os.path.exists(os.getcwd() + os.sep + 'history.pkl'):
        print('loading weights...')
        try:
            G.load_weights(generator_fname)
            D.load_weights(discriminator_fname)
        except Exception as e:
            print(f'weights.h5 load failed ({e}), trying checkpoint...')
            G_checkpoint = tf.train.Checkpoint(model=G)
            G_checkpoint.restore(generator_ckpt + '-1').expect_partial()
            D_checkpoint = tf.train.Checkpoint(model=D)
            D_checkpoint.restore(discriminator_ckpt + '-1').expect_partial()
            print('loaded from checkpoint successfully')
        G_checkpoint = tf.train.Checkpoint(model = G, optim = gen_optimizer)
        G_checkpoint.restore(generator_ckpt + '-1')
        D_checkpoint = tf.train.Checkpoint(model = D, optim = dis_optimizer)
        D_checkpoint.restore(discriminator_ckpt + '-1')
        with open('history.pkl', 'rb') as f:
            print('loading training history...')
            if validation:
                e0, on_lr, lr, hist_train, hist_val, original_train_paths, original_test_paths, starless_train_paths, starless_test_paths = pickle.load(f)
                print("\repoch %d: loss %f PSNR %f SSIM %f    val: loss %f PSNR %f SSIM %f                    \n"%(e0, hist_train['total'][-1], hist_train['psnr'][-1], hist_train.get('ssim', [0.0])[-1], hist_val['total'][-1], hist_val['psnr'][-1], hist_val.get('ssim', [0.0])[-1]), end = '')
            else: 
                e0, on_lr, lr, hist_train, original_train_paths, starless_train_paths = pickle.load(f)
                print("\repoch %d: loss %f PSNR %f SSIM %f                    \n"%(e0, hist_train['total'][-1], hist_train['psnr'][-1], hist_train.get('ssim', [0.0])[-1]), end = '')
    
    if validation:
        print('number of original images for training = %i'%original_train_paths.shape)
        print('number of original images for testing = %i'%original_test_paths.shape)
        print('number of starless images for training = %i'%starless_train_paths.shape)
        print('number of starless images for testing = %i'%starless_test_paths.shape)
    else:
        print('number of original images for training = %i'%original_train_paths.shape)
        print('number of starless images for training = %i'%starless_train_paths.shape)
        
    # number of epochs and steps
    print('number of training epochs = %i'%epochs)
    steps_per_epoch_train = int(len(original_train_paths)/BATCH_SIZE)
    print('steps per training epoch = %i'%steps_per_epoch_train)
    steps_per_epoch_validation = 1
    if validation:
        steps_per_epoch_validation = int(len(original_test_paths)/BATCH_SIZE)
        ema_val = 1. - (1. - ema)*steps_per_epoch_train/steps_per_epoch_validation
        print('steps per validation epoch = %i'%steps_per_epoch_validation)

    # data generators
    augmentation_lst = [left_right_flip, up_down_flip, rotate_90, adjust_color, adjust_brightness]
    image_generator_train = data_generator(X=original_train_paths, y=starless_train_paths, m=mask_train_paths, batch_size=BATCH_SIZE, augmentations=augmentation_lst)
    train_iter = iter(image_generator_train)
    image_generator_test = []
    if validation:
        image_generator_test = data_generator(X=original_test_paths, y=starless_test_paths, m=mask_test_paths, batch_size=BATCH_SIZE)
    
    @tf.function
    def train_step(e, x, y, mask):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            gen_output = G(x)

            p1_real, p2_real, p3_real, p4_real, p5_real, predict_real = D([x, y])
            p1_fake, p2_fake, p3_fake, p4_fake, p5_fake, predict_fake = D([x, gen_output])

            d = {}

            dis_loss = tf.reduce_mean(-(tf.math.log(predict_real + 1E-8) + tf.math.log(1. - predict_fake + 1E-8)))
            d['dis_loss'] = dis_loss

            gen_loss_GAN = tf.reduce_mean(-tf.math.log(predict_fake + 1E-8))
            d['gen_loss_GAN'] = gen_loss_GAN

            gen_p1 = tf.reduce_mean(tf.abs(p1_fake - p1_real))
            d['gen_p1'] = gen_p1
            gen_p2 = tf.reduce_mean(tf.abs(p2_fake - p2_real))
            d['gen_p2'] = gen_p2
            gen_p3 = tf.reduce_mean(tf.abs(p3_fake - p3_real))
            d['gen_p3'] = gen_p3
            gen_p4 = tf.reduce_mean(tf.abs(p4_fake - p4_real))
            d['gen_p4'] = gen_p4
            gen_p5 = tf.reduce_mean(tf.abs(p5_fake - p5_real))
            d['gen_p5'] = gen_p5

            pixel_weights = 1.0 + mask * 4.0  # star regions weighted 10x, background 1x
            gen_L1 = tf.reduce_mean(pixel_weights * tf.abs(y - gen_output))
            d['gen_L1'] = gen_L1 * 20.

            # Phase-scheduled loss weights — all computed as tensors so @tf.function traces cleanly
            e_f = tf.cast(e, tf.float32)

            # GAN: zero until epoch 20, ramps to 0.3 over 50 epochs
            gan_weight = tf.minimum(tf.maximum(e_f - 20.0, 0.0) / 50.0, 0.3)

            # VGG: zero until epoch 40, ramps to 0.1 over 30 epochs
            vgg_weight = tf.minimum(tf.maximum(e_f - 40.0, 0.0) / 30.0, 0.1)

            # SSIM: zero until epoch 60, ramps to 0.5 over 30 epochs
            ssim_weight = tf.minimum(tf.maximum(e_f - 60.0, 0.0) / 30.0, 0.5)

            gen_vgg = vgg_loss_fn(y, gen_output)
            d['gen_vgg'] = gen_vgg * vgg_weight

            gen_ssim = 1.0 - tf.reduce_mean(SSIM(y, gen_output))
            d['gen_ssim'] = gen_ssim * ssim_weight

            d['psnr'] = tf.reduce_mean(PSNR(y, gen_output))
            d['ssim'] = tf.reduce_mean(SSIM(y, gen_output))

            gen_loss = (gen_loss_GAN * gan_weight + gen_p1 * 1.0 + gen_p2 * 5.0 +
                        gen_p3 * 5.0 + gen_p4 * 5.0 + gen_p5 * 5.0 +
                        gen_L1 * 20.0 + gen_vgg * vgg_weight * 3.0 + gen_ssim * ssim_weight)
            d['total'] = gen_loss

        if e > 0:
            gen_grads = gen_tape.gradient(gen_loss, G.trainable_variables)
            gen_grads, _ = tf.clip_by_global_norm(gen_grads, 0.5)
            gen_optimizer.apply_gradients(zip(gen_grads, G.trainable_variables))

            dis_grads = dis_tape.gradient(dis_loss, D.trainable_variables)
            dis_grads, _ = tf.clip_by_global_norm(dis_grads, 0.5)
            dis_optimizer.apply_gradients(zip(dis_grads, D.trainable_variables))

        return d
        # END: This portion of code is Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License
        
    @tf.function
    def test_step(x, y, mask):
        gen_output = G(x)

        p1_real, p2_real, p3_real, p4_real, p5_real, predict_real = D([x, y])
        p1_fake, p2_fake, p3_fake, p4_fake, p5_fake, predict_fake = D([x, gen_output])

        d = {}

        dis_loss = tf.reduce_mean(-(tf.math.log(predict_real + 1E-8) + tf.math.log(1. - predict_fake + 1E-8)))
        d['dis_loss'] = dis_loss

        gen_loss_GAN = tf.reduce_mean(-tf.math.log(predict_fake + 1E-8))
        d['gen_loss_GAN'] = gen_loss_GAN

        gen_p1 = tf.reduce_mean(tf.abs(p1_fake - p1_real))
        d['gen_p1'] = gen_p1
        gen_p2 = tf.reduce_mean(tf.abs(p2_fake - p2_real))
        d['gen_p2'] = gen_p2
        gen_p3 = tf.reduce_mean(tf.abs(p3_fake - p3_real))
        d['gen_p3'] = gen_p3
        gen_p4 = tf.reduce_mean(tf.abs(p4_fake - p4_real))
        d['gen_p4'] = gen_p4
        gen_p5 = tf.reduce_mean(tf.abs(p5_fake - p5_real))
        d['gen_p5'] = gen_p5

        pixel_weights = 1.0 + mask * 4.0
        gen_L1 = tf.reduce_mean(pixel_weights * tf.abs(y - gen_output))
        d['gen_L1'] = gen_L1 * 20.

        gen_vgg = vgg_loss_fn(y, gen_output)
        d['gen_vgg'] = gen_vgg * 0.1

        gen_ssim = 1.0 - tf.reduce_mean(SSIM(y, gen_output))
        d['gen_ssim'] = gen_ssim * 0.5

        d['psnr'] = tf.reduce_mean(PSNR(y, gen_output))
        d['ssim'] = tf.reduce_mean(SSIM(y, gen_output))

        # test_step uses fixed max weights (no warmup needed for validation)
        gen_loss = (gen_loss_GAN * 0.3 + gen_p1 * 1.0 + gen_p2 * 5.0 +
                    gen_p3 * 5.0 + gen_p4 * 5.0 + gen_p5 * 5.0 +
                    gen_L1 * 20.0 + gen_vgg * 0.1 + gen_ssim * 0.5)
        d['total'] = gen_loss

        return d
    # END: This portion of code is Copyright (c) 2018-2019 Nikita Misiura and used here under the MIT License

    # training epochs with validation
    for e in range(e0 + 1, epochs + 1, 1):
        # Unfreeze top ResNet layers after some training
        if e == 80:
            gen_optimizer.learning_rate.assign(1e-6)
            dis_optimizer.learning_rate.assign(1e-6 / 4)
            print("Reduced lr to 5e-6 for backbone fine-tuning")
            # ResNet layers are embedded directly — unfreeze conv5 blocks only
            resnet_layer_names = [l.name for l in G.layers if l.name.startswith('conv5')]
            unfrozen = 0
            for layer in G.layers:
                if layer.name.startswith('conv5'):
                    layer.trainable = True
                    unfrozen += 1
                elif layer.name.startswith('conv4_block6') or layer.name.startswith('conv4_block5'):
                    layer.trainable = True
                    unfrozen += 1
            print(f"Unfroze {unfrozen} upper ResNet50V2 layers for fine-tuning")
        
        # training batches
        pbar = tqdm(range(steps_per_epoch_train), desc=f"Epoch {e}/{epochs} [train]", leave=False)
        for i in pbar:
            x, y, mask = next(train_iter)
            x = x*2. - 1.
            y = y*2. - 1.

            d = train_step(tf.constant(e), x, y, mask)
            # Update progress bar every 10 steps (reduce overhead)
            if i % 10 == 0:
                pbar.set_postfix({
                "loss": float(d["total"].numpy()),
                "psnr": float(d["psnr"].numpy()),
                "D": float(d["dis_loss"].numpy()),
                "GAN": float(d["gen_loss_GAN"].numpy()),
            })

            for k in d:
                if k in hist_train.keys():
                    hist_train[k].append((d[k]*(1. - ema) + hist_train[k][-1]*ema).numpy())
                else:
                    hist_train[k] = [d[k].numpy()]
            
            if 'loss_us' in hist_train.keys(): # include unsmoothed loss and psnr in logs
                hist_train['loss_us'].append(d['total'].numpy())
                hist_train['psnr_us'].append(d['psnr'].numpy())
                hist_train['ssim_us'].append(d['ssim'].numpy())
            else:
                hist_train['loss_us'] = [d['total'].numpy()]
                hist_train['psnr_us'] = [d['psnr'].numpy()]
                hist_train['ssim_us'] = [d['ssim'].numpy()]
            
        if e == 0: # force retracing after initial pass (so that gradients will be applied in future)
            tf.config.run_functions_eagerly(True)
            tf.config.run_functions_eagerly(False)
        
        # validation batches
        if validation:
            val_iter = iter(image_generator_test)
            vbar = tqdm(range(steps_per_epoch_validation), desc=f"Epoch {e}/{epochs} [val]", leave=False)
            for i in vbar:
                x, y, mask = next(val_iter)
                x = x*2. - 1.
                y = y*2. - 1.
                d = test_step(x, y, mask)

                if i % 10 == 0:
                    vbar.set_postfix({
                    "val_loss": float(d["total"].numpy()),
                    "val_psnr": float(d["psnr"].numpy()),
                })

                for k in d:
                    if k in hist_val.keys():
                        hist_val[k].append((d[k]*(1. - ema_val) + hist_val[k][-1]*ema_val).numpy())
                    else:
                        hist_val[k] = [d[k].numpy()]
                        
                if 'val_loss_us' in hist_val.keys(): # include unsmoothed loss and psnr in logs
                    hist_val['val_loss_us'].append(d['total'].numpy())
                    hist_val['val_psnr_us'].append(d['psnr'].numpy())
                    hist_val['val_ssim_us'].append(d['ssim'].numpy())
                else:
                    hist_val['val_loss_us'] = [d['total'].numpy()]
                    hist_val['val_psnr_us'] = [d['psnr'].numpy()]
                    hist_val['val_ssim_us'] = [d['ssim'].numpy()]
            
            print("\repoch %d: loss %f PSNR %f SSIM %f    val: loss %f PSNR %f SSIM %f                    \n"%(e, hist_train['total'][-1], hist_train['psnr'][-1], hist_train['ssim'][-1], hist_val['total'][-1], hist_val['psnr'][-1], hist_val['ssim'][-1]), end = '')
        else:
            print("")
        
        if e % save_freq == 0:
            if not generator_fname.endswith(".weights.h5"):
                 generator_fname += ".weights.h5"
            if not discriminator_fname.endswith(".weights.h5"):
                 discriminator_fname += ".weights.h5"

            G.save_weights(generator_fname)
            D.save_weights(discriminator_fname)
            G_checkpoint = tf.train.Checkpoint(model = G, optim = gen_optimizer)
            G_checkpoint.save(generator_ckpt)
            D_checkpoint = tf.train.Checkpoint(model = D, optim = dis_optimizer)
            D_checkpoint.save(discriminator_ckpt)
            if validation:
                with open('history.pkl', 'wb') as f:
                    pickle.dump([e, on_lr, lr, hist_train, hist_val, original_train_paths, original_test_paths, starless_train_paths, starless_test_paths], f)
            else:
                with open('history.pkl', 'wb') as f:
                    pickle.dump([e, on_lr, lr, hist_train, original_train_paths, starless_train_paths], f)
        
            # CSV logger
            with open('history.csv', 'a', newline = '') as csv:
                for i in reversed(range((save_freq, 1)[e == 0])):
                    if validation:
                        if e == 0 and os.path.getsize('history.csv') == 0: csv.write('epoch,loss_ema,...\n')
                        csv.write('%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n'%
                            (e - i,
                            hist_train['total'][-1 - i*steps_per_epoch_train],
                            hist_train['psnr'][-1 - i*steps_per_epoch_train],
                            hist_train['ssim'][-1 - i*steps_per_epoch_train],
                            hist_val['total'][-1 - i*steps_per_epoch_validation],
                            hist_val['psnr'][-1 - i*steps_per_epoch_validation],
                            hist_val['ssim'][-1 - i*steps_per_epoch_validation],
                            hist_train['loss_us'][-1 - i*steps_per_epoch_train],
                            hist_train['psnr_us'][-1 - i*steps_per_epoch_train],
                            hist_train['ssim_us'][-1 - i*steps_per_epoch_train],
                            hist_val['val_loss_us'][-1 - i*steps_per_epoch_validation],
                            hist_val['val_psnr_us'][-1 - i*steps_per_epoch_validation],
                            hist_val['val_ssim_us'][-1 - i*steps_per_epoch_validation],
                            ))
                    else:
                        if e == 0: csv.write('epoch,loss_ema,psnr_ema,ssim_ema,loss_us,psnr_us,ssim_us,lr\n')
                        csv.write('%d,%f,%f,%f,%f,%f,%f,%f\n'%
                            (e - i,
                            hist_train['total'][-1 - i*steps_per_epoch_train],
                            hist_train['psnr'][-1 - i*steps_per_epoch_train],
                            hist_train['ssim'][-1 - i*steps_per_epoch_train],
                            hist_train['loss_us'][-1 - i*steps_per_epoch_train],
                            hist_train['psnr_us'][-1 - i*steps_per_epoch_train],
                            hist_train['ssim_us'][-1 - i*steps_per_epoch_train],
                            lr))
            
            # plotting
            if validation and os.path.exists('history.csv') and os.path.getsize('history.csv') > 0:
                df = pd.read_csv('history.csv')
                if 'epoch' not in df.columns:
                    pass
                else:
                    df = df[pd.to_numeric(df['epoch'], errors='coerce').notna()]
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.dropna(how='all')  # only drop rows that are entirely NaN
                # Skip plotting if dataframe is empty or missing expected columns
                if df.empty or 'loss_us' not in df.columns:
                    pass
                else:
                    plt.close('all')
                    fig, ax = plt.subplots(1, 3, sharex = True)
                    line1, = ax[0].plot(df.epoch, df.loss_us, 'b', label = 'training unsmoothed', linewidth = 0.5)
                    line2, = ax[0].plot(df.epoch, df.val_loss_us, 'r', label = 'validation unsmoothed', linewidth = 0.5)
                    line3, = ax[0].plot(df.epoch, df.loss_ema, 'b', label = 'training ema decay = %.4f'%ema)
                    line4, = ax[0].plot(df.epoch, df.val_loss_ema, 'r', label = 'validation ema decay = %.4f'%ema_val)
                    ax[0].set_xlabel('epoch', fontsize = 12)
                    ax[0].set_ylabel('loss', fontsize = 12)
                    line1, = ax[1].plot(df.epoch, df.psnr_us, 'b', label = 'training unsmoothed', linewidth = 0.5)
                    line2, = ax[1].plot(df.epoch, df.val_psnr_us, 'r', label = 'validation unsmoothed', linewidth = 0.5)
                    line3, = ax[1].plot(df.epoch, df.psnr_ema, 'b', label = 'training ema decay = %.4f'%ema)
                    line4, = ax[1].plot(df.epoch, df.val_psnr_ema, 'r', label = 'validation ema decay = %.4f'%ema_val)
                    ax[1].set_xlabel('epoch', fontsize = 12)
                    ax[1].set_ylabel('psnr (dB)', fontsize = 12)
                    line1, = ax[2].plot(df.epoch, df.ssim_us, 'b', label = 'training unsmoothed', linewidth = 0.5)
                    line2, = ax[2].plot(df.epoch, df.val_ssim_us, 'r', label = 'validation unsmoothed', linewidth = 0.5)
                    line3, = ax[2].plot(df.epoch, df.ssim_ema, 'b', label = 'training ema decay = %.4f'%ema)
                    line4, = ax[2].plot(df.epoch, df.val_ssim_ema, 'r', label = 'validation ema decay = %.4f'%ema_val)
                    ax[2].set_xlabel('epoch', fontsize = 12)
                    ax[2].set_ylabel('ssim', fontsize = 12)
                    fig.legend(handles = [line1, line2, line3, line4], loc = 'upper center', bbox_to_anchor = (0.5, 0), fancybox = True, ncol = 2)
                    fig.tight_layout()
                    plt.savefig(os.getcwd() + os.sep + 'training.png', bbox_inches = 'tight')
                if not validation and os.path.exists('history.csv'):
                    df = pd.read_csv('history.csv')
                    df = df[pd.to_numeric(df['epoch'], errors='coerce').notna()]
                    df = df.apply(pd.to_numeric, errors='coerce')
                    df = df.dropna(how='all')
                    if df.empty or 'loss_us' not in df.columns:
                        pass
                    else:
                        plt.close('all')
                        fig, ax = plt.subplots(1, 2, sharex = True)
                        line1, = ax[0].plot(df.epoch, df.loss_us, 'b', label = 'training unsmoothed', linewidth = 0.5)
                        line2, = ax[0].plot(df.epoch, df.loss_ema, 'b', label = 'training ema decay = %.4f'%ema)
                        ax[0].legend(handles=[line1, line2])
                        ax[0].set_xlabel('epoch', fontsize = 12)
                        ax[0].set_ylabel('loss', fontsize = 12)
                        line1, = ax[1].plot(df.epoch, df.psnr_us, 'b', label = 'training unsmoothed', linewidth = 0.5)
                        line2, = ax[1].plot(df.epoch, df.psnr_ema, 'b', label = 'training ema decay = %.4f'%ema)
                        ax[1].legend(handles = [line1, line2])
                        ax[1].set_xlabel('epoch', fontsize = 12)
                        ax[1].set_ylabel('psnr (dB)', fontsize = 12)
                        fig.legend(handles = [line1, line2], loc = 'upper center', bbox_to_anchor = (0.5, 0), fancybox = True, ncol = 2)
                        fig.tight_layout()
                        plt.savefig(os.getcwd() + os.sep + 'training.png', bbox_inches = 'tight')
#                elif ~validation and os.path.exists('history.csv'):
#                    df = pd.read_csv('history.csv')
#                    fig, ax = plt.subplots(1, 2, sharex = True)
#                    line1, = ax[0].plot(df.epoch, df.loss_us, 'b', label = 'training unsmoothed', linewidth = 0.5)
#                    line2, = ax[0].plot(df.epoch, df.loss_ema, 'b', label = 'training ema decay = %.4f'%ema)
#                    ax[0].legend(handles=[line1, line2])
#                    ax[0].set_xlabel('epoch', fontsize = 12)
#                    ax[0].set_ylabel('loss', fontsize = 12)
#                    line1, = ax[1].plot(df.epoch, df.psnr_us, 'b', label = 'training unsmoothed', linewidth = 0.5)
#                    line2, = ax[1].plot(df.epoch, df.psnr_ema, 'b', label = 'training ema decay = %.4f'%ema)
#                    ax[1].legend(handles = [line1, line2])
#                    ax[1].set_xlabel('epoch', fontsize = 12)
#                    ax[1].set_ylabel('psnr (dB)', fontsize = 12)
#                    fig.legend(handles = [line1, line2], loc = 'upper center', bbox_to_anchor = (0.5, 0), fancybox = True, ncol = 2)
#                    fig.tight_layout()
#                    plt.savefig(os.getcwd() + os.sep + 'training.png', bbox_inches = 'tight')
            
            if validation:
                starless_dir = os.getcwd() + os.sep + 'starless'
                if not os.path.exists(starless_dir):
                    os.makedirs(starless_dir)
                
                # process random image from data set
                i = np.random.randint(0, original_array_paths.shape[0]) # get a random image index
                original_image = cv2.imread(original_array_paths[i])
                original_image = prepare_image(original_image)
                save_image('input.png', original_image)
                gt_image = cv2.imread(starless_array_paths[i])
                gt_image = prepare_image(gt_image)
                save_image('gt.png', gt_image)
                starless_image = infer_image(G, original_image, border = border)
                save_image('output.png', starless_image)
                psnr1 = PSNR(original_image, gt_image).numpy()
                psnr2 = PSNR(starless_image, gt_image).numpy()
                print('PSNR(input, gt) = %f'%psnr1)
                print('PSNR(output, gt) = %f'%psnr2)
        
        # reduce learning rate on increasing loss
        if on_lr > cooldown:
            last_loss = []
            for i in hist_train['total'][::-steps_per_epoch_train]: # check training losses
                if len(last_loss) == 0 or i < last_loss[-1]: last_loss.append(i)
                else: break
                if len(last_loss) > patience:
                    lr/=2
                    gen_optimizer.learning_rate.assign(lr)
                    dis_optimizer.learning_rate.assign(lr/4)
                    on_lr = -1
                    break
                
            last_val_loss = []
            if on_lr != -1 and validation:
                for i in hist_val['total'][::-steps_per_epoch_validation]: # check validation losses
                    if len(last_val_loss) == 0 or i < last_val_loss[-1]: last_val_loss.append(i)
                    else: break
                    if len(last_val_loss) > patience:
                        lr/=2
                        gen_optimizer.learning_rate.assign(lr)
                        dis_optimizer.learning_rate.assign(lr/4)
                        on_lr = -1
                        break
        on_lr += 1

        # housekeeping
        gc.collect()
        tf.keras.backend.clear_session()
        
def prepare_image(image): # convert BGR image to RGB or gray 0..1 float image
    if image.dtype == 'uint8': image = image/255. # convert uint8 to 0..1 float
    if image.dtype == 'float64': image = image.astype(np.float32)
    if image.dtype == 'float32' and np.max(image) > 1.: image = image/255. # convert 0..255 float to 0..1 float
    if len(image.shape) == 3 and n_channels == 1: # if color but should be gray
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:, :, np.newaxis]
    elif len(image.shape) == 3 and n_channels == 3: # if color and should be color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else: # if gray
        image = image[:, :, np.newaxis]
    return image
    
def save_image(name, image): # save 0..1 float RGB or gray image
    if image.dtype == 'float64': image = image.astype(np.float32)
    if len(image.shape) == 3: image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # if color, convert to BGR
    if image.dtype == 'float32': image *= 255. # floats at this point will always be in range 0..1 so convert to 0..255 float
    cv2.imwrite(name, image)

def save_weights():
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    # create model
    tf.keras.backend.clear_session() # Release global state of old models and layers
    model = generator()
    model.summary()
    
    # load weights
    if os.path.exists(generator_fname):
        print('loading weights...')
        model.load_weights(generator_fname)
    else: return
    
    full_model = tf.function(lambda inputs: model(inputs))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def = frozen_func.graph,
        logdir = os.getcwd(),
        name = ('noxGeneratorGrayscale.pb', 'noxGeneratorColor.pb')[n_channels == 3],
        as_text = False)
    
    tf.saved_model.save(model, './tensorflow')
    
    if n_channels == 3:
        os.system('python -m tf2onnx.convert --saved-model ./tensorflow --output noxGeneratorColor.onnx')
        os.system('onnxsim noxGeneratorColor.onnx noxGeneratorColor.onnx --overwrite-input-shape 1,512,512,3')
    else:
        os.system('python -m tf2onnx.convert --saved-model ./tensorflow --output noxGeneratorGrayscale.onnx')
        os.system('onnxsim noxGeneratorGrayscale.onnx noxGeneratorGrayscale.onnx --overwrite-input-shape 1,512,512,3')

def evaluate(data_dir=None, max_items=500, use_masks=False):  # Evaluates current data
    print("building generator...")
    G = generator()

    gen_weights = "generator_color_resnet.weights.h5" if n_channels == 3 else "generator_gray_resnet.weights.h5"
    if os.path.exists(gen_weights):
        print("loading weights...")
        G.load_weights(gen_weights)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        lpips_fn.eval()

        def tf01_to_torch_m11(img_tf01):
            if isinstance(img_tf01, tf.Tensor):
                img_np = img_tf01.numpy()
            else:
                img_np = img_tf01
            t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
            t = t * 2.0 - 1.0
            return t.to(device)
    else:
        print(f"Could not find generator weights: {gen_weights}")
        return

    if data_dir is None:
        data_dir = os.getcwd() + os.sep + "nox data" + os.sep
    else:
        # ensure trailing separator for globbing consistency in your codebase
        if not data_dir.endswith(os.sep):
            data_dir = data_dir + os.sep

    x_paths, y_paths, _ = get_images_paths(data_dir)
    n = min(len(x_paths), len(y_paths), max_items)
    if n == 0:
        print(f"No x/y pairs found in: {data_dir}")
        return

    psnr_vals = []
    ssim_vals = []
    halo_vals = []
    lpips_vals = []
    
    for i in range(n):
        # process_path returns float32 in [0,1] and resizes to [input_size, input_size]
        x, y, _ = process_path(x_paths[i], y_paths[i], tf.constant(''))

        # match training scaling
        x_in = (x * 2.0) - 1.0
        y_gt = (y * 2.0) - 1.0

        pred = G(tf.expand_dims(x_in, 0), training=False)[0]  # [-1,1]
        pred01 = (pred + 1.0) / 2.0
        y01 = (y_gt + 1.0) / 2.0

        psnr_vals.append(float(tf.image.psnr(pred01, y01, max_val=1.0).numpy()))
        ssim_vals.append(float(tf.image.ssim(pred01, y01, max_val=1.0).numpy()))
        with torch.no_grad():
            pred_t = tf01_to_torch_m11(pred01)
            gt_t = tf01_to_torch_m11(y01)
            lp = lpips_fn(pred_t, gt_t).item()
        lpips_vals.append(lp)
        
        if use_masks:
            m_path = os.path.join(data_dir, f"m{i}.png")
            if tf.io.gfile.exists(m_path):
                m = tf.io.decode_image(tf.io.read_file(m_path), dtype=tf.float32, channels=1)
                m = tf.image.resize(m, [input_size, input_size])
                resid = tf.abs(pred01 - y01)
                halo = tf.reduce_sum(resid * m) / (tf.reduce_sum(m) + 1e-6)
                halo_vals.append(float(halo.numpy()))

    print(f"[EVAL] folder={data_dir}")
    print(f"[EVAL] N={n}  PSNR={np.mean(psnr_vals):.3f} dB  SSIM={np.mean(ssim_vals):.4f}  LPIPS={np.mean(lpips_vals):.4f}")
    if use_masks and len(halo_vals) > 0:
        print(f"[EVAL] Star-residual={np.mean(halo_vals):.6f} (lower is better)")
    elif use_masks:
        print("[EVAL] Masks not found (m####.png). Star-residual skipped.")

#def infer(path_in=None, out_dir=None):
#    """
#    Run star removal on a single image or on all images in a folder.
#    Saves *_starless.png outputs.
#    """
#
#    print("building generator...")
#    G = generator()
#
#    gen_weights = "generator_color.weights.h5" if n_channels == 3 else "generator_gray.weights.h5"
#    if not os.path.exists(gen_weights):
#        print(f"Could not find generator weights: {gen_weights}")
#        print("Train first, or put the weights file next to nox.py")
#        return
#
#    print("loading weights...")
#    G.load_weights(gen_weights)
#
#    # Collect input files
#    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
#
#    if path_in is None:
#        print(f"saved: {out_path}")
        
if __name__ == "__main__":
    mode = 'train' # train mode by default
    if len(sys.argv) > 1: mode = sys.argv[1]
    
    if mode == 'train':
        train()

    elif mode == 'infer':
        if len(sys.argv) > 2:
            infer(sys.argv[2])  # file name
        else:
            infer()  # draw randomly from dataset

    elif mode == 'eval':
        data_dir = sys.argv[2] if len(sys.argv) > 2 else None
        max_items = int(sys.argv[3]) if len(sys.argv) > 3 else 500
        evaluate(data_dir=data_dir, max_items=max_items, use_masks=True)

    elif mode == 'save':
        save_weights()
