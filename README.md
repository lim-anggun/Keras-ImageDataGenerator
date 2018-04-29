# Keras-ImageDataGenerator
This repository contains a modified version of Keras ImageDataGenerator. It generate batches of tensor with real-time data augmentation.
This generator is implemented for foreground segmentation or semantic segmentation.

Please refer to [Keras documentation](https://keras.io/preprocessing/image/#image-preprocessing) for more details.

## Usage
Setting ```class_mode=None```, it returns a tensor of (image, label). 
Initialize paths where images flow from.

```
from keras.preprocessing.image import ImageDataGenerator

batch_size = 1
epoch = 50
h = 360 # image height
w = 480 # image width

# Training path
X_path= os.path.join('camvid', 'train') # input image
Y_path = os.path.join('camvid', 'trainannot') # ground-truth label

# Validation path
val_X_path = os.path.join('camvid', 'val')
val_Y_path = os.path.join('camvid', 'valannot')

```
Create ```train_datagen``` and ```val_datagen``` objects:

```
train_datagen = ImageDataGenerator(
        #shear_range=0.2,
        #zoom_range=0.5,
        #width_shift_range=0.5,
        #height_shift_range=?,
        #rotation_range = 10,
        #horizontal_flip=True,
        fill_mode = 'constant',
        cval = 0., # value to fill input images when fill_mode='constant'
        label_cval = 11. # value to fill labels when fill_mode='constant'
        )
val_datagen = ImageDataGenerator(
        fill_mode = 'constant',
        cval = 0.,
        label_cval = 11.
        )

```

Flow images with corresponding ground-truth labels from given directory: 

```
train_flow = train_datagen.flow_from_directory(
        X_path, Y_path,
        target_size=(h, w),
        batch_size=batch_size,
        shuffle = True,
        #save_to_dir = os.path.join('camvid', 'debugs'), # uncomment to save (image, label) to dir for debuging mode
        #save_prefix = 'd',
        #save_format = 'png',
        class_mode=None
        )

val_flow = val_datagen.flow_from_directory(
        val_X_path, val_Y_path,
        target_size=(h, w),
        batch_size=batch_size,
        shuffle= False,
        #save_to_dir = os.path.join('camvid', 'debugs'),
        #save_prefix = 'd',
        #save_format = 'png',
        class_mode=None
        )
```
Fit the generator:
```

model.fit_generator(train_flow,
                    steps_per_epoch = len(train_flow)/batch_size, 
                    validation_data=val_flow, 
                    validation_steps =len(val_flow)/batch_size,
                    epochs=epochs, 
                    #callbacks=[reduce, tb, early],
                    verbose=1
                    )
```

## Contribution
Any contributions to improve this modification would be appreciated.
