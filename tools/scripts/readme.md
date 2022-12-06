# Note: This folder is the remnants of our first attempt to improve our algorithm.
We did not include this work in the report due to a lack of space, and it not working (at all) for our application.

Using opencv 3.x, it is possible to train custom Viola-Jones like classifiers with annotated positive images and a large set of negative images.
We were able to train a single classifier for one of our water bottles. however, we found that this type of detector does not work well for
objects that heavily vary between different views. This is probably why this type of detector is used for face detection, faces tend to look 
about the same, and its possible for the classifier to determine what features are important in its 24x24 detection window. 

# Training a custom classifier

run `opencv_traincascade -vec bottle_vec.vec -data trained/ -bg bottle_negatives.txt -numStages 5 -w 200 -h 200 -featureType LBP -numPos 58 -numNeg 48`

then to run the cascade classifier

`python src/cascade_classifier.py --camera data/trainingimg/bottle/bottle_positives/20221204_115320.jpg --cascade_file data/trainingimg/bottle/trained/cascade.xml`

## Pipeline example

expected directory layout for this example:
```console
    data/trainingimg:
        multimeter:
            positives:
                ...png
            negatives:
                ...png
            trained:
                ...xml
            annotations.txt
            annotations.vec
            negatives.txt
```


### 1) build positive dataset

`python src/optical_flow_pt.py data/trainingimg/multimeter/IMG_1842.MOV --output_images data/trainingimg/multimeter/positives/ --output_annotations data/trainingimg/multimeter/annotations.txt`

Note: use the *--append* flag when you want to add another set of video images to the dataset. This will append the new filenames to the end of *annotations.txt* rather than overwriting it.

### 2) generate vec file for positives

`cd data/trainingimg/multimeter`

`../../../tools/scripts/make_vec_dataset.sh annotations`

builds the vec annotations file using *annotations.txt* and *positives/* image directory 

### 3) build negatives image set from video and any stills

The *negatives/* directory needs to be populated with images that do not contain the desired image. More variation here helps with training.

Image sets can be extracted from videos:

`python src/mov_to_img.py <input_video_path> <output_dir>`

An optional *--prefix* argument can be used to name output files

`python src/mov_to_img.py data/trainingimg/multimeter/IMG_2145.MOV data/trainingimg/multimeter/negatives/ --prefix IMG_2145_`

### 4) generate listing file for negatives

The listings file needs to contain relative path for each negative.

`cd data/trainingimg/multimeter`

`../../../tools/scripts/make_negative_listing_file.sh negatives`

Will create negatives.txt with appropriate content. Note that *negatives* here is the directory name

### 5) training

`cd data/trainingimg/multimeter`

`mkdir trained`

`opencv_traincascade -vec annotations.vec -data trained/ -bg negatives.txt -numStages 5 -featureType LBP -numPos <npos> -numNeg <nneg> -maxFalseAlarmRate 0.1`

*npos* and *nneg* are the number of positive and negative sample images, respectively.