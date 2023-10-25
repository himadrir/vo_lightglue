# vo_lightglue
Visual Odometry with light glue for local feature matching and [SuperPoint](https://github.com/rpautrat/SuperPoint) for feature extraction.
* The code works on KITTI dataset using image from 1 camera.
* For feature matching FLANN based matcher was used as a baseline and then LightGlue was applied to observe change in accuracy of the odometry inferred from the images.
* With LightGlue as the matching algorithm, [SuperPoint](https://github.com/rpautrat/SuperPoint) was used as feature extraction method as recommended by LightGlue for maximum accuracy and performance.
* 10 FPS output was obtained when processing the frame compared to 18-22 FPS on FLANN based matcher, however a drop in error in odometric measurements was seen( 1% for LightGlue and FLANN produced an error of 3.5-4.1% for the dataset.

## Dependencies: ##
1. [LightGlue](https://github.com/cvg/LightGlue) 
2. tqdm
3. torch >= 1.9.0


## Citations: ##
```
@inproceedings{lindenberger2023lightglue,
  author    = {Philipp Lindenberger and
               Paul-Edouard Sarlin and
               Marc Pollefeys},
  title     = {{LightGlue: Local Feature Matching at Light Speed}},
  booktitle = {ICCV},
  year      = {2023}
}

```

