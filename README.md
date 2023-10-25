# vo_lightglue
visual odometry with light glue for feature matching
The code works on KITTI dataset using image from 1 camera. For feature matching FLANN based matcher was used as a baseline and then LightGlue was applied to observe change in accuracy of the odometry inferred from the images

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

