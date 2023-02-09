


Large Scale Garment Measurement can be a tedious task and requires large amount of man power. This will not only result in loss of accuracy but also increase the expenditure. This is one of the major problem of the Garment Industry and requires an automated solution. 

Landmark Detection has been a technique which has seen major success in recent years, especially in pose estimation. Our System is basically an extension of HRNET - Human Pose Estimation Network to design a complete real time system that extracts the coordinates of the Keypoints in real time and measures the distance.

For the landmark detection part, we have used https://github.com/lzhbrian/deepfashion2-kps-agg-finetune

We trained our model on Deep Fashion2 Images Dataset, that contains images of 10 Categories of Images  We used the Aggregation as our training method, which resulted quite better than existing state of the art model for Clothes Landmark Detection.

After, the training part, we inferenced the model on Garment Images, accessed from an Industrial Outlet. Then we shifted to the development phase of the coordinate extraction pipeline of the each plotted Keypoint.

Next, we optimized the detected keypoints, and then went on with the distance measurement between two opposite keypoints.
