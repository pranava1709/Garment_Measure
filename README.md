


Large Scale Garment Measurement can be a tedious task and requires a large amount of manpower. This will not only result in loss of accuracy but also increase the expenditure. This is one of the major problems of the Garment Industry and requires an automated solution. 

Landmark Detection is a technique that has seen major success in recent years, especially in pose estimation. Our System is basically an extension of HRNET - Human Pose Estimation Network to design a complete real-time system that extracts the coordinates of the Keypoints in real-time and measures the distance.

For the landmark detection part, we have used https://github.com/lzhbrian/deepfashion2-kps-agg-finetune

We trained our model on the Deep Fashion2 Images Dataset, which contains images of 10 Categories of Images  We used the Aggregation as our training method, which resulted in quite better than the existing state-of-the-art model for Clothes Landmark Detection.

After, the training part, we inerenced the model on Garment Images, accessed from an Industrial Outlet. Then we shifted to the development phase of the coordinate extraction pipeline of the each plotted key point.

Next, we optimized the detected keypoints and then went on with the distance measurement between two opposite keypoints.


![vis128B](https://user-images.githubusercontent.com/60814171/217742683-246691c5-4891-43e2-8938-ff2849e564c5.png)
![image](https://github.com/pranava1709/Garment_Measure/assets/60814171/cc13eccd-4b6a-47b4-9fe0-79bd9463db9b)
![image](https://github.com/pranava1709/Garment_Measure/assets/60814171/81d4cdad-4947-4f82-be18-147547755342)

