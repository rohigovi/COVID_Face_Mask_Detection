# COVID_Face_Mask_Detection
The unchecked spread of COVID 19 has caused an extraordinary number of deaths across the world that could have easily been prevented. According to the CDC, if everyone in the US wore masks for just 3 weeks, an estimated 130,000 lives could be saved.  

Inspired by this statistic, I created used deep learning techniques to create a Face Mask detector that checks whether or not the people in a live video stream are wearing masks. I made the detector by training a conventional neural network on a dataset of around 1500 "Masked" and "Unmasked" images.


**DEMO**  
![](demo.gif)  

The face detector uses the MobileNetV2 architecture so it’s computationally efficient and easy to deploy the model to embedded systems.  

I used a Caffe-based detector. Brief summary on the Caffe framework - https://caffe.berkeleyvision.org/

​You can also view this project on my website - https://rohitgovindan.wixsite.com/gettoknowme/projects-1

**The general flow of logic is -**  

1.)Train the CNN model on the image dataset.  

2.)Use OpenCV to loop over frames of a video - extract the region of interest(the person's face)  

3.)Apply the face detector to the region of interest and update the resulting message on the video screen.  
  

**HOW TO RUN -**  
1.) Clone the repo.  
2.) Download the dataset from this link and add it to the root of the repo -https://drive.google.com/drive/folders/169WRYNsw-Gw8-glnT2iu35DnpJ7skSIM?usp=sharing  
3.) Train the face detector by running - *python3 detector_training.py --dataset dataset*  
4.) Run the detector on a live video stream - *python3 video_stream_mask_detection.py*  
