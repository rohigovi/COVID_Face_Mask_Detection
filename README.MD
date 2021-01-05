# COVID_Face_Mask_Detection
The unchecked spread of COVID 19 has caused an extraordinary number of deaths across the world that could have easily been prevented. According to the CDC, if everyone in the US wore masks for just 3 weeks, an estimated 130,000 lives could be saved.

Inspired by this statistic, I created a Face Mask detector that checks whether or not the people in a live video stream are wearing masks. I made the detector by training a conventional neural network on a dataset of over 4000 "Masked" and "Unmasked" images.

DEMO
![Alt Text]
(https://j.gifs.com/YW1j4K.gif)

​

The face detector uses the MobileNetV2 architecture so it’s computationally efficient and relatively easy to deploy the model to embedded systems.

I used a Caffe-based Face Detector.

​

T The general flow of logic is -

    Train the CNN model on the image dataset.

    Use OpenCV to loop over frames of a video - extract the region of interest(the person's face)

    Apply the face detector to the region of interest and update the resulting message on the video screen.
