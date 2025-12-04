The transformation file has two types of functions:
1. functions that are used to generate offline transformations when the dataset class is initiated. These functions take the originally collected dataset and generates a dataset that has both the greyscale version of the original copies, and the transformed images.
2. The sceond are more milder transformations that are passed to the dataset class as transforms and are applied randomly when the training data is being loaded.

I am using v2 and tv_tensors from torchvision.transforms. It is a relatively new addition that accepts different data structure, particularly boudning boxes and key points. I think of the gate labels as key points and pass them to the transform function as KeyPoints class. 

I have manually checked that the rotation transformation is applied correctly on the labels. However, the images generated has lower quality due to resizing and other issues, so now I must write a resize function that can be passed to trasforms.Compose to preserve the aspect ratio and to prevent any sort of blurring or averaging.
