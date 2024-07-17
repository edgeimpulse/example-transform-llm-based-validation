# Validate Object Detection Datasets Using GPT-4o

Let GPT-4o look at every image in your object detection dataset and check against up to three validation prompts you provide. If images are not compliant they will be disabled and reasoning will be put into the metadata. The block passes each image's bounding boxes and labels to GPT-4o so you can for example check if "The bounding boxes and labels do not correspond to to the objects in the image" among other things. 

You must pass an OPENAI API key either in the field below or via a secret which you can add from the Custom Blocks->Transformation page in your org with the name OPENAI_API_KEY. 

![image](https://github.com/user-attachments/assets/47544b24-f16f-4a84-ac7b-53ca9159e581)


