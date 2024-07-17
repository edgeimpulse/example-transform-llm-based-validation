# Validate Object Detection Datasets Using GPT-4o

Let GPT-4o look at every image in your object detection dataset and check against up to three validation prompts you provide. If images are not compliant they will be disabled and reasoning will be put into the metadata. The block passes each image's bounding boxes and labels to GPT-4o so you can for example check if "The bounding boxes and labels do not correspond to to the objects in the image" among other things. 

You must pass an OPENAI API key either in the field below or via a secret which you can add from the Custom Blocks->Transformation page in your org with the name OPENAI_API_KEY. 

![image](https://github.com/user-attachments/assets/47544b24-f16f-4a84-ac7b-53ca9159e581)
## How to run (Edge Impulse) (Enterprise Only)

1. Go to an Enterprise project, upload an Object Detection dataset which you wish to validatae
2. Choose **Data acquisition->Data Sources->Add new data source**.
3. Select Transformation Block and the 'Validate Object Detection Datasets Using GPT-4o' block, fill in your prompts and and run the block.

    > You need an OPENAI API Key to run the GPT4o model 

4. Any items which are invalid will be disabled. Reasoning will be provided in the metadata for each data item
   ![image](https://github.com/user-attachments/assets/5c042831-4301-46e6-8431-f209f09f1694)


### Customizing this repository (enterprise only)

You can modify this repository and push it as a new custom transformation block.

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/tools/edge-impulse-cli).
2. Open a command prompt or terminal, and navigate to this folder.
3. Create a new transformation block:

    ```
    $ edge-impulse-blocks init

    ? Choose a type of block: Transformation block
    ? Choose an option: Create a new block
    ? Enter the name of your block: Custom Validate Object Detection Datasets Using GPT-4o
    ? Enter the description of your block: Use GPT4o to validate a dataset
    ? What type of data does this block operate on? Standalone (runs the container, but no files / data items passed in)
    ? Which buckets do you want to mount into this block (will be mounted under /mnt/s3fs/BUCKET_NAME, you can change these mount points in the St
    udio)?
    ? Would you like to download and load the example repository? no
    ```

4. Push the block:

    ```
    $ edge-impulse-blocks push
    ```
