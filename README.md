# ANPR - Automatic Number Plate Recognition

## Project Architecture

![Project Architecture](https://github.com/Mudit-Sharma-30/ANPR-Automatic-Number-Plate-Recognition/assets/99650506/c9a99201-2f19-4a4a-8d20-fe9b284ddbc0)

## Introduction

This project focuses on Automatic Number Plate Recognition (ANPR), a crucial component of modern traffic management and security systems. ANPR allows for the automatic detection and recognition of license plates from images or video streams. This README guides you through the process of setting up and using our ANPR system.

## Getting Started

Before diving into the project, you'll need to obtain the necessary data and set up your environment.

### Data Collection

To get started with the project, you'll need the image dataset, which can be downloaded from [this link](https://drive.google.com/file/d/13x4LNA1O6uSr5_U3PjkhZmohrftwWXcz/view?usp=sharing) (approximately 200 MB).

### Data Labeling

We use the LabelImg image annotation tool for labeling the dataset. Follow these steps:

1. Download LabelImg from the [GitHub repository](https://github.com/tzutalin/labelImg).
2. Install the package as instructed in the repository.
3. Open the GUI, create rectangular bounding boxes around license plates, and save the output in XML format.
4. Ensure that both the images and XML files are stored in the same 'images' folder.

### Data Preparation

To convert the labeled data from XML to CSV, run the provided script `01_xml_to_csv.ipynb`.

## Model Building

Execute the `02_Object_detection.ipynb` code to build the ANPR model. Please note that this step may take several hours to complete. The trained model will be saved in the 'model' folder upon completion.

## TensorBoard Model

You can visualize training progress using TensorBoard. Run the following command to start TensorBoard:

!tensorboard --logdir="./object_detection"

Then, access TensorBoard in your browser via the provided link.

## Object Detection Pipeline

To use the ANPR model for object detection, run the code in `03_Make_prediction.ipynb`.

## Optical Character Recognition (OCR)

To extract text from images, we utilize Tesseract OCR. Follow the installation instructions for Tesseract OCR [here](https://sourceforge.net/projects/tesseract-ocr-alt/files/tesseract-ocr-setup-3.02.02.exe/download).

### Limitations of PyTesseract

Tesseract OCR works best with clean text segmentation from the background. To achieve optimal results, consider preprocessing your images for improved OCR accuracy.

## Web App Integration

1. Create a new folder named 'static' inside your web app directory.
2. Inside 'static', create subfolders: 'models', 'predict', 'roi', and 'upload'.
3. In the 'models' folder, copy the saved model file (with the extension .h5) for use in your web application.

![Web App Folder Structure](https://github.com/Mudit-Sharma-30/ANPR-Automatic-Number-Plate-Recognition/assets/99650506/2992734d-8e3c-410e-8213-25e60498f73e)

By following these steps, you'll have a functional ANPR system that can detect and recognize license plates from images and videos.

Feel free to reach out if you have any questions or need further assistance.


