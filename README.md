# Garbage-classification
Garbage classification using CNN

Dataset Details:

- Dataset Name: Garbage Classification Dataset
- Source: Kaggle
- Dataset Link: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
- Format: Images organized into folders by class
- Classes: cardboard, glass, metal, paper, plastic, trash
- Folder Structure:
  dataset/
    train/  (80% of source data)
      cardboard/
      glass/
      metal/
      paper/
      plastic/
      trash/
    test/    (20% of source data)
      cardboard/
      glass/
      ...
    test_images/
     small sample of images for predicting


Setup/Environment: Python

List of required modules:Tensorflow, numpy, matplotlib and os


Instructions to Run the Code:
1. Download the dataset from Kaggle and extract it.
2. Organize it into `dataset/train/` and `dataset/test/' with class subfolders such as cardboard, plastic, paper, metal, trash, glass. Add 3-4 sample images in the test_images folder rather than organizing it.
3. Place your '.py' file notebook in the same folder as the `dataset/`.
4. Run the Python script.
5. Ignore few of the warnings and since the epochs is at 15 times, it takes 4-5 mins of training time before producing the output.


Source Reference:
Kaggle Dataset: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification


Output Result:
Predicted images, graphs of accuracy vs epochs and loss vs epochs
