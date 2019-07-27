> Disclaimer: The code in this repository is apadted from: https://github.com/MWransky/leaf-classification

# Leaf Classification

The goal of this project is to automate the process of plant recognition using a leaf image as an input. The input to the system is an image of a leaf and the output is the name of the plant species to which it belongs. There are millions of plant species around the globe, therefore, this is not an easy task. Many medical fields which involve plants in creating medicines can find an extensive use of this classifier. It can also be a simple smart-phone application where the user will take a picture of a plant leaf and instantly know the name of the species it belongs to.

## Dataset Used

The dataset for this problem contains around 1500 binary images. Apart from this, some feature details are also provided along with the image e.g. texture and shape, for which a separate attribute vector is given for each image individually. The original dataset is hosted on the University of California (UCL)'s website (https://archive.ics.uci.edu/ml/datasets/leaf). There are a total of 99 plant species to which a leaf can belong to and there are around 15 samples taken from each species. This allows us to train the model efficiently.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This software has the following requirements:

* Python 2.7 or later!
* `NumPy` 1.11 or later
* `TensorFlow`
* `Matplotlib`
  
### Installation

```
git clone https://github.com/adl1995/leaf-classification.git
cd leaf-classification
pip install -r requirements.txt
python learn.py
```

## License

This project is not under any license.
