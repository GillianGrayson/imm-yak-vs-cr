
# Region classification using immunological profile (Yakutia VS Central Russia)

Repository with source code for paper "How extremely cold climate affects immunological profile: Yakutian population study" by 
[A. Kalyakulina](https://orcid.org/0000-0001-9277-502X),
[I. Yusipov](http://orcid.org/0000-0002-0540-9281),
[E. Kondakova](http://orcid.org/0000-0002-6123-8181),
[C. Franceschi](http://orcid.org/0000-0001-9841-6386),
[M. Ivanchenko](http://orcid.org/0000-0002-1903-7423). 

## Description 

This repository contains source code for plotting figures and building machine learning models used to classify the region using immunology profile data.

## Abstract
Yakutia is one of the coldest permanently inhabited regions in the world, characterized by subarctic climate with the average January temperature in the regional capital Yakutsk below -40 °C. Immunological mechanisms of adaptation to such a harsh environment have not been extensively studied previously. This paper reports a study of the immunologic profile of the Yakutian population, compared with residents of Central Russia. Using state-of-the-art deep learning models for tabular data, a classifier was built that distinguishes Yakutia from Central Russia with an accuracy of more than 0.95, and explainable artificial intelligence methods revealed immunologic parameters whose levels differ most between the populations. The study of the biological functions of these cytokines helps to identify possible associations between their levels and susceptibility to various groups of diseases in the studied cohorts, as well as to find links with such climatic factors as outdoor temperature, humidity, UV exposure, and air pollution. 

## Project Structure

```
├── configs                         <- Configs for PyTorchTabular
├── notebooks                       <- Jupyter notebooks
│   ├── 1_preprocessing.ipynb           <- Data preprocessing and plotting basic figures 
│   └── 2_classification.ipynb          <- Building ML classificator and plotting figures
├── data                            <- Immunological data and generated results
├── src                             <- Source code for auxiliary functions
├── .gitignore                      <- List of files ignored by git
├── .project-root                   <- File for inferring the position of project root directory
├── requirements.txt                <- File for installing python dependencies
└── LICENSE                         <- MIT license
└── README.md                       <- This file
```

## Install dependencies

```bash
# clone project
git clone https://github.com/GillianGrayson/SImAge
cd SImAge

# [OPTIONAL] create conda environment
conda create -n env_name python=3.9
conda activate env_name

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## License

```
The MIT License (MIT)

Copyright (c) 2024 Alena Kalyakulina, Igor Yusipov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
