FROM continuumio/miniconda3

# Install jupyter 
RUN conda install jupyter 

# Install other required packages
RUN pip install tensorflow keras seaborn scikit-learn

# Make a data folder which will be connected to the host
RUN mkdir /deep_learning_assignment
