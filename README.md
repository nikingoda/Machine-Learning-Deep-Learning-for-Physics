### Machine Learning and Deep Learning approaches for Physics Problem ###

**1. Introduction:**
- This first version, I am re-implement the model to predict the electric field-dependent absorption
coefficient in CdTe/CdS quantum dots (which calculated by a really complex calculation)

**2. Implementation:**
- Simply, it use the ANN model, with **one input layer** (input features like Photon energy, Electric Field, the number of neurons equal to number of features), **two hidden layers** (400 neurons each), and **one output layer (one single neuron)** which provide the predict values of electric field-dependent absorption
coefficient

- To generate the data, I will use the limited element method to solve the original formula. However, this method might be really expensive to calculate (that's why we implement this model) **(Update in future)**

- This model is to verify again a research paper, also show its efficient in predict the result of some **really complex** formula in Physics.


**3. Reference:**
- Idea: [Machine learning prediction of electric field-dependent absorption coefficient in CdTe/CdS quantum dots](https://www.sciencedirect.com/science/article/abs/pii/S254252932500207X)