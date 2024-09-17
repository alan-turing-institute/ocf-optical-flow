# First we'll load a sample from the dataset. We'll use this sample later
from cloudcasting.dataset import SatelliteDataset

dataset = SatelliteDataset(
    zarr_path="/bask/homes/l/lwcf1795/vjgo8416-climate/users/lwcf1795/data/2020_training_nonhrv.zarr",
    start_time="2020-06-01",
    end_time="2020-06-15",
    history_mins=15,
    forecast_mins=180,
    sample_freq_mins=15,
)

# Grab a single sample
X, y = dataset[1000]

# optical flow model requirements
import seaborn as sns
import xarray as xr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # CRS stands for "coordinate reference system"
from skimage import metrics

# We import the AbstractModel class
from cloudcasting.models import AbstractModel

# We define a new class that inherits from AbstractModel
class OpticalFlowTVL1Model(AbstractModel):
    """An optical flow model which predicts the next image given the prior two images using the TVL1 method"""

    def __init__(self, history_steps: int) -> None:
        # All models must include `history_steps` as a parameter. This is the number of previous
        # frames that the model uses to makes its predictions. This should not be more than 25, i.e.
        # 6 hours (inclusive of end points) of 15 minutely data

        # You also must inlude the following line in your init function
        super().__init__(history_steps)


        ###### YOUR CODE HERE ######
        # Here you can add any other parameters that you need to initialize your model
        # You might load your trained ML model or set up an optical flow method here

        ############################


    def forward(self, X):
        # This is where you will make predictions with your model
        # The input X is a numpy array with shape (batch_size, channels, time, height, width)

        ###### YOUR CODE HERE ######

        # Grab the most recent frame from the input data
        samples_list = []
        
        for b in range (X.shape[0]): 
            channel_list = []

            for c in range (X.shape[1]): 
                previous_image = X[b, c, -2, :, :]
                current_image = X[b, c, -1, :, :]

                new_image_list = []

                for i in range (12):

                    # convert to uint8 data type
                    image1 = (previous_image * 255).astype(np.uint8)
                    image2 = (current_image * 255).astype(np.uint8)
                    
                    optical_flow = cv2.optflow.createOptFlow_DualTVL1()
                
                    #optimise parameters (taken from Urbich et al. 2018 cloud imagery use case)
                    optical_flow.setTau(0.1)
                    optical_flow.setGamma(0.1)
                    optical_flow.setLambda(0.03) #Urbich et al. 2018 
                    optical_flow.setTheta(0.3) #default
                    optical_flow.setEpsilon(0.01)
                    optical_flow.setOuterIterations(2) #Urbich et al. 2018
                    optical_flow.setInnerIterations(10) #Urbich et al. 2018
                    optical_flow.setScalesNumber(3) #Urbich et al. 2018
                    optical_flow.setWarpingsNumber(3) #Urbich et al. 2018
                    optical_flow.setScaleStep(0.5) #Urbich et al. 2018
                    
                    #calculate TVL1 optical flow
                    flow = optical_flow.calc(image1, image2, None)

                    # extracting flow components
                    u = flow[..., 0] # width
                    v = flow[..., 1] # height

                    h = flow.shape[0]
                    w = flow.shape[1]
                    flow2 = -flow.copy()
                    flow2[:,:,0] += np.arange(w)
                    flow2[:,:,1] += np.arange(h)[:,np.newaxis]

                    new_image = cv2.remap(image2, flow2, None, cv2.INTER_LINEAR)
                    new_image_list.append(new_image.astype(np.float32)/255)

                    previous_image = image2.astype(float)/255
                    current_image = new_image.astype(float) / 255
                ############################
                image_stack = np.stack(new_image_list)
                channel_list.append(image_stack)
            channel_stack = np.stack(channel_list)
            samples_list.append(channel_stack)
        samples_stack = np.stack(samples_list)
        return samples_stack

    def hyperparameters_dict(self):

        # This function should return a dictionary of hyperparameters for the model
        # This is just for your own reference and will be saved with the model scores to wandb

        ###### YOUR CODE HERE ######

        params_dict =  {
            "setTau": 0.1,
            "setGamma": 0.1,
            "setLambda": 0.03,
            "setTheta" : 0.3,
            "setEpsilon" : 0.01,
            "setOuterIterations" : 2,
            "setInnerIterations" : 10,
            "setScalesNumber": 3,
            "setWarpingsNumber" : 3,
            "setScaleStep": 0.5
            
        }

        ############################

        return params_dict

# Instantiate the model
model = OpticalFlowTVL1Model(
    history_steps=2)

# Import the validate function
from cloudcasting.validation import validate

# Define the wandb project and run name where the results will be logged

# The wandb project is essentially the folder where the results will be stored on wandb.
# You should log your results to the "cloudcasting" folder. However, you can create a new project
# just for your owen results if you want to keep them separate from the main project. To create your
# own project, just change the WANDB_PROJECT variable to a new name.
WANDB_PROJECT = "cloudcasting"


# The wandb run name is the name that this specific call of the validate function will be logged
# under within the project. You can change this to whatever you like.
WANDB_RUN_NAME = "optical-flow-TVL1-val"

#run model validation process
validate(
    model=model,
    data_path="/bask/homes/l/lwcf1795/vjgo8416-climate/shared/data/eumetsat/testing2022/2022_test_nonhrv.zarr",
    wandb_project_name=WANDB_PROJECT,
    wandb_run_name=WANDB_RUN_NAME,
    batch_size = 2,
    num_workers = 0,
    batch_limit = 20,  # We limit the number of batches to 20 for this example
)


