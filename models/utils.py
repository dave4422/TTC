from typing import List
import torch, math
import numpy as np
import torch.nn as nn


class TemperatureScheduleConfig:
    def __init__(self,
                 temperature_schedule:str = "constant",

                 temperature = 0.1,
                 majority_temperature = None,
                 minority_temperature = None,
                 high_temp = 0.5,
                 low_temp = 0.07, 
                 switch_at_epoch= 100,) -> None:
        self.temperature_schedule = temperature_schedule
        self.temperature = temperature
        self.majority_temperature = majority_temperature
        self.minority_temperature = minority_temperature
        self.high_temp = high_temp
        self.low_temp = low_temp
        self.switch_at_epoch = switch_at_epoch

    def get_temperature(self, epoch:int, max_epoch:int) -> float:

        if self.temperature_schedule == "constant":
            if self.majority_temperature is None or self.minority_temperature is None or self.majority_temperature<0.0 or self.minority_temperature<0.0:
                return self.temperature,self.temperature
            return self.majority_temperature,self.minority_temperature
        elif self.temperature_schedule == "cos":
            tau_plus = 1.0
            tau_minus = 0.1
            T = 200
            result = (tau_plus - tau_minus) * (1 + math.cos(2 * math.pi * epoch / T)) / 2 + tau_minus
            return result,result
        elif self.temperature_schedule == "high-low":
            if  epoch< self.switch_at_epoch:
                return self.high_temp,self.high_temp
            else:
                return self.low_temp,self.low_temp
        elif self.temperature_schedule == "low-high":
            if epoch < self.switch_at_epoch:
                return self.low_temp,self.low_temp
            else:
                return self.low_temp,self.low_temp


def random_points_on_n_sphere(n, num_points=1):
    """
    Generate uniformly distributed random points on the unit (n-1)-sphere.
    (Marsaglia, 1972) algorithm

    :param n: Dimension of the space
    :param num_points: Number of points to generate
    :return: Array of points on the (n-1)-sphere
    """
   
    
    x = np.random.normal(0, 1, (num_points, n))

    # Calculate the radius for each point
    radius = np.linalg.norm(x, axis=1).reshape(-1, 1)

    
    # Normalize each point to lie on the unit n-sphere
    points_on_sphere = x / radius

    return torch.from_numpy(points_on_sphere)


from torchmetrics.functional import pairwise_cosine_similarity

from sklearn.metrics.pairwise import cosine_similarity

def assign_to_prototypes(projections, prototypes):
    """ Assign each projection to the closest prototype using pairwise cosine similarity. """
    prototype_tensor = torch.stack([p.data for p in prototypes])
    cosine_sim = pairwise_cosine_similarity(projections, prototype_tensor)
    # Assign each projection to the prototype with the highest similarity
    assignments = torch.argmax(cosine_sim, dim=1)
    return assignments

def find_n_prototypes(n, projections):
    print(f"searching {n} prototypes")
    dim = projections.shape[1]
    #
    # random_vectors_on_sphere = random_points_on_n_sphere(dim, num_points=n).float()



    # find vector with smalles similiarity:

    vector = nn.Parameter(torch.randn(projections.shape[1]))

    optimizer = torch.optim.Adam([vector], lr=0.0001)

    for e in range(10000):   # number of iterations
        optimizer.zero_grad()  
        # cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(projections, vector.unsqueeze(0).expand_as(projections), dim=1)
        
        # minimize the sum (or average) of the cosine similarity
        loss = torch.sum(cos_sim)
        loss.backward()  
        optimizer.step()
        
        # optional: renormalize vector back to magnitude 1
        vector.data = vector.data / torch.norm(vector.data)

        if e % 500 == 0:
            print('epoch', e, 'loss', loss.item())
    if n == 1:
        return vector.data
    if n == 2:
        return torch.stack([vector.data, -vector.data])
    
    raise NotImplementedError("n>2 not implemented")


def find_n_prototypes_spreaded(projections):
    dim = projections.shape[1]

    # Initialize two vectors with random values
    vector = nn.Parameter(torch.randn(dim))
    

    optimizer = torch.optim.Adam([vector], lr=0.0001)

    print(projections.shape)
    print(vector.unsqueeze(0).shape)
    for e in range(10000):  # number of iterations
        optimizer.zero_grad()

        # Cosine similarity for each vector with all projections
        cos_sim = pairwise_cosine_similarity(projections, vector.unsqueeze(0))
      
        loss = torch.sum(torch.abs((cos_sim)))
        
        loss.backward()
        optimizer.step()

        # Optional: renormalize vectors back to magnitude 1
        vector.data = vector.data / torch.norm(vector.data)


        if e % 500 == 0:
            print('epoch', e, 'loss', loss.item())

    
    print(np.mean(torch.nn.functional.cosine_similarity(projections, vector.data.unsqueeze(0)).numpy()))
    print(np.mean(torch.nn.functional.cosine_similarity(projections, -vector.data.unsqueeze(0)).numpy()))
    return torch.stack([vector.data, -vector.data])