import torch
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

smoothstep = lambda x: 3*x**2 - 2*x**3
smootherstep = lambda x: 6*x**5 - 15*x**4 + 10*x**3

def generate_perlin_noise(gradient_base_x_interval:float, gradient_base_y_interval:float, x_interval:float, y_interval:float, bounds:list, gradients = None, interpolation_function=smoothstep, 
                 nb_layers:int=1, aggregation_type:str="addition", frequency_growth_rate:float=2):
    x_range = bounds[1][0] - bounds[0][0]
    y_range = bounds[1][1] - bounds[0][1]
    nb_x = int(x_range //x_interval)
    nb_y = int(y_range //y_interval)

    input_grid = np.empty((nb_x, nb_y, 2))
    input_grid[np.arange(nb_x), :, 0] = np.linspace(bounds[0][0], bounds[1][0], nb_x)[:, None]
    input_grid[:, np.arange(nb_y), 1] = np.linspace(bounds[0][1], bounds[1][1], nb_y)[None, :]
    input_grid = torch.tensor(input_grid).to(device)

    # grad_interval_tensor = torch.tensor((gradient_x_interval, gradient_y_interval)).to(device)
    if aggregation_type=="addition":
        grad_interval_tensor = torch.tensor((gradient_base_x_interval, gradient_base_y_interval)).unsqueeze(0) / torch.tensor([frequency_growth_rate**i for i in range(nb_layers)]).unsqueeze(1)
    elif aggregation_type=="multiplication":
        grad_interval_tensor = torch.tensor((gradient_base_x_interval, gradient_base_y_interval)).unsqueeze(0).repeat(nb_layers, 1)
    else: raise ValueError(f"Non-valid aggregation type: '{aggregation_type}'")
    grad_interval_tensor = grad_interval_tensor.to(device)

    if gradients is None:
        gradients = F.normalize(torch.rand(nb_layers, int(x_range // grad_interval_tensor[-1][0])+2, int(y_range // grad_interval_tensor[-1][1])+2, 2) *2-1, dim=-1)
    gradients = gradients.to(device)
    
    anchor_point_coords = input_grid.unsqueeze(0) // grad_interval_tensor.view(nb_layers, 1, 1, 2)
    bounding_square_coords = torch.stack([anchor_point_coords + torch.tensor([(i%2), (i>=2)], dtype=int).to(device) for i in range(4) ], dim=3).to(torch.long) # some bullshit to iterate over [0,0], [0,1], [1,0] and [1,1], could have just hard coded it but 
    bounding_squares = gradients[torch.arange(nb_layers).view(-1, 1, 1, 1), bounding_square_coords[...,0], bounding_square_coords[...,1]]
    bounding_square_offsets = input_grid.view(1, nb_x, nb_y, 1, 2) - bounding_squares
    bounding_square_dotproducts = (bounding_square_offsets * bounding_squares).sum(dim=-1)

    #Bilinear Interpolation
    # assert torch.all(input_grid[..., 0] / grad_interval_tensor[..., 0] - anchor_point_coords[..., 0]>=0)  and torch.all(input_grid[..., 0] / grad_interval_tensor[..., 0] - anchor_point_coords[..., 0]<=1)
    alpha = interpolation_function(input_grid[..., 0].unsqueeze(0) / grad_interval_tensor[..., 0].view(-1,1,1) - anchor_point_coords[..., 0]).unsqueeze(-1)
    vertical_points = alpha * bounding_square_dotproducts[..., (1,3)] + (1-alpha) * bounding_square_dotproducts[..., (0,2)]   

    # assert torch.all(input_grid[..., 1] / grad_interval_tensor[..., 1] - anchor_point_coords[...,1]>=0) and torch.all(input_grid[..., 1] / grad_interval_tensor[..., 1] - anchor_point_coords[...,1]<=1)
    alpha_prime = interpolation_function(input_grid[..., 1].unsqueeze(0) / grad_interval_tensor[..., 1].view(-1,1,1) - anchor_point_coords[...,1])
    height_map = alpha_prime * vertical_points[..., 1] + (1-alpha_prime) * vertical_points[..., 0]

    if aggregation_type=="addition": 
        layer_coeffs = torch.tensor([1/2**i for i in range(nb_layers)]).view(nb_layers, 1, 1).to(device)
        height_map = (height_map * layer_coeffs).sum(axis=0)
    elif aggregation_type=="multiplication": 
        height_map = abs(height_map)
        height_map = (height_map).prod(axis=0)
    else: raise ValueError(f"Non-valid aggregation type: '{aggregation_type}'")
    height_map = (height_map - height_map.mean()) / (height_map.std())
    # height_map = (height_map.max() - height_map ) / (height_map.max() - height_map.min())

    fullmap = torch.cat((input_grid, height_map.unsqueeze(-1)), dim=-1)

    return fullmap


def generate_gaussian_mixture(means, covariances, x_interval:float, y_interval:float, bounds:list, gaussian_coeffs=None, noise_coeff:float = 0):
    nb_gaussians = len(means)

    means=torch.tensor(means).to(device)
    if gaussian_coeffs is None:
        gaussian_coeffs = torch.rand(nb_gaussians)/2
        gaussian_coeffs/= gaussian_coeffs.sum()
        gaussian_coeffs = gaussian_coeffs.to(device)
    else : gaussian_coeffs = torch.tensor(gaussian_coeffs).to(device)

    sigma_determinants = [_ for _ in range(nb_gaussians)]
    inverse_sigmas = [_ for _ in range(nb_gaussians)]
    for i in range(nb_gaussians):
        sigma_determinants[i] = np.linalg.det(covariances[i])
        inverse_sigmas[i] = np.linalg.inv(covariances[i])
    sigma_determinants = torch.tensor(sigma_determinants).to(device)
    inverse_sigmas = torch.tensor(np.array(inverse_sigmas)).to(device)

    x_range = bounds[1][0] - bounds[0][0]
    y_range = bounds[1][1] - bounds[0][1]
    nb_x = int(x_range //x_interval)
    nb_y = int(y_range //y_interval)

    input_grid = np.empty((nb_x, nb_y, 2))
    input_grid[np.arange(nb_x), :, 0] = np.linspace(bounds[0][0], bounds[1][0], nb_x)[:, None]
    input_grid[:, np.arange(nb_y), 1] = np.linspace(bounds[0][1], bounds[1][1], nb_y)[None, :]
    input_grid = torch.tensor(input_grid).to(device)

    diffs = input_grid.unsqueeze(2) - means # shape [nb_x, nb_y, nb_gaussians, 2]

    height_map = torch.sum(
        1/torch.sqrt(2*np.pi*sigma_determinants) * torch.exp(-0.5* ((diffs.unsqueeze(-2)@inverse_sigmas)@diffs.unsqueeze(-1)).squeeze()) * gaussian_coeffs
    , dim=2) 
    if noise_coeff>0: height_map+= noise_coeff * torch.rand(nb_x, nb_y).to(device)
    
    height_map = (height_map - height_map.mean()) / (height_map.std())
    fullmap = torch.cat((input_grid, height_map.unsqueeze(-1)), dim=-1)

    return fullmap

if __name__ == "__main__":
    fullmap = generate_perlin_noise(10, 10, 1, 1, [[0,0], [100, 100]], interpolation_function=smoothstep, nb_layers=3, aggregation_type="multiplication")
    height, xy = fullmap[..., 2], fullmap[..., (0,1)]
    height = height.cpu()
    xy = xy.cpu()

    color = (height.max() - height) / (height.max() - height.min())
    plt.scatter(xy[...,0], xy[...,1], c=color)
    plt.show()