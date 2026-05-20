import numpy as np
import torch
import matplotlib.pyplot as plt


top = [0.5, 1]
right = [1, 0.5]
bottom = [0.5, 0]
left = [0, 0.5]

vertical = [top, bottom]
horizontal = [left, right]
diag_topright = [top, right]
diag_topleft = [top, left]
diag_botleft = [bottom, left]
diag_botright = [bottom, right]

#botleft, botright, topleft, topright
LOOKUP_tensor = torch.zeros(2, 2, 2, 2, 2, 2, dtype=torch.long)
LOOKUP_tensor[tuple(torch.tensor([[1,1,0,0, 0], [0,0,1,1, 0]]).T)] = torch.tensor([1, 3]) # horizontal
LOOKUP_tensor[tuple(torch.tensor([[0,1,0,1, 0], [1,0,1,0, 0]]).T)] = torch.tensor([0, 2]) # vertical
LOOKUP_tensor[tuple(torch.tensor([[1,0,0,0, 0], [0,1,1,1, 0]]).T)] = torch.tensor([2, 3]) # diag_botleft
LOOKUP_tensor[tuple(torch.tensor([[0,1,0,0, 0], [1,0,1,1, 0]]).T)] = torch.tensor([1, 2]) # diag_botright
LOOKUP_tensor[tuple(torch.tensor([[0,0,0,1, 0], [1,1,1,0, 0]]).T)] = torch.tensor([0, 1]) # diag_topright
LOOKUP_tensor[tuple(torch.tensor([[0,0,1,0, 0], [1,1,0,1, 0]]).T)] = torch.tensor([0, 3]) # diag_topleft

# We don't handle diagonal cases by sampling midpoint because this is annoying and pretty should prety much never show up at a high enough resolution
# LOOKUP_tensor[tuple(torch.tensor([[0,1,1,0], [1,0,0,1]]).T)] = torch.tensor([[1, 2], [0, 3]]) # diag_botright, diag_topleft 
# LOOKUP_tensor[tuple(torch.tensor([[0,1,1,0], [1,0,0,1]]).T)] = torch.tensor([[2, 3], [0, 1]]) # diag_botleft, diag_topright

device = "cuda" if torch.cuda.is_available() else "cpu"


def march_squares_gaussian_mixture(means, covariances, x_interval:float, y_interval:float, bounds:list, threshold_list:list, gaussian_coeffs=None):
    nb_gaussians = len(means)

    means=torch.tensor(means).to(device)
    if gaussian_coeffs is None:
        gaussian_coeffs = torch.rand(nb_gaussians)/2
        gaussian_coeffs/= gaussian_coeffs.sum()
        gaussian_coeffs = gaussian_coeffs.to(device)
    else : gaussian_coeffs = torch.tensor(gaussian_coeffs).to(device)

    thresholds = torch.tensor(threshold_list).to(device)
    nb_thresholds = len(thresholds)

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
    interval_vector = np.array((x_interval, y_interval))

    input_grid = np.empty((nb_x, nb_y, 2))
    input_grid[np.arange(nb_x), :, 0] = np.linspace(bounds[0][0], bounds[1][0], nb_x)[:, None]
    input_grid[:, np.arange(nb_y), 1] = np.linspace(bounds[0][1], bounds[1][1], nb_y)[None, :]
    input_grid = torch.tensor(input_grid).to(device)

    diffs = input_grid.unsqueeze(2) - means # shape [nb_x, nb_y, nb_gaussians, 2]

    output_grid = torch.sum(
        1/torch.sqrt(2*np.pi*sigma_determinants) * torch.exp(-0.5* ((diffs.unsqueeze(-2)@inverse_sigmas)@diffs.unsqueeze(-1)).squeeze()) * gaussian_coeffs
    , dim=2) #+ torch.rand(nb_x, nb_y).to(device) * 0.0001

    binary_grid = output_grid.unsqueeze(2) > thresholds
    binary_indexes = torch.stack([
                                    binary_grid[:-1, :-1], 
                                    binary_grid[1:,  :-1], 
                                    binary_grid[:-1,  1:], 
                                    binary_grid[1:,   1:]
                                ], dim=0).to(torch.long)

    interpolation_coords = torch.stack([
        torch.cat([(thresholds - output_grid[:-1, :].unsqueeze(-1)) / (output_grid[1:, :] - output_grid[:-1, :]).unsqueeze(-1), torch.zeros(1, nb_y, nb_thresholds).to(device)], dim=0),
        torch.cat([(thresholds - output_grid[:, :-1].unsqueeze(-1)) / (output_grid[:, 1:] - output_grid[:, :-1]).unsqueeze(-1), torch.zeros(nb_x, 1, nb_thresholds).to(device)], dim=1)
    ], dim=-1)
    interpolation_coords  = interpolation_coords * torch.tensor(interval_vector).to(device) + input_grid.unsqueeze(-2)

    tmp = input_grid.unsqueeze(-2).repeat(1, 1, nb_thresholds, 1)
    interpolated_points = torch.stack([
        torch.stack([interpolation_coords[:-1, 1:, :, 0], tmp[:-1, 1:, :, 1]], dim=-1), # top
        torch.stack([tmp[1:, :-1, :, 0], interpolation_coords[1:, :-1, :, 1]], dim=-1), # right
        torch.stack([interpolation_coords[:-1, :-1, :, 0], tmp[:-1, :-1, :, 1]], dim=-1), # bot
        torch.stack([tmp[:-1, :-1, :, 0], interpolation_coords[:-1, :-1, :, 1]], dim=-1) # left
    ], dim=-2) # shape [nb_x, nb_y, nb_thresholds, 4, 2]

    linepoint_indexes = LOOKUP_tensor.to(device)[tuple(binary_indexes)].to(torch.long)
    mask = torch.any(linepoint_indexes != 0, dim=(-1))
    
    line_fulltensor = torch.take_along_dim(interpolated_points.unsqueeze(4), linepoint_indexes.unsqueeze(-1), 3)
    line_tensor = line_fulltensor.transpose(0,2)[mask.transpose(0,2)].view(-1, 2, 2)
    linelist = line_tensor.cpu().numpy()

    level_count = torch.sum(mask, dim=(0,1,3)).cpu().numpy()

    return linelist, level_count


def march_squares_on_grid(fullmap, threshold_list:list):
    thresholds = torch.tensor(threshold_list).to(device)
    nb_thresholds = len(thresholds)
    nb_x, nb_y, _ = fullmap.shape
    height_map, input_grid = fullmap[..., 2], fullmap[..., (0,1)]
    interval_vector = torch.tensor([input_grid[1,0,0] - input_grid[0,0,0], input_grid[0,1,1] - input_grid[0,0,1]]).to(device)

    binary_grid = height_map.unsqueeze(2) > thresholds
    binary_indexes = torch.stack([
                                    binary_grid[:-1, :-1], 
                                    binary_grid[1:,  :-1], 
                                    binary_grid[:-1,  1:], 
                                    binary_grid[1:,   1:]
                                ], dim=0).to(torch.long)

    interpolation_coords = torch.stack([
        torch.cat([(thresholds - height_map[:-1, :].unsqueeze(-1)) / (height_map[1:, :] - height_map[:-1, :]).unsqueeze(-1), torch.zeros(1, nb_y, nb_thresholds).to(device)], dim=0),
        torch.cat([(thresholds - height_map[:, :-1].unsqueeze(-1)) / (height_map[:, 1:] - height_map[:, :-1]).unsqueeze(-1), torch.zeros(nb_x, 1, nb_thresholds).to(device)], dim=1)
    ], dim=-1)
    interpolation_coords  = interpolation_coords * torch.tensor(interval_vector).to(device) + input_grid.unsqueeze(-2)

    tmp = input_grid.unsqueeze(-2).repeat(1, 1, nb_thresholds, 1)
    interpolated_points = torch.stack([
        torch.stack([interpolation_coords[:-1, 1:, :, 0], tmp[:-1, 1:, :, 1]], dim=-1), # top
        torch.stack([tmp[1:, :-1, :, 0], interpolation_coords[1:, :-1, :, 1]], dim=-1), # right
        torch.stack([interpolation_coords[:-1, :-1, :, 0], tmp[:-1, :-1, :, 1]], dim=-1), # bot
        torch.stack([tmp[:-1, :-1, :, 0], interpolation_coords[:-1, :-1, :, 1]], dim=-1) # left
    ], dim=-2) # shape [nb_x, nb_y, nb_thresholds, 4, 2]

    linepoint_indexes = LOOKUP_tensor.to(device)[tuple(binary_indexes)].to(torch.long)
    mask = torch.any(linepoint_indexes != 0, dim=(-1))
    
    line_fulltensor = torch.take_along_dim(interpolated_points.unsqueeze(4), linepoint_indexes.unsqueeze(-1), 3)
    line_tensor = line_fulltensor.transpose(0,2)[mask.transpose(0,2)].view(-1, 2, 2)
    linelist = line_tensor.cpu().numpy()

    level_count = torch.sum(mask, dim=(0,1,3)).cpu().numpy()

    return linelist, level_count


# Comment purgatory

# def march_squares_arbitrary_function(f:function, x_interval:int, y_interval, bounds:list, thresholds:list) -> list:
#     x_range = bounds[1][0] - bounds[0][0]
#     y_range = bounds[1][1] - bounds[0][1]
#     nb_x = int(x_range //x_interval)
#     nb_y = int(y_range //y_interval)
#     interval_vector = np.array((x_interval, y_interval))

#     input_grid = np.empty((nb_x, nb_y, 2))
#     # input_grid[np.arange(nb_x), :, 0] = np.linspace(bounds[0][0], bounds[1][0], nb_x)
#     # input_grid[:, np.arange(nb_y), 1] = np.linspace(bounds[0][1], bounds[1][1], nb_y)
#     input_grid[:, :, 0] = np.repeat(np.arange(nb_x), nb_y).reshape(nb_x, nb_y) /nb_x * x_range
#     input_grid[:, :, 1] = np.tile(np.arange(nb_y), nb_x).reshape(nb_x, nb_y) /nb_y * y_range
#     output_grid = np.apply_along_axis(f, 2, input_grid)

#     linelists = []
#     for k, thresh in enumerate(thresholds):
#         binary_grid = (output_grid > thresh)
#         linelist = []
#         abs_diffs = abs(output_grid - thresh)

#         for i in range(nb_x-1):
#             for j in range(nb_y-1):
#                 anchor_bin = binary_grid[tuple(np.array([[i,j], [i+1,j], [i,j+1], [i+1,j+1]]).T)].astype(int)
#                 if np.sum(anchor_bin) == 0 or np.sum(anchor_bin)==4 : continue
#                 anchor_values = output_grid[tuple(np.array([[i,j], [i+1,j], [i,j+1], [i+1,j+1]]).T)] - thresh
#                 anchor_abs_values = abs_diffs[tuple(np.array([[i,j], [i+1,j], [i,j+1], [i+1,j+1]]).T)]
#                 # linepoints = get_linepoints(anchor_bin)
#                 # linepoints = LOOKUP[anchor_bin[0]][anchor_bin[1]][anchor_bin[2]][anchor_bin[3]]
#                 linepoints = get_interpolated_linepoints(anchor_bin, anchor_values, anchor_abs_values)
#                 # print(f'Anchors : {anchor_bin}\nLinepoints : {linepoints}')
#                 if len(linepoints):
#                     # print("hello")
#                     # print(linepoints)
#                     linelist.extend(np.array(linepoints)*interval_vector + np.array([i/nb_x * x_range, j/nb_y * y_range]))
#                     # print(np.array(linepoints)*interval + np.array([i/nb_x * x_range, j/nb_y * y_range]))
#         print(f"Threshold {k+1}/{len(thresholds)}: value {thresh:.3f}. {len(linelist)} lines found")
#         linelists.append(linelist)

#     return linelists

# LOOKUP = [
#     [ # 0
#         [ # 0 0
#             [[[]], [diag_topright]], # 0 0 0
#             [[diag_topleft], [horizontal]] # 0 0 1
#         ],
#         [ # 0 1
#             [[diag_botright], [vertical]], # 0 1 0
#             [[diag_botright, diag_topleft], [diag_botleft]] #0 1 1
#         ]
#     ],
#     [ # 1
#         [ # 1 0
#             [[diag_botleft], [diag_botleft, diag_topright]], # 1 0 0
#             [[vertical], [diag_botright]] # 1 0 1
#         ],
#         [ # 1 1
#             [[horizontal], [diag_topleft]], # 1 1 0
#             [[diag_topright], [[]]] # 1 1 1
#         ]
#     ]
# ]

# def get_interpolated_linepoints(anchor_binary:np.ndarray, anchor_values, anchor_abs_values):
#     top[0] = anchor_abs_values[2]/abs(anchor_values[3] - anchor_values[2])
#     bottom[0] = anchor_abs_values[0]/abs(anchor_values[1] - anchor_values[0])
#     right[1] = anchor_abs_values[1]/abs(anchor_values[3] - anchor_values[1])
#     left[1] = anchor_abs_values[0]/abs(anchor_values[0] - anchor_values[2])

#     return LOOKUP[anchor_binary[0]][anchor_binary[1]][anchor_binary[2]][anchor_binary[3]]