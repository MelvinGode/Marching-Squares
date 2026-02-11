import numpy as np


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
LOOKUP = [
    [ # 0
        [ # 0 0
            [[], [diag_topright]], # 0 0 0
            [[diag_topleft], [horizontal]] # 0 0 1
        ],
        [ # 0 1
            [[diag_botright], [vertical]], # 0 1 0
            [[diag_botright, diag_topleft], [diag_botleft]] #0 1 1
        ]
    ],
    [ # 1
        [ # 1 0
            [[diag_botleft], [diag_botleft, diag_topright]], # 1 0 0
            [[vertical], [diag_botright]] # 1 0 1
        ],
        [ # 1 1
            [[horizontal], [diag_topleft]], # 1 1 0
            [[diag_topright], []] # 1 1 1
        ]
    ]
]

def get_interpolated_linepoints(anchor_binary:np.ndarray, anchor_values, anchor_abs_values):
    top[0] = anchor_abs_values[2]/abs(anchor_values[3] - anchor_values[2])
    bottom[0] = anchor_abs_values[0]/abs(anchor_values[1] - anchor_values[0])
    right[1] = anchor_abs_values[1]/abs(anchor_values[3] - anchor_values[1])
    left[1] = anchor_abs_values[0]/abs(anchor_values[0] - anchor_values[2])

    return LOOKUP[anchor_binary[0]][anchor_binary[1]][anchor_binary[2]][anchor_binary[3]]

def get_linepoints(anchor_binary:np.ndarray, anchor_values:list[float]=None):
    # TODO replace with an actual lookup table
    nb_on = anchor_binary.sum()
    if nb_on==0 or nb_on==4 : return []
    if nb_on==1 or nb_on==3:
        if nb_on==3 : anchor_binary = 1-anchor_binary # unify 1on and 3on cases so that we're always looking for the only 1 value
        if anchor_binary[0]: return [[[0, 0.5], [0.5, 0]]]
        elif anchor_binary[1]: return [[[0.5, 0], [1, 0.5]]]
        elif anchor_binary[2]: return [[[0, 0.5], [0.5, 1]]]
        elif anchor_binary[3]: return [[[0.5, 1], [1, 0.5]]]

    elif nb_on==2:
        if anchor_binary[0] == anchor_binary[2] : 
            return [[[0.5, 1], [0.5, 0]]]
        elif anchor_binary[0] == anchor_binary[1] : 
            return [[[0, 0.5], [1, 0.5]]]
        elif anchor_binary[0] == 1 and anchor_binary[2] == 1 : 
            return [[[0, 0.5], [0.5, 1]], [[0.5, 0], [1, 0.5]]]
        elif anchor_binary[0] == 0 and anchor_binary[2] == 0 : 
            return [[[0.5, 1], [1, 0.5]], [[0, 0.5], [0.5, 0]]]

    assert False, "this case should not happen"
    #TODO weighted case



def march_squares(f:callable, x_interval:int, y_interval, bounds:list, thresholds:list) -> list:
    x_range = bounds[1][0] - bounds[0][0]
    y_range = bounds[1][1] - bounds[0][1]
    nb_x = int(x_range //x_interval)
    nb_y = int(y_range //y_interval)
    interval_vector = np.array((x_interval, y_interval))

    input_grid = np.empty((nb_x, nb_y, 2))
    # input_grid[np.arange(nb_x), :, 0] = np.linspace(bounds[0][0], bounds[1][0], nb_x)
    # input_grid[:, np.arange(nb_y), 1] = np.linspace(bounds[0][1], bounds[1][1], nb_y)
    input_grid[:, :, 0] = np.repeat(np.arange(nb_x), nb_y).reshape(nb_x, nb_y) /nb_x * x_range
    input_grid[:, :, 1] = np.tile(np.arange(nb_y), nb_x).reshape(nb_x, nb_y) /nb_y * y_range
    output_grid = np.apply_along_axis(f, 2, input_grid)

    linelists = []
    for k, thresh in enumerate(thresholds):
        binary_grid = (output_grid > thresh)
        linelist = []
        abs_diffs = abs(output_grid - thresh)

        for i in range(nb_x-1):
            for j in range(nb_y-1):
                anchor_bin = binary_grid[tuple(np.array([[i,j], [i+1,j], [i,j+1], [i+1,j+1]]).T)].astype(int)
                if np.sum(anchor_bin) == 0 or np.sum(anchor_bin)==4 : continue
                anchor_values = output_grid[tuple(np.array([[i,j], [i+1,j], [i,j+1], [i+1,j+1]]).T)] - thresh
                anchor_abs_values = abs_diffs[tuple(np.array([[i,j], [i+1,j], [i,j+1], [i+1,j+1]]).T)]
                # linepoints = get_linepoints(anchor_bin)
                # linepoints = LOOKUP[anchor_bin[0]][anchor_bin[1]][anchor_bin[2]][anchor_bin[3]]
                linepoints = get_interpolated_linepoints(anchor_bin, anchor_values, anchor_abs_values)
                # print(f'Anchors : {anchor_bin}\nLinepoints : {linepoints}')
                if len(linepoints):
                    # print("hello")
                    # print(linepoints)
                    linelist.extend(np.array(linepoints)*interval_vector + np.array([i/nb_x * x_range, j/nb_y * y_range]))
                    # print(np.array(linepoints)*interval + np.array([i/nb_x * x_range, j/nb_y * y_range]))
        print(f"Threshold {k+1}/{len(thresholds)}: value {thresh:.3f}. {len(linelist)} lines found")
        linelists.append(linelist)

    return linelists

# a = np.arange(9).reshape((3,3))
# print(a[tuple(np.array([[0,1], [2,2]]).T)])