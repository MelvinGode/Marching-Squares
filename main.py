import numpy as np
from scipy.stats import multivariate_normal
from marching_squares import march_squares_gaussian_mixture, march_squares_on_grid
from generate_mountains import generate_gaussian_mixture, generate_perlin_noise

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from PIL import Image
from tqdm import tqdm
import os
from time import time
import imageio.v2 as imageio
from moviepy.editor import VideoFileClip
import torch
import torch.nn.functional as F
import subprocess


zerostart= time()

# General parameters
dpi = 0.5
# bounds = [[0.0, 0.0], [19.2/2, 10.8/2]]
bounds = [[0.0, 0.0], [10.8, 19.2]]
nb_lines = 20
time_in_seconds = 20
fps = 30
nb_frames = time_in_seconds * fps
amplitude = 0.5
# x_interval = dpi / (bounds[1][0] - bounds[0][0])
# y_interval = dpi / (bounds[1][1] - bounds[0][1])
x_interval=y_interval = dpi / (bounds[1][0] - bounds[0][0])
x_range = bounds[1][0] - bounds[0][0]
y_range = bounds[1][1] - bounds[0][1]

# Gaussian parameters
nb_gaussians  = 25
margin = [(bounds[1][0] - bounds[0][0])/5000, (bounds[1][1] - bounds[0][1])/5000]
gaussian_spanwns = torch.randn(nb_gaussians, 2) * torch.tensor(bounds)[1] / 5 + torch.tensor(bounds)[1]/2
# gaussian_spawns_x = np.random.uniform( low=bounds[0][0] + margin[0], high=bounds[1][0] - margin[0], size = (nb_gaussians))
# gaussian_spawns_y = np.random.uniform( low=bounds[0][1] + margin[1], high=bounds[1][1] - margin[1], size = (nb_gaussians))
# gaussian_spanwns = np.vstack((gaussian_spawns_x, gaussian_spawns_y)).T
gaussian_spreads = np.array([np.eye(2) + (np.random.rand(2, 2) *2 -1)/2 * 1 for _ in range(nb_gaussians)]) * 2
gaussian_coeffs = np.random.uniform(-1, 1, nb_gaussians) * 2
gaussian_coeffs/= abs(gaussian_coeffs).sum()

# Perlin parameters
gradient_base_x_interval = 8.5
gradient_base_y_interval = 8.5
nb_layers = 4
# aggregation_type="multiplication"
aggregation_type="addition"
frequency_growth_rate = 2

def compute_circle_gif(nb_frames:int):
    centers = gaussian_spanwns
    tvalues = np.linspace(-np.pi, 0, nb_frames)

    for i, t in tqdm(enumerate(tvalues)):
        means = centers + np.array([np.sin(t), np.cos(t)]) * 5
        fullmap = generate_gaussian_mixture(means, gaussian_spreads, x_interval, y_interval, bounds, gaussian_coeffs)
        linelist, color_counts = march_squares_on_grid(fullmap, thresholds)
        
        fig, ax = plt.subplots(1,1)
        ax.set_xlim(bounds[0][0], bounds[1][0])
        ax.set_ylim(bounds[0][1], bounds[1][1])
        lc = LineCollection(np.array(linelist), linewidths=0.4, colors=(1,1,1))
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(tick1On=False)
        ax.set_facecolor('black')
        # ax.set_aspect(1)

        plt.savefig(f'F:/Personal_Projects/Marching-Squares/GIF/image_{i}.png')#, dpi=1000)
        plt.close()
    
    compile_gif(nb_frames, 10)

def mp4_to_gif(input_mp4, output_gif, fps=60, scale=1080):
    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", input_mp4,
        "-fps_mode", "passthrough",
        "-filter_complex",
        (
            f"[0:v]fps={fps},scale={scale}:-1:flags=lanczos,"
            "split[x][z];"
            "[z]palettegen[p];"
            "[x][p]paletteuse"
        ),
        output_gif
    ], check=False)

def generate_plot(linelists, color_counts, colors, save=True, fname="mountains.png", dpi=None, show=False, return_array:bool=False):
    lc = LineCollection(linelists, linewidths=1, cmap = colors, linestyle = "-")
    group_ids = np.concatenate([np.full(nb_in_level, i) for i, nb_in_level in enumerate(color_counts)])
    lc.set_array(group_ids)

    fig, ax = plt.subplots(1,1, frameon=False, figsize=(bounds[1][0], bounds[1][1]))
    ax.add_collection(lc)
    ax.set_xlim(bounds[0][0], bounds[1][0])
    ax.set_ylim(bounds[0][1], bounds[1][1])
    # ax.set_axis_off()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.set_frame_on(False)
    ax.tick_params(tick1On=False)
    # ax.axis('off')
    ax.set_facecolor(basecolor)
    # ax.set_facecolor((0.9, 0.9, 0.9))
    ax.set_aspect(1)
    ax.set_position([0, 0, 1, 1])
    if return_array:
        fig.canvas.draw()

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()
        return image
    if save: plt.savefig(f'F:\Personal_Projects\Marching-Squares\{fname}', dpi=dpi, pad_inches=0)
    if show: plt.show()
    plt.close()

def compute_coef_change_gif(nb_frames:int, colormap, dpi=None, fps=24, online:bool=True):
    tvalues = np.linspace(0, 2*np.pi, nb_frames)
    alpharray, betarray = np.cos(tvalues), np.sin(tvalues)
    coefs_bis = np.random.uniform(-1, 1, nb_gaussians) * 2
    coefs_bis/= abs(coefs_bis).sum()
    coefs_t = alpharray[:, None] * gaussian_coeffs[None, :] + betarray[:, None] * coefs_bis[None, :]

    if online : 
        with imageio.get_writer("F:/Personal_Projects/Marching-Squares/GIF/circle.mp4", fps=fps, codec="libx264", macro_block_size=1) as writer:
            for i in tqdm(range(nb_frames)):
                fullmap = generate_gaussian_mixture(gaussian_spanwns, gaussian_spreads, x_interval, y_interval, bounds, coefs_t[i])
                linelist, color_counts = march_squares_on_grid(fullmap, thresholds)
                frame = generate_plot( linelist, color_counts, colormap, return_array=True, dpi=dpi)
                writer.append_data(frame)

    else:
        for i in tqdm(range(nb_frames)):
            # alpha, beta = alpha/(alpha+beta), beta/(alpha+beta)
            fullmap = generate_gaussian_mixture(gaussian_spanwns, gaussian_spreads, x_interval, y_interval, bounds, coefs_t[i])
            linelist, color_counts = march_squares_on_grid(fullmap, thresholds)
            generate_plot(linelist, color_counts, colormap, save=True, fname=f"GIF/image_{i}.png", dpi=dpi)

        compile_mp4(nb_frames, fps)
        videoClip = VideoFileClip("F:/Personal_Projects/Marching-Squares/GIF/circle.mp4")
        videoClip.write_gif("F:/Personal_Projects/Marching-Squares/GIF/circle.gif")
    
    mp4_to_gif("F:/Personal_Projects/Marching-Squares/GIF/circle.mp4", output_gif='F:/Personal_Projects/Marching-Squares/GIF/circle.gif', fps=fps)


def compute_cov_change_gif(nb_frames:int, dpi=None, fps=24, online:bool=True):
    tvalues = np.linspace(0, 2*np.pi, nb_frames)
    alpharray, betarray = np.cos(tvalues), np.sin(tvalues)
    covariances_bis = gaussian_spreads = np.array([np.eye(2) + np.random.rand(2, 2) * 2 - 1 for _ in range(nb_gaussians)]) * 0.25
    cov_t = alpharray[:, None, None, None]* gaussian_spreads[None, ...] + betarray[:, None, None, None]* covariances_bis[None, ...]

    if online : 
        with imageio.get_writer("F:/Personal_Projects/Marching-Squares/GIF/circle.mp4", fps=fps, codec="libx264", macro_block_size=1) as writer:
            for i in tqdm(range(nb_frames)):
                linelist, color_counts = march_squares_gaussian_mixture(gaussian_spanwns, cov_t[i], x_interval, y_interval, bounds, thresholds, gaussian_coeffs)
                frame = generate_plot( linelist, color_counts, colormap, return_array=True, dpi=dpi)
                writer.append_data(frame)

    else:
        for i in tqdm(range(nb_frames)):
            # alpha, beta = alpha/(alpha+beta), beta/(alpha+beta)
            linelist, color_counts = march_squares_gaussian_mixture(gaussian_spanwns, cov_t[i], x_interval, y_interval, bounds, thresholds, gaussian_coeffs)
            generate_plot(linelist, color_counts, colormap, save=True, fname=f"GIF/image_{i}.png", dpi=dpi)

        compile_mp4(nb_frames, fps)
        videoClip = VideoFileClip("F:/Personal_Projects/Marching-Squares/GIF/circle.mp4")
        videoClip.write_gif("F:/Personal_Projects/Marching-Squares/GIF/circle.gif")

def compute_double_change_gif(nb_frames:int, dpi=None, fps=24, online:bool=True):
    start1, start2 = np.random.rand(2) * 2*np.pi
    tvalues1 = np.linspace(start1, start1 + 2*np.pi, nb_frames)
    tvalues2 = np.linspace(start2, start2 + 2*np.pi, nb_frames)
    alpharray1, betarray1 = np.cos(tvalues1)/2 + 0.5, np.sin(tvalues1)/2 + 0.5
    alpharray2, betarray2 = np.cos(tvalues2)/2 + 0.5, np.sin(tvalues2)/2 + 0.5
    covariances_bis = gaussian_spreads = np.array([np.eye(2) + 1 * np.random.rand(2, 2) for _ in range(nb_gaussians)]) * 1
    cov_t = alpharray1[:, None, None, None]* gaussian_spreads[None, ...] + betarray1[:, None, None, None]* covariances_bis[None, ...]
    coefs_bis = np.random.uniform(-1, 1, nb_gaussians) * 2
    coefs_bis/= coefs_bis.sum()
    coefs_t = alpharray2[:, None] * gaussian_coeffs[None, :] + betarray2[:, None] * coefs_bis[None, :]

    if online : 
        with imageio.get_writer("F:/Personal_Projects/Marching-Squares/GIF/circle.mp4", fps=fps, codec="libx264", macro_block_size=1) as writer:
            for i in tqdm(range(nb_frames)):
                linelist, color_counts = march_squares_gaussian_mixture(gaussian_spanwns, cov_t[i], x_interval, y_interval, bounds, thresholds, coefs_t[i])
                frame = generate_plot( linelist, color_counts, colormap, return_array=True, dpi=dpi)
                writer.append_data(frame)

    else:
        for i in tqdm(range(nb_frames)):
            # alpha, beta = alpha/(alpha+beta), beta/(alpha+beta)
            linelist, color_counts = march_squares_gaussian_mixture(gaussian_spanwns, cov_t[i], x_interval, y_interval, bounds, thresholds, coefs_t[i])
            generate_plot(linelist, color_counts, colormap, save=True, fname=f"GIF/image_{i}.png", dpi=dpi)

        compile_mp4(nb_frames, fps)
        videoClip = VideoFileClip("F:/Personal_Projects/Marching-Squares/GIF/circle.mp4")
        videoClip.write_gif("F:/Personal_Projects/Marching-Squares/GIF/circle.gif")


def compute_perlin_gradient_change_gif(nb_frames:int, colormap, dpi=None, fps=24, amplitude:float = 1):
    tvalues = torch.linspace(0, 2*np.pi, nb_frames).view(nb_frames, 1, 1, 1, 1)
    alpharray, betarray = torch.cos(tvalues) * amplitude, torch.sin(tvalues) * amplitude
    grad_interval_tensor = torch.tensor((gradient_base_x_interval, gradient_base_y_interval)).unsqueeze(0) / torch.tensor([frequency_growth_rate**i for i in range(nb_layers)]).unsqueeze(1)
    base_gradients = F.normalize(torch.rand(nb_layers, int(x_range // grad_interval_tensor[-1][0])+2, int(y_range // grad_interval_tensor[-1][1])+2, 2) *2-1, dim=-1) 
    target_gradients = F.normalize(torch.rand(nb_layers, int(x_range // grad_interval_tensor[-1][0])+2, int(y_range // grad_interval_tensor[-1][1])+2, 2) *2-1, dim=-1) 
    gradients_t = alpharray* base_gradients.unsqueeze(0) + betarray* target_gradients.unsqueeze(0)

    with imageio.get_writer("F:/Personal_Projects/Marching-Squares/GIF/circle.mp4", fps=fps, codec="libx264", macro_block_size=1) as writer:
        for i in tqdm(range(nb_frames)):
            fullmap = generate_perlin_noise(gradient_base_x_interval, gradient_base_y_interval, x_interval, y_interval, bounds=bounds, nb_layers=nb_layers, aggregation_type=aggregation_type, gradients = gradients_t[i])
            linelist, color_counts = march_squares_on_grid(fullmap, thresholds)
            frame = generate_plot(linelist, color_counts, colormap, return_array=True, dpi=dpi)
            writer.append_data(frame)
    mp4_to_gif("F:/Personal_Projects/Marching-Squares/GIF/circle.mp4", output_gif='F:/Personal_Projects/Marching-Squares/GIF/circle.gif', fps=fps)


def compile_gif(nb_frames:int, duration:int):
    # frames = [Image.open("F:/Personal_Projects/Marching-Squares/GIF/image_"+str(i)+".png").convert("RGB") for i in range(nb_frames)]
    firstframe = Image.open("F:/Personal_Projects/Marching-Squares/GIF/image_0.png").convert("RGB").quantize(colors=256, method=Image.FASTOCTREE)
    
    def frame_generator():
        for i in tqdm(range(1, nb_frames)):
            im = Image.open(f"F:/Personal_Projects/Marching-Squares/GIF/image_{i}.png").convert("RGB")
            yield im.quantize(palette=firstframe)

    firstframe.save("circlegif.gif", save_all=True, append_images=frame_generator(), duration= duration/nb_frames * 1000, loop=0, optimize=False)

def compile_mp4(nb_frames:int, fps:int):
    with imageio.get_writer("F:/Personal_Projects/Marching-Squares/GIF/circle.mp4", fps= fps) as writer:
        for i in tqdm(range(nb_frames)):
            image = imageio.imread(f"F:/Personal_Projects/Marching-Squares/GIF/image_{i}.png")
            writer.append_data(image)


# TODO exponential threshold map
thresholds = np.linspace(-2.5, 2.5, nb_lines)

dodgerblue = np.array([30, 144, 255])/256
slategray = np.array([112, 128, 144])/256
niceblue = np.array([0, 51, 102])/256
pitchblack = np.array([0.0, 0.0, 0.0])
basecolor = pitchblack

# colormap = np.array([np.arange(nb_lines)/nb_lines*2, np.arange(nb_lines)/(nb_lines**2), np.arange(nb_lines, 0, -1)/(nb_lines*1.5)]).T
# colormap /= colormap.max(axis=1)[:, None]

# colormap = np.array([np.arange(nb_lines)/nb_lines*2]).repeat(3, axis=0).T
# colormap/= colormap[-1]

# colormap = np.array([np.arange(nb_lines, 0, -1)/nb_lines*2]).repeat(3, axis=0).T
# colormap/= colormap[0]

# colormap = (np.concatenate((np.arange(1, nb_lines//2+1), np.arange(nb_lines//2+1, 1 - (nb_lines%2), -1))))[None, :].repeat(3, axis=0).T
colormap = np.concatenate((np.arange(nb_lines//2, 0, -1), np.arange(nb_lines//2 + 1 - (nb_lines%2))), dtype=np.float64)[None, :].repeat(3, axis=0).T
colormap/= colormap.max(axis=0)
colormap = basecolor + (colormap) * (1-basecolor)


# colormap = np.array([np.concatenate((np.arange(1, nb_lines//2+1), np.arange(nb_lines//2+1, 1, -1)), dtype=float)]).repeat(3, axis=0).T
# colormap/= float(nb_lines//2+1)
colormaplist = ListedColormap(colormap)




# mp4_to_gif("F:/Personal_Projects/Marching-Squares/GIF/circle.mp4", output_gif='F:/Personal_Projects/Marching-Squares/GIF/circle.gif', fps=fps)
# exit()

if __name__ == "__main__":
    compute_perlin_gradient_change_gif(nb_frames, dpi=100, fps=fps, colormap=colormaplist, amplitude = amplitude)
    # compute_coef_change_gif(nb_frames, dpi=100, fps=fps, online=True, colormap=colormaplist)
    exit()

    start = time()
    # fullmap = generate_gaussian_mixture(gaussian_spanwns, gaussian_spreads, x_interval, y_interval, bounds, gaussian_coeffs)
    fullmap = generate_perlin_noise(gradient_base_x_interval, gradient_base_y_interval, x_interval, y_interval, bounds= bounds, nb_layers=nb_layers, aggregation_type=aggregation_type)
    linelists, color_counts = march_squares_on_grid(fullmap, thresholds)
    print(f"Compute time : {time()-start:.2f}")


    generate_plot(linelists, color_counts, colormaplist, dpi=100, show=True, save=True)