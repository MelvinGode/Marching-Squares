import numpy as np
from scipy.stats import multivariate_normal
from marching_squares import march_squares
import matplotlib.pyplot as plt


nb_gaussians  = 30

bounds = [[0.0, 0.0], [10.80, 19.80]]
margin = [(bounds[1][0] - bounds[0][0])/10, (bounds[1][1] - bounds[0][1])/10]

gaussian_spawns_x = np.random.uniform( low=bounds[0][0] + margin[0], high=bounds[1][0] - margin[0], size = (nb_gaussians))
gaussian_spawns_y = np.random.uniform( low=bounds[0][1] + margin[1], high=bounds[1][1] - margin[1], size = (nb_gaussians))
gaussian_spanwns = np.vstack((gaussian_spawns_x, gaussian_spawns_y)).T
# gaussian_spanwns = np.array([1.5, 1.5])
gaussian_spreads = np.array([np.eye(2) + 1*np.random.rand(2, 2) for _ in range(nb_gaussians)]) / 3
# gaussian_spreads = np.eye(2)
gaussian_coeffs = np.random.uniform(0.5, 1, nb_gaussians)
gaussian_coeffs/= gaussian_coeffs.sum()

sigma_determinants = [_ for _ in range(nb_gaussians)]
inverse_sigmas = [_ for _ in range(nb_gaussians)]

for i in range(nb_gaussians):
    sigma_determinants[i] = np.linalg.det(gaussian_spreads[i])
    inverse_sigmas[i] = np.linalg.inv(gaussian_spreads[i])

def get_mixture_pdf(x):
    pdfsum = 0
    for i in range(nb_gaussians):
        pdfsum += 1/np.sqrt(2*np.pi*sigma_determinants[i]) * np.exp(-0.5*(x - gaussian_spanwns[i])@inverse_sigmas[i]@(x-gaussian_spanwns[i])) * gaussian_coeffs[i]
    return pdfsum.mean()

# def get_mixture_pdf(x):
#     pdfsum = 0
#     for i in range(nb_gaussians):
#         pdfsum += multivariate_normal.pdf(x, gaussian_spanwns[i], gaussian_spreads[i]) * gaussian_coeffs[i]
#     return pdfsum.mean()

# print(get_mixture_pdf([1, 1]))
# print(get_mixture_pdf([1, 2]))
# print(get_mixture_pdf([2, 2]))
# print(get_mixture_pdf([2, 1]))
# exit()

nb_lines = 30

thresholds = np.linspace(1e-4, 0.11, nb_lines)
# thresholds = [0.12]
x_interval = 0.01
y_interval = 0.05

fig, ax = plt.subplots(1,1)
linelists = march_squares(get_mixture_pdf, x_interval, y_interval, bounds, thresholds)
# print("Gaussian spawns :",gaussian_spanwns)
# print("Gaussian spawns shape:",gaussian_spanwns.shape)
plt.xlim(bounds[0][0], bounds[1][0])
plt.ylim(bounds[0][1], bounds[1][1])
for i, linelist in enumerate(linelists):
    for line in linelist :
        colors = np.array((i / nb_lines*2, i/(nb_lines*3), (nb_lines-i) / (nb_lines*1.5)))
        colors/= colors.max()
        plt.plot((line[0][0], line[1][0]), (line[0][1], line[1][1]), c=colors, linewidth=0.4)
# plt.axis('off')
# plt.grid(which="major", alpha=0.3)
# plt.grid(which="minor", alpha=0.1)
# ax.minorticks_on()
ax.set_xticklabels([])
ax.set_yticklabels([])
# ax.set_frame_on(False)
ax.tick_params(tick1On=False)
ax.set_facecolor('lightgrey')
# ax.set_facecolor((0.9, 0.9, 0.9))
ax.set_aspect(1)
plt.savefig('F:\Personal Projects\Marching-Squares\mountains.png', dpi=2000)
plt.show()


