# import matplotlib.pyplot as plt
# import numpy as np
# import time

# # create the figure
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # im = ax.imshow(np.random.random((50,50)))
# im = ax.imshow(np.zeros((50,50)))
# plt.show(block=False)

# # draw some data in loop
# for i in range(10):
#     # wait for a second
#     time.sleep(0.1)
#     # replace the image contents
#     im.set_array(np.random.random((50,50)))
#     # redraw the figure
#     fig.canvas.draw()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Our 2-dimensional distribution will be over variables X and Y
N = 100
X = np.linspace(0, 1, N)
Y = np.linspace(0, 1, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([1/4, 1/4])
Sigma = np.array([[ 1/50 , 0], [0,  1/50]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)

plt.show()