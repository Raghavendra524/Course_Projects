import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io as io


def Canny_detector(img, weak_th=None, strong_th=None):
    # conversion of image to grayscale
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculating the gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    # Conversion of Cartesian coordinates to polar
    magnitude, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # setting the minimum and maximum thresholds
    # for double thresholding
    magnitude_max = np.max(magnitude)
    if not weak_th: weak_th = magnitude_max * 0.1
    if not strong_th: strong_th = magnitude_max * 0.5

    # getting the dimensions of the input image
    m, n = np.shape(img)

    # Looping through every pixel of the grayscale
    # image
    for p in range(n):
        for q in range(m):

            grad_ang = ang[q, p]
            grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang <= 22.5:
                nei1_x, nei1_y = p - 1, q
                nei2_x, nei2_y = p + 1, q

            # top right (diagonal-1) direction
            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                nei1_x, nei1_y = p - 1, q - 1
                nei2_x, nei2_y = p + 1, q + 1

            # In y-axis direction
            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                nei1_x, nei1_y = p, q - 1
                nei2_x, nei2_y = p, q + 1

            # top left (diagonal-2) direction
            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                nei1_x, nei1_y = p - 1, q + 1
                nei2_x, nei2_y = p + 1, q - 1

            # Now it restarts the cycle
            elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                nei1_x, nei1_y = p - 1, q
                nei2_x, nei2_y = p + 1, q

            # Non-maximum suppression step
            if n > nei1_x >= 0 and m > nei1_y >= 0:
                if magnitude[q, p] < magnitude[nei1_y, nei1_x]:
                    magnitude[q, p] = 0
                    continue

            if n > nei2_x >= 0 and m > nei2_y >= 0:
                if magnitude[q, p] < magnitude[nei2_y, nei2_x]:
                    magnitude[q, p] = 0

    # weak_ids = np.zeros_like(img)
    # strong_ids = np.zeros_like(img)
    ids = np.zeros_like(img)

    # double thresholding step
    for p in range(n):
        for q in range(m):

            grad_magnitude = magnitude[q, p]

            if grad_magnitude < weak_th:
                magnitude[q, p] = 0
            elif strong_th > grad_magnitude >= weak_th:
                ids[q, p] = 1
            else:
                ids[q, p] = 2

    # finally returning the magnitude of
    # gradients of edges
    return magnitude



# This is the function that will build the Hough Accumulator for the given image
def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    ''' A function for creating a Hough Accumulator for lines in an image. '''
    m, n = img.shape  # we need heigth and n to calculate the diag
    img_diagonal = np.ceil(np.sqrt(m ** 2 + n ** 2))  # a**2 + b**2 = c**2
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    # create the empty Hough Accumulator with dimensions equal to the size of
    # rhos and thetas
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # find all edge (nonzero) pixel indexes

    for i in range(len(x_idxs)):  # cycle through edge points
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)):  # cycle through thetas and calc rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas


def hough_peaks(H, num_peaks, threshold=0, nhood_size=3):
    ''' A function that returns the indicies of the accumulator array H that
        correspond to a local maxima.  If threshold is active all values less
        than this value will be ignored, if neighborhood_size is greater than
        (1, 1) this number of indicies around the maximum will be surpessed. '''
    # loop through number of peaks to identify
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1)  # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape)  # remap to shape of H
        indicies.append(H1_idx)

        # surpess indicies in neighborhood
        idx_y, idx_x = H1_idx  # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (nhood_size / 2)) < 0:
            min_x = 0
        else:
            min_x = int(idx_x - (nhood_size / 2))
        if ((idx_x + (nhood_size / 2) + 1) > H.shape[1]):
            max_x = H.shape[1]
        else:
            max_x = int(idx_x + (nhood_size / 2) + 1)

        # if idx_y is too close to the edges choose appropriate values
        if (idx_y - (nhood_size / 2)) < 0:
            min_y = 0
        else:
            min_y = int(idx_y - (nhood_size / 2))
        if ((idx_y + (nhood_size / 2) + 1) > H.shape[0]):
            max_y = H.shape[0]
        else:
            max_y = int(idx_y + (nhood_size / 2) + 1)

        # bound each index by the neighborhood size and set all values to 0
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                # remove neighborhoods in H1
                H1[y, x] = 0

                # highlight peaks in original H
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    # return the indicies and the original Hough space with selected points
    return indicies, H


# a simple funciton used to plot a Hough Accumulator
def plot_hough_acc(H, plot_title='Hough Accumulator Plot'):
    ''' A function that plot a Hough Space using Matplotlib. '''
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(plot_title)


    plt.imshow(H, cmap='gray')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.show()


# drawing the lines from the Hough Accumulatorlines using OpevCV cv2.line
def hough_lines_draw(img, indicies, rhos, thetas):
    ''' A function that takes indicies a rhos table and thetas table and draws
        lines on the input images that correspond to these values. '''
    for i in range(len(indicies)):
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)





# read in shapes image and convert to grayscale
shapes = io.imread("home.tif")
# shapes = cv2.resize(shapes,(800,800))

plt.subplot(2, 2, 1)
plt.imshow(shapes, cmap='gray')
plt.title("Input image")

# shapes_grayscale = cv2.cvtColor(shapes, cv2.COLOR_RGB2GRAY)
shapes_grayscale = shapes
# blur image (this will help clean up noise for Canny Edge Detection)
shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)

# find Canny Edges and show resulting image
# canny_edges = cv2.Canny(shapes_blurred, 20, 220)
canny_edges = Canny_detector(shapes_blurred)

plt.subplot(2, 2, 2)
plt.imshow(canny_edges, cmap='gray')
plt.title("Canny edge image")

# run hough_lines_accumulator on the shapes canny_edges imagnitudee
H, rhos, thetas = hough_lines_acc(canny_edges)
indicies, H = hough_peaks(H, 8, nhood_size=11)  # find peaks

# plot hough space, brighter spots have higher votes
plt.subplot(2, 2, 3)
plt.imshow(H, cmap='gray')
plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
plt.tight_layout()


hough_lines_draw(shapes, indicies, rhos, thetas)

# Show image with Hough Transform Lines
plt.subplot(2, 2, 4)
plt.imshow(shapes, cmap='gray')
plt.title("Output image")
plt.show()