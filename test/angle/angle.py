import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
import os


SIZE = (100, 100)
DELTA_MIN = 30
DELTA_MAX = 50

def parameter(min_val, max_val):
    val = np.random.randint(min_val, max_val + 1)
    p = max_val - min_val + 1
    return val, p

def angle(flags=[False, False], preset=None):
    var_y = flags[0]
    var_x = flags[1]
    parameters = 1
    Y_RANGE = (DELTA_MIN, DELTA_MAX)
    X_RANGE = (DELTA_MIN, DELTA_MAX)
    DOF = 90
    ANGLE = np.random.randint(1, DOF + 1)
    parameters *= DOF
    if preset:
        ANGLE = preset
    LENGTH = DELTA_MIN
    X = int(SIZE[1] / 2)
    if var_x:
        X, p = parameter(X_RANGE[0], X_RANGE[1])
        parameters *= p
    Y = int(SIZE[0] / 2)
    if var_y:
        Y, p = parameter(Y_RANGE[0], Y_RANGE[1])
        parameters *= p

    image = np.zeros(SIZE, dtype=bool)

    first_angle = np.random.randint(360)
    theta = -(np.pi / 180.0) * first_angle
    END = (Y - LENGTH * np.cos(theta), X - LENGTH * np.sin(theta))
    rr, cc = skimage.draw.line(Y, X, int(np.round(END[0])), int(np.round(END[1])))
    image[rr, cc] = 1

    second_angle = first_angle + ANGLE
    theta = -(np.pi / 180.0) * second_angle
    END = (Y - LENGTH * np.cos(theta), X - LENGTH * np.sin(theta))
    rr, cc = skimage.draw.line(Y, X, int(np.round(END[0])), int(np.round(END[1])))
    image[rr, cc] = 1


    sparse = [Y, X, ANGLE, first_angle]
    label = ANGLE
    return sparse, image, label, parameters

def main():
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    os.makedirs('images', exist_ok=True)

    for i in range(20):
        sparse, image, label, _ = angle()
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title('Angle: ' + str(label) + 'deg', fontsize=10)
        axes[i].axis('off')
        plt.imsave('images/angle_' + str(i+1) + '_' + str(label) + 'deg.png', image, cmap='gray')

    plt.tight_layout()

    plt.savefig('angles_grid.png', dpi=150, bbox_inches='tight')
    
    plt.show()

    print("Saved 20 images to 'images/' folder")

if __name__ == '__main__':
    main()