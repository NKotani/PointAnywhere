import os
import numpy as np
import cv2

def equi2pers(inimg, outimg, in_paramat, rotmat):
    in_height, in_width = inimg.shape[:2]
    out_height, out_width = outimg.shape[:2]

    x = np.arange(out_width)
    y = np.arange(out_height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    xyz = xyz @ in_paramat.T
    xyz = xyz @ rotmat.T

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    rotsphere_x = xyz_norm[..., 0:1]
    rotsphere_y = xyz_norm[..., 1:2]
    rotsphere_z = xyz_norm[..., 2:]

    polar_theta = np.arctan2(rotsphere_x, rotsphere_z)
    polar_phi = np.arcsin(rotsphere_y)
            
    X = (polar_theta / (2 * np.pi) + 0.5) * (in_width - 1)
    Y = (polar_phi / (np.pi) + 0.5) * (in_height - 1)
           
    outimg = cv2.remap(inimg, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    return outimg

def get_perspective(input:str, fov_horizontal:int, theta:float, phi:float, out_height:int, out_width:int, output='output/'):
    output_shape = [out_height, out_width]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    rotmat1, _ = cv2.Rodrigues(y_axis * np.radians(theta))
    rotmat2, _ = cv2.Rodrigues(np.dot(rotmat1, x_axis) * np.radians(phi))
    rotmat = rotmat2 @ rotmat1

    f = 0.5 * out_width / np.tan(0.5 * fov_horizontal / 180.0 * np.pi)
    in_paramat = np.array([[f, 0, (out_width-1)/2],
                        [0, f, (out_height-1)/2],
                        [0, 0, 1]], np.float32)
    in_paramat = np.linalg.inv(in_paramat)

    inimg = cv2.imread(input, cv2.IMREAD_COLOR)
    outimg = np.zeros(output_shape, np.uint8)
    outimg = equi2pers(inimg, outimg, in_paramat, rotmat)
    if not os.path.exists(output):
            os.mkdir(output)
    cv2.imwrite(os.path.join(output, os.path.splitext(os.path.basename(input))[0] + '_' + str(int(theta)) + '_' + str(int(phi)) + '.jpg'), outimg)
