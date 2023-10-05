import argparse
import numpy as np
import open3d

from read_write_model import read_model, qvec2rotmat, rotmat2qvec


class Model3D:
    def __init__(self, camera, camera_scale = 1):
        self.__vis = None
        self.camera = camera
        self.camera_scale = camera_scale

    def add_points(self, xyz, rgb, remove_statistical_outlier=True):
        pcd = open3d.geometry.PointCloud()

        # xyz = []
        # rgb = rgb.tolist() 

        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)

        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                      std_ratio=2.0)

        # open3d.visualization.draw_geometries([pcd])
        self.__vis.add_geometry(pcd)
        self.__vis.poll_events()
        self.__vis.update_renderer()

    def add_camera(self, pose, gt = False, othermethod = False):

        plane_scale = 1
        # rotation
        R = qvec2rotmat(pose[0,3:])

        # translation
        t = pose[0,:3]

        # invert
        t = -R.T @ t
        R = R.T

        # intrinsics

        if self.camera['model'] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = self.camera['params'][0]
            cx = self.camera['params'][1]
            cy = self.camera['params'][2]
        elif self.camera['model'] in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
            fx = self.camera['params'][0]
            fy = self.camera['params'][1]
            cx = self.camera['params'][2]
            cy = self.camera['params'][3]
        else:
            raise Exception("Camera model not supported")

        # intrinsics
        K = np.identity(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        if othermethod:
            color = [0,1,0]
        else:
            color = [1, 0, 0] if gt else [0, 0, 1]
        # create axis, plane and pyramed geometries that will be drawn
        cam_model = draw_camera(K, R, t, self.camera['width']*plane_scale, self.camera['height']*plane_scale, self.camera_scale, color)
        for i in cam_model:
            self.__vis.add_geometry(i)

    def create_window(self):
        self.__vis = open3d.visualization.Visualizer()
        self.__vis.create_window(width=1400, height=800)

    def show(self):
        self.__vis.poll_events()
        self.__vis.update_renderer()
        self.__vis.run()
        self.__vis.destroy_window()


def draw_camera(K, R, t, w, h,
                scale=1, color=[1, 0, 0]):
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5 * scale)
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    # plane.paint_uniform_color([0.5,0,0])
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines))
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    # return [axis, plane, line_set]
    return [plane, line_set]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize COLMAP binary and text models")
    parser.add_argument("--input_model", required=True, help="path to input model folder")
    parser.add_argument("--input_format", choices=[".bin", ".txt"],
                        help="input model format", default="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # read COLMAP model
    model = Model()
    model.read_model(args.input_model, ext=args.input_format)

    print("num_cameras:", len(model.cameras))
    print("num_images:", len(model.images))
    print("num_points3D:", len(model.points3D))

    # display using Open3D visualization tools
    model.create_window()
    model.add_points()
    model.add_cameras(scale=0.25)
    model.show()


if __name__ == "__main__":
    main()