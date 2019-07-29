#Copyright
"""
This file collects different visualization functions
"""

import vtk
import time
from random import randint
from vtk.util.numpy_support import vtk_to_numpy
CLASS_COLORS = {
    'tr': (0xff, 0xd7, 0x00),  # Yellow-ish
    'white': (255, 255, 255),
    'gt': (0, 255, 0)  # Light-green
}

ID_COLORS = {}
INITIAL_CAMERA = {
    'pos': (-90, 90, 0),
    'dir': (0, 0, 0)
 }

global_cur_frame = 0

def visualize(story, gt_objs = None, det_objs =None):
    """
    Function for scene visualization
    :param story: Data frames to be visualized
    :return: None
    """
    global global_cur_frame
    global_cur_frame = 0
    global CLASS_COLORS
    if story is None or len(story) == 0:
        return
    camera = vtk.vtkCamera()
    reset_camera(camera)
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1024, 1024)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.SetInteractorStyle(
        KeyInteractorStyle(render_window_interactor, story, gt_objs, det_objs, camera))
    if gt_objs is None:
        used_gt_obj = None
    else:
        used_gt_obj = gt_objs
    if det_objs is None:
        used_det_obj = None
    else:
        used_det_obj = det_objs
    process_frame(render_window, camera, story[0], used_gt_obj[0], used_det_obj, write_framenum=True)
    render_window_interactor.Initialize()
    render_window_interactor.Start()


def get_frame_seq(story, gt_objs = None, det_objs =None):
    """
    Function for scene visualization into video
    :param story: Data frames to be visualized
    :param filename: path to output file
    :return: None
    """
    global CLASS_COLORS
    camera = vtk.vtkCamera()
    reset_camera(camera)
    render_window = vtk.vtkRenderWindow()
    render_window.OffScreenRenderingOn()
    render_window.SetSize(540, 1080)
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.SetInputBufferTypeToRGB()
    window_to_image_filter.ReadFrontBufferOff()
    window_to_image_filter.Update()

    out_frames = []
    for i in range(len(story)):
        if gt_objs is None:
            used_gt_obj = None
        else:
            used_gt_obj = gt_objs[i]
        if det_objs is None:
            used_det_obj = None
        else:
            used_det_obj = det_objs[i]
        out_frames.append(get_frame(render_window, camera, story[i], used_gt_obj, used_det_obj))
    return out_frames


def get_frame(render_window, camera, data, gt_obj, det_obj):
    renderer = vtk.vtkRenderer()
    render_window.AddRenderer(renderer)
    render_window.SetOffScreenRendering(1)
    renderer.SetActiveCamera(camera)
    bg_actor = create_vtk_bg_actor(data)
    renderer.AddActor(bg_actor)
    if gt_obj is not None:
        gt_actor = create_vtk_box_actor(gt_obj.bbox)
        renderer.AddActor(gt_actor)
    if det_obj is not None:
        det_actor = create_vtk_box_actor(det_obj.bbox, color=CLASS_COLORS['tr'])
        renderer.AddActor(det_actor)
    vtk_win_im = vtk.vtkWindowToImageFilter()
    vtk_win_im.SetInput(render_window)
    vtk_win_im.Update()
    vtk_image = vtk_win_im.GetOutput()
    width, height, _ = vtk_image.GetDimensions()
    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()
    arr = vtk_to_numpy(vtk_array).reshape(height, width, components)
    return arr


def process_frame(render_window, camera, data, gt_objs, det_objs, write_framenum=False, video_mode=False):
    renderer = vtk.vtkRenderer()
    render_window.AddRenderer(renderer)
    renderer.SetActiveCamera(camera)
    bg_actor = create_vtk_bg_actor(data)
    renderer.AddActor(bg_actor)

    if gt_objs is not None:
        for gt_obj in gt_objs:
            gt_actor = create_vtk_box_actor(gt_obj.bbox)
            renderer.AddActor(gt_actor)
    if det_objs is not None:
        for det_obj in det_objs:
            det_actor = create_vtk_box_actor(det_obj.bbox, color = CLASS_COLORS['tr'])
            renderer.AddActor(det_actor)
    if not video_mode:
        time.sleep(0.01)
    render_window.Render()

def create_vtk_box_actor(bbox, color = CLASS_COLORS['gt']):
    """
    Creates vtkActor with drawn ground truth box (lines)
    :param box3d: object of type BBox3d
    :return: vtkActor with drawn bbox
    """
    rgb = color
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    vtk_colors = init_vtk_colors()
    corners = bbox
    ids = []
    # Init corners points
    for p in corners:
        ind = points.InsertNextPoint([p[0], p[2], -p[1]])
        ids.append(ind)

    # Draw lines
    pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    # 2D_BOX:
    # pairs = [(5, 6), (5, 1), (6, 2), (1, 2)]
    for p in pairs:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, ids[p[0]])
        line.GetPointIds().SetId(1, ids[p[1]])
        vtk_colors.InsertNextTypedTuple(rgb)
        lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetCellData().SetScalars(vtk_colors)
    polydata.Modified()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2)
    return actor


def draw_colored_points(in_points, color, points_obj, vertices, colors):
    """
    Update vtk classes with numpy points or array
    :param in_points: array of 3d points in lidar coordinates
    :param color: tuple (red, green, blue)
    :param points_obj: vtk object to append points
    :param vertices: vtk object to append corresponding vertices
    :param colors: vtk object to append colors of every point
    """
    r, g, b = color
    for p in in_points:
        ind = points_obj.InsertNextPoint([p[0], p[2], -p[1]])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(ind)
        colors.InsertNextTuple3(r, g, b)


def create_vtk_bg_actor(in_points):
    """
    Creates vtk actor for visualizing bg point in white
    :param in_points: point cloud for scene
    :return: vtkActor filled with points
    """
    bg_pntsize = 1
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    vtk_colors = init_vtk_colors()
    rgb = CLASS_COLORS['white']
    draw_colored_points(in_points, rgb, points, vertices, vtk_colors)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    polydata.GetPointData().SetScalars(vtk_colors)
    polydata.Modified()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(bg_pntsize)
    return actor





def init_vtk_colors():
    vtk_colors = vtk.vtkUnsignedCharArray()
    vtk_colors.SetNumberOfComponents(3)
    vtk_colors.SetName("Colors")
    return vtk_colors


def reset_camera(camera):
    pos = INITIAL_CAMERA['pos']
    dir = INITIAL_CAMERA['dir']
    camera.SetPosition(pos[0], pos[1], pos[2])
    camera.SetFocalPoint(dir[0], dir[1], dir[2])


global_iren = None
global_camera = None
global_data = None
global_gt_box = None
global_det_box = None


class KeyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent, data, gt_objs, det_objs, camera):
        global global_iren
        global global_camera
        global global_data
        global global_gt_box
        global global_det_box

        global_data = data
        global_iren = parent
        global_camera = camera
        global_gt_box = gt_objs
        global_det_box = det_objs

        self.AddObserver("KeyPressEvent", self.keyPressEvent)

    def keyPressEvent(self, obj, event):
        global global_iren
        global global_cur_frame
        global global_camera
        global global_data
        global global_gt_box
        global global_det_box
        key = global_iren.GetKeySym()
        playing = False
        do_smth = False
        if key == 'Left':  # Arrow Left
            global_cur_frame = max(0, global_cur_frame - 1)
            do_smth = True
        elif key == 'Right':  # Arrow right
            global_cur_frame = min(len(global_data) - 1, global_cur_frame + 1)
            do_smth = True
        elif key == 'r':  # Reset camera
            reset_camera(global_camera)
            do_smth = True
        elif key == 'p':  # Play
            playing = True
            do_smth = True
        while do_smth:
            ren_win = global_iren.GetRenderWindow()

            if global_gt_box is None:
                process_frame(ren_win,
                              global_camera,
                              global_data[global_cur_frame],
                              None,
                              global_det_box[global_cur_frame],
                              write_framenum=True)
            if global_det_box is None:
                process_frame(ren_win,
                              global_camera,
                              global_data[global_cur_frame],
                              global_gt_box[global_cur_frame],
                              None,
                              write_framenum=True)
            if global_gt_box is None and global_det_box is None:
                process_frame(ren_win,
                              global_camera,
                              global_data[global_cur_frame],
                              None,
                              None,
                              write_framenum=True)
            if global_gt_box is not None and global_det_box is not None:
                process_frame(ren_win,
                              global_camera,
                              global_data[global_cur_frame],
                              global_gt_box[global_cur_frame],
                              global_det_box[global_cur_frame],
                              write_framenum=True)
            if playing:
                if global_cur_frame >= len(global_data) - 1:
                    return
                global_cur_frame += 1
            else:
                do_smth = False
