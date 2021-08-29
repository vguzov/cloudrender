# On some systems, EGL does not start properly if OpenGL was already initialized, that's why it's better
# to keep EGLContext import on top
from cloudrenderer.libegl import EGLContext
import logging
import numpy as np
import sys
import os
import json
import smplpytorch
from cloudrenderer.render import SimplePointcloud, DirectionalLight
from cloudrenderer.render.smpl import AnimatableSMPLModel
from cloudrenderer.camera import PerspectiveCameraModel
from cloudrenderer.camera.trajectory import Trajectory
from cloudrenderer.scene import Scene
from cloudrenderer.capturing import AsyncPBOCapture
from videoio import VideoWriter
from OpenGL import GL as gl
from tqdm import tqdm
from cloudrenderer.utils import trimesh_load_from_zip, load_hps_sequence

logger = logging.getLogger("main_script")
logger.setLevel(logging.INFO)


# This example shows how to:
# - render pointcloud
# - render a sequence of frames with moving SMPL mesh
# - smoothly move the camera
# - dump rendered frames to a video


# First, let's set the target resolution, framerate, video length and initialize OpenGL context.
# We will use EGL offscreen rendering for that, but you can change it to whatever context you prefer (e.g. OsMesa, X-Server)
resolution = (1280,720)
fps = 30.
video_start_time = 6.
video_length_seconds = 12.
logger.info("Initializing EGL and OpenGL")
context = EGLContext()
if not context.initialize(*resolution):
    print("Error during context initialization")
    sys.exit(0)

# Now, let's create and set up OpenGL frame and renderbuffers
_main_cb, _main_db = gl.glGenRenderbuffers(2)
viewport_width, viewport_height = resolution

gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, _main_cb)
gl.glRenderbufferStorage(
    gl.GL_RENDERBUFFER, gl.GL_RGBA,
    viewport_width, viewport_height
)

gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, _main_db)
gl.glRenderbufferStorage(
    gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24,
    viewport_width, viewport_height
)

_main_fb = gl.glGenFramebuffers(1)
gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, _main_fb)
gl.glFramebufferRenderbuffer(
    gl.GL_DRAW_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
    gl.GL_RENDERBUFFER, _main_cb
)
gl.glFramebufferRenderbuffer(
    gl.GL_DRAW_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
    gl.GL_RENDERBUFFER, _main_db
)

gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, _main_fb)
gl.glDrawBuffers([gl.GL_COLOR_ATTACHMENT0])

# Let's configure OpenGL
gl.glEnable(gl.GL_BLEND)
gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
gl.glClearColor(1.0, 1.0, 1.0, 0)
gl.glViewport(0, 0, *resolution)
gl.glEnable(gl.GL_DEPTH_TEST)
gl.glDepthMask(gl.GL_TRUE)
gl.glDepthFunc(gl.GL_LESS)
gl.glDepthRange(0.0, 1.0)

# Create and set a position of the camera
camera = PerspectiveCameraModel()
camera.init_intrinsics(resolution, fov=75, far=50)
camera.init_extrinsics(np.array([1,np.pi/5,0,0]), np.array([0,-1,2]))

# Create a scene
main_scene = Scene()

# Load pointcloud
logger.info("Loading pointcloud")
renderable_pc = SimplePointcloud(camera=camera)
# Turn off shadow generation from pointcloud
renderable_pc.generate_shadows = False
renderable_pc.init_context()
pointcloud = trimesh_load_from_zip("test_assets/MPI_Etage6.zip", "*/pointcloud.ply")
renderable_pc.set_buffers(pointcloud)
main_scene.add_object(renderable_pc)


# Load human motion
logger.info("Loading SMPL animation")
# set different smpl_root to SMPL .pkl files folder if needed
# Make sure to fix the typo for male model while unpacking SMPL .pkl files:
# basicmodel_m_lbs_10_207_0_v1.0.0.pkl -> basicModel_m_lbs_10_207_0_v1.0.0.pkl
renderable_smpl = AnimatableSMPLModel(camera=camera, gender="male",
    smpl_root=os.path.join(os.path.dirname(smplpytorch.__file__), "native/models"))
# Turn off shadow drawing for SMPL model, as self-shadowing produces artifacts usually
renderable_smpl.draw_shadows = False
renderable_smpl.init_context()
motion_seq = load_hps_sequence("test_assets/SUB4_MPI_Etage6_working_standing.pkl", "test_assets/SUB4.json")
renderable_smpl.set_sequence(motion_seq, default_frame_time=1/30.)
# Let's set diffuse material for SMPL model
renderable_smpl.set_material(0.3,1,0,0)
main_scene.add_object(renderable_smpl)


# Let's add a directional light with shadows for this scene
light = DirectionalLight(np.array([0., -1., -1.]), np.array([0.8, 0.8, 0.8]))


# We'll create a 4x4x10 meter shadowmap with 1024x1024 texture buffer and center it above the model along the direction
# of the light. We will move the shadomap with the model in the main loop
smpl_model_shadowmap_offset = -light.direction*3
smpl_model_shadowmap = main_scene.add_dirlight_with_shadow(light=light, shadowmap_texsize=(1024,1024),
                                    shadowmap_worldsize=(4.,4.,10.),
                                    shadowmap_center=motion_seq[0]['translation']+smpl_model_shadowmap_offset)


# Set camera trajectory and fill in spaces between keypoints with interpolation
logger.info("Creating camera trajectory")
camera_trajectory = Trajectory()
camera_trajectory.set_trajectory(json.load(open("test_assets/TRAJ_SUB4_MPI_Etage6_working_standing.json")))
camera_trajectory.refine_trajectory(time_step=1/30.)


### Main drawing loop ###
logger.info("Running the main drawing loop")
# Create a video writer to dump frames to and an async capturing controller
with VideoWriter("test_assets/output.mp4", resolution=resolution, fps=fps) as vw, \
        AsyncPBOCapture(resolution, queue_size=50) as capturing:
    for current_time in tqdm(np.arange(video_start_time, video_start_time+video_length_seconds, 1/fps)):
        # Update dynamic objects
        renderable_smpl.set_time(current_time)
        smpl_model_shadowmap.camera.init_extrinsics(
            pose=renderable_smpl.translation_params.cpu().numpy()+smpl_model_shadowmap_offset)
        # Move the camera along the trajectory
        camera_trajectory.apply(camera, current_time)
        # Clear OpenGL buffers
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # Draw the scene
        main_scene.draw()
        # Request color readout; optionally receive previous request
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, _main_fb)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        color = capturing.request_color_async()
        # If received the previous frame, write it to the video
        if color is not None:
            vw.write(color[::-1])
    # Flush the remaining frames
    logger.info("Flushing PBO queue")
    color = capturing.get_first_requested_color()
    while color is not None:
        vw.write(color[::-1])
        color = capturing.get_first_requested_color()
logger.info("Done")
