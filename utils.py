## This class draws arrows to depic the force in meshcat
## Author : Huaijiang Zhu

import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf
import pinocchio as pin
import meshcat
import mujoco
from robot_env import ROOT_JOINT_INDEX

class Arrow(object):
    def __init__(self, meshcat_vis, name, 
                 location=[0,0,0], 
                 vector=[0,0,1],
                 length_scale=1,
                 color=0xff22dd):
        
        self.vis = meshcat_vis[name]
        self.cone = self.vis["cone"]
        self.line = self.vis["line"]
        self.material = g.MeshBasicMaterial(color=color, reflectivity=0.5)
        
        self.location, self.length_scale = location, length_scale
        self.anchor_as_vector(location, vector)
    
    def _update(self):
        translation = tf.translation_matrix(self.location)
        rotation = self.orientation
        offset = tf.translation_matrix([0, self.length/2, 0])
        self.pose = translation @ rotation @ offset
        self.vis.set_transform(self.pose)
        
    def set_length(self, length, update=True):
        self.length = length * self.length_scale
        cone_scale = self.length/0.08
        self.line.set_object(g.Cylinder(height=self.length, radius=0.005), self.material)
        self.cone.set_object(g.Cylinder(height=0.015, 
                                        radius=0.01, 
                                        radiusTop=0., 
                                        radiusBottom=0.01),
                             self.material)
        self.cone.set_transform(tf.translation_matrix([0.,cone_scale*0.04,0]))
        if update:
            self._update()
        
    def set_direction(self, direction, update=True):
        orientation = np.eye(4)
        if np.linalg.norm(direction - np.array([1.0, 0, 0])) < 1e-1 or np.linalg.norm(direction - np.array([-1.0, 0, 0])) < 1e-1:
            orientation[:3, 0] = np.cross([0.0,1.0,0], direction)
        else:
            orientation[:3, 0] = np.cross([1.0,0,0], direction)
        orientation[:3, 1] = direction.copy()
        orientation[:3, 2] = np.cross(orientation[:3, 0], orientation[:3, 1])
        self.orientation = orientation
        if update:
            self._update()
    
    def set_location(self, location, update=True):
        self.location = location
        if update:
            self._update()
        
    def anchor_as_vector(self, location, vector, update=True):
        self.set_direction(np.array(vector)/np.linalg.norm(vector), False)
        self.set_location(location, False)
        self.set_length(np.linalg.norm(vector), False)
        if update:
            self._update()

    def delete(self):
        self.vis.delete()


def change_convention_pin2mj(x):
    x_tmp = x.copy()
    quat = x[3:7]
    quat = xyzw2wxyz(quat)
    x_tmp[3:7] = quat
    return x_tmp

def change_convention_mj2pin(x):
    x_tmp = x.copy()
    quat = x[3:7]
    quat = wxyz2xyzw(quat)
    x_tmp[3:7] = quat
    return x_tmp

# Converting quaternion from mujoco to pinocchio
def wxyz2xyzw(quat):
    return np.roll(quat, -1)

# Converting quaternion from pinocchio to mujoco
def xyzw2wxyz(quat):
    return np.roll(quat, 1)

def stateMapping_mj2pin(x_mj, rmodel):
    rdata = rmodel.createData()
    nq, nv = rmodel.nq, rmodel.nv
    x_pin = change_convention_mj2pin(x_mj)
    v_mj = x_mj[-nv:]
    q_pin = x_pin[:nq]

    pin.framesForwardKinematics(rmodel, rdata, q_pin)

    pin.updateFramePlacements(rmodel, rdata)
    R = rdata.oMi[ROOT_JOINT_INDEX].rotation
    BaseLinVel_mj = v_mj[:3]
    BaseLinVel_pin = R.T @ BaseLinVel_mj  # Change the base linear velocity to body frame
    x_pin[19:22] = BaseLinVel_pin.copy()
    
    M_ = np.eye(2*nv)
    M_[nv:nv+3, nv:nv+3] = R.T
    M_inv_ = np.eye(2*nv)
    M_inv_[nv:nv+3, nv:nv+3] = R
    return np.array(x_pin), M_, M_inv_

def stateMapping_pin2mj(x_pin, rmodel):
    rdata = rmodel.createData()
    nq, nv = rmodel.nq, rmodel.nv
    q_pin, v_pin = x_pin[:nq], x_pin[-nv:]
    pin.forwardKinematics(rmodel, rdata, q_pin, v_pin)
    pin.updateFramePlacements(rmodel, rdata)

    x_mj = change_convention_pin2mj(x_pin)
    R = rdata.oMi[ROOT_JOINT_INDEX].rotation
    BaseLinVel_pin = x_pin[19:22]
    # BaseLinVel_mj = pin.getFrameVelocity(rmodel, rdata, ROOT_JOINT_INDEX, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).linear
    BaseLinVel_mj = R @ BaseLinVel_pin # Change the base linear velocity to global frame
    x_mj[19:22] = BaseLinVel_mj.copy()

    M_ = np.eye(2*nv)
    M_[nv:nv+3, nv:nv+3] = R
    M_inv_ = np.eye(2*nv)
    M_inv_[nv:nv+3, nv:nv+3] = R.T
    return np.array(x_mj), M_, M_inv_


# def _numdiffSE3toEuclidian_tensor(f, x0, rmodel, h=1e-8):
#     f0 = f(x0).copy()
#     nq, nv = rmodel.nq, rmodel.nv
#     dx = np.zeros(2 * nv)
#     Fx = np.zeros((2 * nv, 2 * nv, 2 * nv))  # Initialize the tensor

#     for ix in range(2 * nv):
#         dx[ix] += h
#         x_prime = np.hstack((pin.integrate(rmodel, x0[:nq].copy(), dx[:nv]), x0[nq:].copy() + dx[nv:]))
        
#         f_prime = f(x_prime).copy()
#         tmp = (f_prime - f0) / h

#         Fx[:, :, ix] = tmp.reshape(2 * nv, 2 * nv)
#         dx[ix] = 0.0

#     return Fx

# def compute_dM_inv_dx_pin_numDiff(x_pin, rmodel, h=1e-8):
#     def f(x_pin):
#         _, _, Minv = stateMapping_pin2mj(x_pin, rmodel)
#         return Minv

#     dM_inv_dx_pin_tensor = _numdiffSE3toEuclidian_tensor(f, x_pin, rmodel, h)
    
#     return dM_inv_dx_pin_tensor
