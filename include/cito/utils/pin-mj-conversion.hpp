// This file contains utility functions to convert states between Pinocchio and MuJoCo (for floating base only)
// Author: Jianghan Zhang
#ifndef PIN_MJ_CONVERSION_HPP
#define PIN_MJ_CONVERSION_HPP

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>  
#include <crocoddyl/core/mathbase.hpp>
#include <mujoco.h>

namespace cito {
namespace pin_mj_converter {
// Helper functions for quaternion conversion
Eigen::Vector4d wxyz2xyzw(const Eigen::Vector4d& quat_mj);
Eigen::Vector4d xyzw2wxyz(const Eigen::Vector4d& quat_pin);

// Change convention from MuJoCo to Pinocchio
Eigen::VectorXd change_convention_mj2pin(const Eigen::VectorXd& x);

// Change convention from Pinocchio to MuJoCo
Eigen::VectorXd change_convention_pin2mj(const Eigen::VectorXd& x);

// Convert state from MuJoCo to Pinocchio
Eigen::VectorXd stateMapping_mj2pin(const Eigen::VectorXd& x_mj, pinocchio::Model& pin_model);

// Convert state from Pinocchio to MuJoCo
Eigen::VectorXd stateMapping_pin2mj(const Eigen::VectorXd& x_pin, pinocchio::Model& pin_model);

// Numerical differentiation of stateMapping_mj2pin
Eigen::MatrixXd stateMappingDerivative_mj2pin_numDiff(const Eigen::VectorXd& x0_mj, pinocchio::Model& pin_model, const mjModel* mj_model);

// Numerical differentiation of stateMapping_pin2mj
Eigen::MatrixXd stateMappingDerivative_pin2mj_numDiff(const Eigen::VectorXd& x0_pin, pinocchio::Model& pin_model, const mjModel* mj_model);
}

void extract_forces(const mjModel* mj_model, mjData* mj_data, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& forces);
} // namespace cito

#endif // PIN_MJ_CONVERSION_HPP
