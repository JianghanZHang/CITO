// This file containts utility functions to convert states between Pinocchio and MuJoCo (for floating base only)
#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>  
#include <mujoco.h>
#include "cito/utils/pin-mj-conversion.hpp"

#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <unordered_map>


namespace cito {
namespace pin_mj_converter {

    
// Helper functions for quaternion conversion
Eigen::Vector4d wxyz2xyzw(const Eigen::Vector4d& quat_mj) {
    // MuJoCo uses (w, x, y, z) -> Pinocchio uses (x, y, z, w)
    return Eigen::Vector4d(quat_mj[1], quat_mj[2], quat_mj[3], quat_mj[0]);
}

Eigen::Vector4d xyzw2wxyz(const Eigen::Vector4d& quat_pin) {
    // Pinocchio uses (x, y, z, w) -> MuJoCo uses (w, x, y, z)
    return Eigen::Vector4d(quat_pin[3], quat_pin[0], quat_pin[1], quat_pin[2]);
}

// Change convention from MuJoCo to Pinocchio
Eigen::VectorXd change_convention_mj2pin(const Eigen::VectorXd& x) {
    // std::cout << "Entered change_convention_mj2pin" << std::endl;
    
    Eigen::VectorXd x_tmp = x;
    Eigen::Vector4d quat_mj = x.segment<4>(3);  // Extract quaternion from MuJoCo
    Eigen::Vector4d quat_pin = wxyz2xyzw(quat_mj);  // Convert to Pinocchio convention
    x_tmp.segment<4>(3) = quat_pin;

    // std::cout << "Exiting change_convention_mj2pin" << std::endl;
    
    return x_tmp;
}

// Change convention from Pinocchio to MuJoCo
Eigen::VectorXd change_convention_pin2mj(const Eigen::VectorXd& x) {
    Eigen::VectorXd x_tmp = x;
    Eigen::Vector4d quat_pin = x.segment<4>(3);  // // Extract quaternion from Pinocchio
    Eigen::Vector4d quat_mj = xyzw2wxyz(quat_pin);  // Convert to MuJoCo convention (w, x, y, z)
    x_tmp.segment<4>(3) = quat_mj;
    return x_tmp;
}

// Convert state from MuJoCo to Pinocchio
Eigen::VectorXd stateMapping_mj2pin(const Eigen::VectorXd& x_mj, pinocchio::Model& pin_model) {
    pinocchio::Data rdata(pin_model);

    int nq = pin_model.nq, nv = pin_model.nv;

    Eigen::VectorXd x_pin = change_convention_mj2pin(x_mj);
    

    Eigen::VectorXd v_mj = x_mj.tail(nv);
    Eigen::VectorXd q_pin = x_pin.head(nq);

    // Update kinematics and frame placements
    pinocchio::framesForwardKinematics(pin_model, rdata, q_pin); // TODO: Find a way to remove this 2 lines (Passing rdata as argument to this function)
    pinocchio::updateFramePlacements(pin_model, rdata);

    // Get the rotation matrix of the base frame
    Eigen::Matrix3d R = rdata.oMi[1].rotation();
    Eigen::Vector3d BaseLinVel_mj = v_mj.head<3>();
    Eigen::Vector3d BaseLinVel_pin = R.transpose() * BaseLinVel_mj; // global to local base linear velocity

    // Update x_pin with the transformed velocity
    x_pin.segment<3>(19) = BaseLinVel_pin;
    return x_pin;
}

// Convert state from Pinocchio to MuJoCo
Eigen::VectorXd stateMapping_pin2mj(const Eigen::VectorXd& x_pin, pinocchio::Model& pin_model) {
    pinocchio::Data rdata(pin_model);
    int nq = pin_model.nq, nv = pin_model.nv;
    Eigen::VectorXd q_pin = x_pin.head(nq);
    Eigen::VectorXd v_pin = x_pin.tail(nv);

    // Update kinematics and frame placements
    pinocchio::forwardKinematics(pin_model, rdata, q_pin, v_pin);  // TODO: Find a way to remove this 2 lines
    pinocchio::updateFramePlacements(pin_model, rdata);

    Eigen::VectorXd x_mj = change_convention_pin2mj(x_pin);

    // Get the rotation matrix of the base frame
    Eigen::Matrix3d R = rdata.oMi[1].rotation();
    Eigen::Vector3d BaseLinVel_pin = x_pin.segment<3>(19);
    Eigen::Vector3d BaseLinVel_mj = R * BaseLinVel_pin; // local to global base linear velocity

    // Update x_mj with the transformed velocity
    x_mj.segment<3>(19) = BaseLinVel_mj;
    return x_mj;
}
// Numerical differentiation of stateMapping_mj2pin
Eigen::MatrixXd stateMappingDerivative_mj2pin_numDiff(const Eigen::VectorXd& x0_mj, pinocchio::Model& pin_model, const mjModel* mj_model) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(x0_mj.size()-1);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(dx.size(), dx.size());
    Eigen::VectorXd x0_pin = stateMapping_mj2pin(x0_mj, pin_model);
    Eigen::VectorXd x1_mj = x0_mj;
    Eigen::VectorXd x1_pin = x0_pin;
    Eigen::VectorXd diff = Eigen::VectorXd::Zero(dx.size());
    const std::size_t nq = pin_model.nq;
    const std::size_t nv = pin_model.nv;
    double h = 1e-8;

    // Eigen::VectorXd diff_q = Eigen::VectorXd::Zero(nv);
    // Eigen::VectorXd q0_pin = x0_pin.head(nq); 
    // Eigen::VectorXd q1_pin = x1_pin.head(nq); 

    for(int i = 0; i < dx.size(); i++) {
        dx[i] = h;
        x1_mj = x0_mj;
        // Compute the perturbed state for qpos and qvel seperately
        mj_integratePos(mj_model, x1_mj.head(nq).data(), dx.head(nv).data(), 1.0);
        x1_mj.tail(nv) = x0_mj.tail(nv) + dx.tail(nv);
        
        x1_pin = stateMapping_mj2pin(x1_mj, pin_model);
        // Compute difference for pos and vel seperately    
        pinocchio::difference(pin_model, x0_pin.head(nq), x1_pin.head(nq), diff.head(nv));
        
        diff.tail(nv) = x1_pin.tail(nv) - x0_pin.tail(nv);
        J.col(i) = diff / h;
        dx[i] = 0;
    }
    return J;
}

// Numerical differentiation of stateMapping_pin2mj
Eigen::MatrixXd stateMappingDerivative_pin2mj_numDiff(const Eigen::VectorXd& x0_pin, pinocchio::Model& pin_model, const mjModel* mj_model){
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(x0_pin.size()-1);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(dx.size(), dx.size());
    Eigen::VectorXd x0_mj = stateMapping_pin2mj(x0_pin, pin_model);
    Eigen::VectorXd x1_mj = x0_mj;
    Eigen::VectorXd x1_pin = x0_pin;
    Eigen::VectorXd diff = Eigen::VectorXd::Zero(dx.size());
    const std::size_t nq = pin_model.nq;
    const std::size_t nv = pin_model.nv;
    double h = 1e-8;

    for(int i = 0; i < dx.size(); i++) {
        dx[i] = h;
        x1_pin = x0_pin;
        pinocchio::integrate(pin_model, x1_pin.head(nq), dx.head(nv), x1_pin.head(nq));
        x1_pin.tail(nv) = x0_pin.tail(nv) + dx.tail(nv);

        x1_mj = stateMapping_pin2mj(x1_pin, pin_model);
        mj_differentiatePos(mj_model, diff.head(nq).data(), 1.0, x0_mj.head(nq).data(), x1_mj.head(nq).data());
        diff.tail(nv) = x1_mj.tail(nv) - x0_mj.tail(nv);
        J.col(i) = diff / h;
        dx[i] = 0;
    }
    return J;
}

} // namespace pin_mj_converter

void extract_forces(const mjModel* mj_model, mjData* mj_data, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& forces) {
    // Create the geom2_to_index map
    // std::cout << "Entered extract_forces" << std::endl;
    std::unordered_map<int, int> geom2_to_index = {
        {20, 0},
        {32, 1},
        {44, 2},
        {56, 3}
    };

    // Iterate through contacts
    for (int i = 0; i < mj_data->ncon; ++i) {
        // Get the geom2 ID for the contact
        int geom2 = mj_data->contact[i].geom[1];  // geom2 is the second geom involved in the contact

        // Check if geom2 is in the map
        if (geom2_to_index.find(geom2) != geom2_to_index.end()) {
            int index = geom2_to_index[geom2];

            // Get the contact force
            mjtNum contact_force[6];  // Forces and torques (6D vector)
            mj_contactForce(mj_model, mj_data, i, contact_force);

            // Copy the contact force to the forces matrix
            for (int j = 0; j < 6; ++j) {
                forces(j, index) = contact_force[j];
            }

            // Get the contact frame (3x3 rotation matrix)
            Eigen::Matrix3d R;
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    R(row, col) = mj_data->contact[i].frame[row * 3 + col];  // 1D array to 3x3 matrix
                }
            }

            // Apply the rotation to the first 3 components (force part) of the contact force
            Eigen::Vector3d force = forces.col(index).head<3>();  // First 3 components are force
            force = R * force;  // Transform the force according to the contact frame

            // Update the transformed force back to the forces matrix
            forces.col(index).head<3>() = force;
        }
    }
    // std::cout << "Exiting extract_forces" << std::endl;
}


} // namespace cito