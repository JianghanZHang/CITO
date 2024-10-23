/*
Author: Jianghan Zhang
This is a class for Integrated Action Model with Mujoco for Contact Dynamics
*/
// #include <pybind11/pybind11.h>
#include <boost/core/demangle.hpp>
#include <iostream>
#include <typeinfo>
#include <unordered_map>
#include <crocoddyl/core/utils/exception.hpp>
#include "cito/actions/integrated-contact-mujoco.hpp"
#include "cito/actions/differential-contact-mujoco.hpp"
#include <mujoco/mujoco.h>
#include "cito/utils/pin-mj-conversion.hpp"
#include <crocoddyl/core/mathbase.hpp>
#include "cito/utils/pin-mj-conversion.hpp"
using namespace crocoddyl;

namespace cito {

template <typename Scalar>
IntegratedActionModelContactMjTpl<Scalar>::IntegratedActionModelContactMjTpl(
    boost::shared_ptr<DifferentialActionModelAbstract> model,
    const Scalar time_step, const bool with_cost_residual)
    : Base(model, time_step, with_cost_residual) {
      forces_.setZero();

      boost::shared_ptr<DifferentialActionModelContactMj> differential_contact_mj = boost::dynamic_pointer_cast<DifferentialActionModelContactMj>(model);
     
      boost::shared_ptr<DifferentialActionModelContactMjTpl<Scalar>> contact_model =
      boost::dynamic_pointer_cast<DifferentialActionModelContactMjTpl<Scalar>>(model);
  
      if (!differential_contact_mj) {
        throw std::runtime_error("The differential model must be of type DifferentialActionModelContactMj.");
      }
      
      mjModel_ = differential_contact_mj->get_mjModel(); 
      mjData_ = differential_contact_mj->get_mjData();
      fids_ = differential_contact_mj->get_fids();
      mjData_->time = 0;

      std::memset(mjData_->qacc, 0, differential_contact_mj->get_state()->get_ndx() * sizeof(mjtNum));
      A_.resize(differential_contact_mj->get_state()->get_ndx(), differential_contact_mj->get_state()->get_ndx());
      B_.resize(differential_contact_mj->get_state()->get_ndx(), differential_contact_mj->get_nu());
      
      mj_state_.resize(differential_contact_mj->get_state()->get_nq() + differential_contact_mj->get_state()->get_nv());
      mj_state_next_.resize(differential_contact_mj->get_state()->get_nq() + differential_contact_mj->get_state()->get_nv());

      forces_.resize(6, 4);
    }
    
template <typename Scalar>
IntegratedActionModelContactMjTpl<Scalar>::~IntegratedActionModelContactMjTpl() {}

template <typename Scalar>
void IntegratedActionModelContactMjTpl<Scalar>::calc(
    const boost::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ", while x is "+ std::to_string(x.size())+")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  const std::size_t nv = state_->get_nv();
  const std::size_t nq = state_->get_nq();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());
      
  Data* d = static_cast<Data*>(data.get());

  // static cast to access the properties and methods not in the base class
  boost::shared_ptr<DifferentialActionDataContactMj> contact_differential_data = boost::static_pointer_cast<DifferentialActionDataContactMj>(d->differential);
  boost::shared_ptr<DifferentialActionModelContactMj> contact_differential = boost::static_pointer_cast<DifferentialActionModelContactMj>(differential_);
  
  // update pinocchio data
  pinocchio::forwardKinematics(contact_differential->pinocchio_, contact_differential_data->pinocchio, q, v);
  pinocchio::updateGlobalPlacements(contact_differential->pinocchio_, contact_differential_data->pinocchio);

  mj_state_ = pin_mj_converter::stateMapping_pin2mj(x, contact_differential->pinocchio_); // get the mujoco state from pinocchio state
  mjData_->qpos = mj_state_.head(nq).data();
  mjData_->qvel = mj_state_.tail(nv).data();
  Eigen::Map<Eigen::VectorXd>(mjData_->ctrl, u.size()) = u;

  contact_differential->costs_->calc(contact_differential_data->costs, x, u); // compute the costs
  d->differential->cost = contact_differential_data->costs->cost;   // Store in differential data
  d->cost = time_step_ * contact_differential_data->cost;              // multiply by time step and store in integrated data

  mj_forward(mjModel_, mjData_); // forward dynamics without integration    
  d->differential->xout = Eigen::Map<VectorXs>(mjData_->qacc, nv); // store the acceleration in differential data
  mj_Euler(mjModel_, mjData_); // integrate the mujoco model

  // Assign qpos to the first nq elements of mj_state_next_
  mj_state_next_.head(mjModel_->nq) = Eigen::VectorXd::Map(mjData_->qpos, mjModel_->nq);
  // Assign qvel to the last nv elements of mj_state_next_
  mj_state_next_.tail(mjModel_->nv) = Eigen::VectorXd::Map(mjData_->qvel, mjModel_->nv);
  
  VectorXs xnext = pin_mj_converter::stateMapping_mj2pin(mj_state_next_, contact_differential->pinocchio_); 
  // get the pinocchio state from mujoco state (convert back to pinocchio state to store in integrated data)

  d->xnext = xnext; // store the next state in integrated data
  if(contact_differential->constraints_ != nullptr){
    // contact_differential_data->constraints->resize(differential_, d->differential);
    contact_differential_data->constraints->resize(differential_.get(), d->differential.get());
    contact_differential->constraints_->calc(contact_differential_data->constraints, x, u);
    d->g = d->differential->g;
    d->h = d->differential->h;
  }


  if (with_cost_residual_) {
    d->r = d->differential->r;
  }
  set_contact(mjData_->contact); // store the contact in integrated data
  extract_forces(mjModel_, mjData_, forces_); // store the forces in integrated data
}

template <typename Scalar>
void IntegratedActionModelContactMjTpl<Scalar>::calc(
    const boost::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  const std::size_t nv = state_->get_nv();
  const std::size_t nq = state_->get_nq();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
      x.tail(state_->get_nv());

  Data* d = static_cast<Data*>(data.get());

  // static cast to access the properties and methods not in the base class
  boost::shared_ptr<DifferentialActionDataContactMj> contact_differential_data = boost::static_pointer_cast<DifferentialActionDataContactMj>(d->differential);
  boost::shared_ptr<DifferentialActionModelContactMj> contact_differential = boost::static_pointer_cast<DifferentialActionModelContactMj>(differential_);
  
  // update pinocchio data
  pinocchio::forwardKinematics(contact_differential->pinocchio_, contact_differential_data->pinocchio, q, v);
  pinocchio::updateGlobalPlacements(contact_differential->pinocchio_, contact_differential_data->pinocchio);

  mj_state_ = pin_mj_converter::stateMapping_pin2mj(x, contact_differential->pinocchio_); // get the mujoco state from pinocchio state
  mjData_->qpos = mj_state_.head(nq).data();
  mjData_->qvel = mj_state_.tail(nv).data();
  
  mjData_->time = 0;
  // mjData_->qacc.setZero();
  std::memset(mjData_->qacc, 0, nv * sizeof(mjtNum));
  std::memset(mjData_->ctrl, 0, differential_->get_nu() * sizeof(mjtNum)); // Assign the zero control to mjData->ctrl 
  
  contact_differential->costs_->calc(contact_differential_data->costs, x); // compute the costs
  d->differential->cost = contact_differential_data->costs->cost;   // Store in differential data
  d->cost = time_step_ * d->differential->cost;              // multiply by time step and store in integrated data

  mj_forward(mjModel_, mjData_); // forward dynamics without integration    
  d->differential->xout = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(mjData_->qacc, nv); // store the acceleration in differential data
  mj_Euler(mjModel_, mjData_); // integrate the mujoco model

  // Assign qpos to the first nq elements of mj_state_next_
  mj_state_next_.head(mjModel_->nq) = Eigen::VectorXd::Map(mjData_->qpos, mjModel_->nq);

  // Assign qvel to the last nv elements of mj_state_next_
  mj_state_next_.tail(mjModel_->nv) = Eigen::VectorXd::Map(mjData_->qvel, mjModel_->nv);
  
  VectorXs xnext = pin_mj_converter::stateMapping_mj2pin(mj_state_next_, contact_differential->pinocchio_); 
  // get the pinocchio state from mujoco state (convert back to pinocchio state to store in integrated data)

  d->xnext = xnext; // store the next state in integrated data
  if(contact_differential->constraints_ != nullptr){
    // contact_differential_data->constraints->resize(differential_, d->differential);
    contact_differential_data->constraints->resize(differential_.get(), d->differential.get());

    contact_differential->constraints_->calc(contact_differential_data->constraints, x);
    d->g = d->differential->g;
    d->h = d->differential->h;
  }


  if (with_cost_residual_) {
    d->r = d->differential->r;
  }
  set_contact(mjData_->contact); // store the contact in integrated data
  extract_forces(mjModel_, mjData_, forces_); // store the forces in integrated data
}

/*
calcDiff() should always be called after calc()
*/
template <typename Scalar>
void IntegratedActionModelContactMjTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  const std::size_t nv = state_->get_nv();
  const std::size_t nq = state_->get_nq();
  
  const VectorXs q = x.head(state_->get_nq());
  const VectorXs v = x.tail(state_->get_nv());

  Data* d = static_cast<Data*>(data.get());

   // static cast to access the properties and methods not in the base class
  boost::shared_ptr<DifferentialActionDataContactMj> contact_differential_data = boost::static_pointer_cast<DifferentialActionDataContactMj>(d->differential);
  boost::shared_ptr<DifferentialActionModelContactMj> contact_differential = boost::static_pointer_cast<DifferentialActionModelContactMj>(differential_);
  
  // get the state mapping derivatives for later use
  // compute dx_mj/dx_pin
  stateMappingDerivatives_ = pin_mj_converter::stateMappingDerivative_pin2mj_numDiff(x, contact_differential->pinocchio_, contact_differential->mjModel_); //dx_mj/dx_pin
  

  // compute the derivatives of the contact dynamics
  mj_state_ = pin_mj_converter::stateMapping_pin2mj(x, contact_differential->pinocchio_);

  mjData_->time = 0;
  std::memset(mjData_->qacc, 0, nv * sizeof(mjtNum));
  Eigen::Map<Eigen::VectorXd>(mjData_->ctrl, u.size()) = u; // Assign the control to mjData->ctrl 

  mjData_->qpos = mj_state_.head(nq).data();
  mjData_->qvel = mj_state_.tail(nv).data();

// // Print qpos
// std::cout << "qpos: ";
// for (int i = 0; i < mjModel_->nq; ++i) {
//     std::cout << mjData_->qpos[i] << " ";
// }
// std::cout << std::endl;

// // Print qvel
// std::cout << "qvel: ";
// for (int i = 0; i < mjModel_->nv; ++i) {
//     std::cout << mjData_->qvel[i] << " ";
// }
// std::cout << std::endl;

// // Print ctrl
// std::cout << "ctrl: ";
// for (int i = 0; i < mjModel_->nu; ++i) {
//     std::cout << mjData_->ctrl[i] << " ";
// }
// std::cout << std::endl;

// // Print qacc
// std::cout << "qacc: ";
// for (int i = 0; i < mjModel_->nv; ++i) {
//     std::cout << mjData_->qacc[i] << " ";
// }
// std::cout << std::endl;

  // std::cout << "time_step:" << mjModel_->opt.timestep << std::endl;

  // Matrix operations are not consistent in C++ and Python (column-major vs row-major)
  // This is becasue of mujoco's internal representation of matrices
  mjd_transitionFD(mjModel_, mjData_, 1e-8, 1, A_.data(), B_.data(), NULL, NULL);
  // A_ = A_.transpose().eval();  // Reassign A_ to its transpose
  // B_ = B_.transpose().eval();  // Reassign B_ to its transpose

  mj_state_next_ = pin_mj_converter::stateMapping_pin2mj(d->xnext, contact_differential->pinocchio_);
  // compute dx_next_pin/dx_next_mj
  stateMappingDerivativesNext_ = pin_mj_converter::stateMappingDerivative_mj2pin_numDiff(mj_state_next_, contact_differential->pinocchio_, contact_differential->mjModel_); 
  // std::cout << "2" << std::endl;

  d->Fx = stateMappingDerivativesNext_ * A_ * stateMappingDerivatives_; // compute Fx
  d->Fu = stateMappingDerivativesNext_ * B_; // compute Fu
  // std::cout << "3" << std::endl;

  contact_differential->costs_->calcDiff(contact_differential_data->costs, x, u); // compute the derivatives of the costs
  d->Lx = d->differential->Lx * time_step_; // store the Lx in integrated data
  d->Lu = d->differential->Lu * time_step_; // store the Lu in integrated data
  d->Lxx = d->differential->Lxx * time_step_; // store the Lxx in integrated data
  d->Lxu = d->differential->Lxu * time_step_; // store the Lxu in integrated data
  d->Luu = d->differential->Luu * time_step_; // store the Luu in integrated data

  if (contact_differential->constraints_ != nullptr) {
  contact_differential->constraints_->calcDiff(contact_differential_data->constraints, x, u); // compute the derivatives of the constraints
  d->Gx = d->differential->Gx; // store the Gx in integrated data
  d->Gu = d->differential->Gu; // store the Gu in integrated data
  d->Hx = d->differential->Hx; // store the Hx in integrated data
  d->Hu = d->differential->Hu; // store the Hu in integrated data
  }

  if (with_cost_residual_) {
    d->r = d->differential->r;
  }
}

template <typename Scalar>
void IntegratedActionModelContactMjTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }

  const std::size_t nv = state_->get_nv();
  Data* d = static_cast<Data*>(data.get());

   // static cast to access the properties and methods not in the base class
  boost::shared_ptr<DifferentialActionDataContactMj> contact_differential_data = boost::static_pointer_cast<DifferentialActionDataContactMj>(d->differential);
  boost::shared_ptr<DifferentialActionModelContactMj> contact_differential = boost::static_pointer_cast<DifferentialActionModelContactMj>(differential_);
  
  // get the state mapping derivatives for later use
  // compute dx_mj/dx_pin
  stateMappingDerivatives_ = pin_mj_converter::stateMappingDerivative_pin2mj_numDiff(x, contact_differential->pinocchio_, contact_differential->mjModel_); //dx_mj/dx_pin
  
  // Make sure that mjData_ corresponds to the current q, v, u
  mjData_->time = 0;
  std::memset(mjData_->qacc, 0, nv * sizeof(mjtNum));
  std::memset(mjData_->ctrl, 0, differential_->get_nu() * sizeof(mjtNum));

  // compute the derivatives of the contact dynamics
  mj_state_ = pin_mj_converter::stateMapping_pin2mj(x, contact_differential->pinocchio_);


  mjData_->qpos = mj_state_.head(state_->get_nq()).data();
  mjData_->qvel = mj_state_.tail(state_->get_nv()).data();

  mjd_transitionFD(mjModel_, mjData_, 1e-8, 1, A_.data(), NULL, NULL, NULL);


  mj_state_next_ = pin_mj_converter::stateMapping_pin2mj(d->xnext, contact_differential->pinocchio_);
  // compute dx_next_pin/dx_next_mj
  stateMappingDerivativesNext_ = pin_mj_converter::stateMappingDerivative_mj2pin_numDiff(mj_state_next_, contact_differential->pinocchio_, contact_differential->mjModel_); 
  
  d->Fx = stateMappingDerivativesNext_ * A_ * stateMappingDerivatives_; // compute Fx

  contact_differential->costs_->calcDiff(contact_differential_data->costs, x); // compute the derivatives of the costs
  d->Lx = d->differential->Lx * time_step_; // store the Lx in integrated data
  // d->Lu = d->differential->Lu * time_step_; // store the Lu in integrated data
  d->Lxx = d->differential->Lxx * time_step_; // store the Lxx in integrated data
  // d->Lxu = d->differential->Lxu * time_step_; // store the Lxu in integrated data
  // d->Luu = d->differential->Luu * time_step_; // store the Luu in integrated data

  if (contact_differential->constraints_ != nullptr) {
  contact_differential->constraints_->calcDiff(contact_differential_data->constraints, x); // compute the derivatives of the constraints
  d->Gx = d->differential->Gx; // store the Gx in integrated data
  // d->Gu = d->differential->Gu; // store the Gu in integrated data
  d->Hx = d->differential->Hx; // store the Hx in integrated data
  // d->Hu = d->differential->Hu; // store the Hu in integrated data
  }

  if (with_cost_residual_) {
    d->r = d->differential->r;
  }
}

// Getter for contact (raw pointer version)
template <typename Scalar>
const mjContact* IntegratedActionModelContactMjTpl<Scalar>::get_contact() const {
  return contact_;
}

// Setter for contact (raw pointer version)
template <typename Scalar>
void IntegratedActionModelContactMjTpl<Scalar>::set_contact(mjContact* contact) {
  contact_ = contact;  // Use raw pointer, no ownership transfer
}


// Getter for A_
template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXsRowMajor&
IntegratedActionModelContactMjTpl<Scalar>::get_A() const {
    return A_;
}

// Setter for A_
template <typename Scalar>
void IntegratedActionModelContactMjTpl<Scalar>::set_A(const MatrixXsRowMajor& A) {
    A_ = A;
}

// Getter for forces_
template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXsRowMajor&
IntegratedActionModelContactMjTpl<Scalar>::get_forces() const {
    return forces_;
}

// Setter for forces_
template <typename Scalar>
void IntegratedActionModelContactMjTpl<Scalar>::set_forces(const MatrixXsRowMajor& forces) {
    forces_ = forces;
}

// Getter for B_
template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXsRowMajor&
IntegratedActionModelContactMjTpl<Scalar>::get_B() const {
    return B_;
}

// Setter for B_
template <typename Scalar>
void IntegratedActionModelContactMjTpl<Scalar>::set_B(const MatrixXsRowMajor& B) {
    B_ = B;
}

// Getter for A_
template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
IntegratedActionModelContactMjTpl<Scalar>::get_stateMappingDerivatives() const {
    return stateMappingDerivatives_;
}

// Setter for A_
template <typename Scalar>
void IntegratedActionModelContactMjTpl<Scalar>::set_stateMappingDerivatives(const MatrixXs& M) {
    stateMappingDerivatives_ = M;
}

// Getter for A_
template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
IntegratedActionModelContactMjTpl<Scalar>::get_stateMappingDerivativesNext() const {
    return stateMappingDerivativesNext_;
}

// Setter for A_
template <typename Scalar>
void IntegratedActionModelContactMjTpl<Scalar>::set_stateMappingDerivativesNext(const MatrixXs& M) {
    stateMappingDerivativesNext_ = M;
}



template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar> >
IntegratedActionModelContactMjTpl<Scalar>::createData() {
  if (control_->get_nu() > differential_->get_nu())
    std::cerr << "Warning: It is useless to use an Euler integrator with a "
                 "control parametrization larger than PolyZero"
              << std::endl;
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool IntegratedActionModelContactMjTpl<Scalar>::checkData(
    const boost::shared_ptr<ActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (data != NULL) {
    return differential_->checkData(d->differential);
  } else {
    return false;
  }
}

// template <typename Scalar>
// void IntegratedActionModelContactMjTpl<Scalar>::quasiStatic(
//     const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
//     const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter,
//     const Scalar tol) {
//   if (static_cast<std::size_t>(u.size()) != nu_) {
//     throw_pretty("Invalid argument: "
//                  << "u has wrong dimension (it should be " +
//                         std::to_string(nu_) + ")");
//   }
//   if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
//     throw_pretty("Invalid argument: "
//                  << "x has wrong dimension (it should be " +
//                         std::to_string(state_->get_nx()) + ")");
//   }

//   const boost::shared_ptr<Data>& d = boost::static_pointer_cast<Data>(data);

//   d->control->w.setZero();
//   differential_->quasiStatic(d->differential, d->control->w, x, maxiter, tol);
//   control_->params(d->control, Scalar(0.), d->control->w);
//   u = d->control->u;
// }

template <typename Scalar>
void IntegratedActionModelContactMjTpl<Scalar>::print(std::ostream& os) const {
  os << "IntegratedActionModelEuler {dt=" << time_step_ << ", "
     << *differential_ << "}";
}

}  // namespace cito