// author: Jianghan Zhang
// This is a Dummy class for Integrated Action Model with Mujoco for Contact Dynamics


#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/utils/exception.hpp>

#include "cito/actions/differential-contact-mujoco.hpp"

#include <iostream>
#include <typeinfo>

namespace cito{

// template <typename Scalar>
// DifferentialActionModelContactMjTpl<Scalar>::
//     DifferentialActionModelContactMjTpl(
//         const mjModel* mj_model,
//         mjData* mj_data,
//         boost::shared_ptr<StateMultibody> state,
//         boost::shared_ptr<ActuationModelAbstract> actuation,
//         boost::shared_ptr<CostModelSum> costs,
//         boost::shared_ptr<ConstraintModelManager> constraints
//         )
//     : Base(state, actuation->get_nu(), costs->get_nr()),
//       // mjModel_(mj_model),
//       // mjData_(mj_data),
//       mjModel_(boost::shared_ptr<const mjModel>(mj_model)),
//       mjData_(boost::shared_ptr<mjData>(mj_data)),
//       actuation_(actuation),
//       costs_(costs),
//       constraints_(constraints),
//       pinocchio_(*state->get_pinocchio().get()),
//       without_armature_(true),
//       armature_(VectorXs::Zero(state->get_nv())) {

//   // Print the types of mjModel and mjData in C++
//   std::cout << "mjModel type in C++: " << typeid(mj_model).name() << std::endl;
//   std::cout << "mjData type in C++: " << typeid(mj_data).name() << std::endl;

//   // Check if mjModel and mjData are valid
//   if (!mj_model || !mj_data) {
//     throw std::invalid_argument("mjModel or mjData is null");
//   }

//   if (costs_->get_nu() != nu_) {
//     throw_pretty(
//         "Invalid argument: "
//         << "Costs doesn't have the same control dimension (it should be " +
//                std::to_string(nu_) + ")");
//   }
//   Base::set_u_lb(Scalar(-1.) * pinocchio_.effortLimit.tail(nu_));
//   Base::set_u_ub(Scalar(+1.) * pinocchio_.effortLimit.tail(nu_));
//   // const boost::shared_ptr<crocoddyl::StateMultibody>& s =
//   // boost::dynamic_pointer_cast<crocoddyl::StateMultibody>(state);
//   // if (s) {
//   //   pin_model_ = s->get_pinocchio();
//   // }
//   // else {
//   //   throw_pretty("Invalid argument: "
//   //             << "the state is not derived from crocoddyl::StateMultibody");
//   // }
// }
namespace bp = boost::python;

template <typename _Scalar>
DifferentialActionModelContactMjTpl<_Scalar>::DifferentialActionModelContactMjTpl(
    // const pybind11::object mj_model_wrapper, pybind11::object mj_data_wrapper,
    const bp::object mj_model,
    // bp::object mj_data,
    boost::shared_ptr<StateMultibody> state,
    boost::shared_ptr<ActuationModelAbstract> actuation,
    boost::shared_ptr<CostModelSum> costs,
    boost::shared_ptr<ConstraintModelManager> constraints)
    : Base(state, actuation->get_nu(), costs->get_nr()),
      actuation_(actuation),
      costs_(costs),
      constraints_(constraints),
      pinocchio_(*state->get_pinocchio().get()),
      without_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())) {
  std::uintptr_t m_raw = boost::python::extract<std::uintptr_t>(mj_model.attr("_address"));
  // std::uintptr_t d_raw = boost::python::extract<std::uintptr_t>(mj_data.attr("_address"));

  // // Step 2: Print the raw addresses in hexadecimal format
  // std::cout << std::showbase << std::hex;
  // std::cout << "m_raw address: " << m_raw << std::endl;
  // std::cout << "d_raw address: " << d_raw << std::endl;
  // std::cout << std::dec;

  // Step 3: Cast raw addresses back to C++ pointers (mjModel* and mjData*)
  const mjModel* m_cpp_ = reinterpret_cast<const mjModel*>(m_raw);
  // mjData* d_cpp_ = reinterpret_cast<mjData*>(d_raw);

  // // Step 4: Access and print the C++ attributes of mjModel
  // std::cout << "cpp model attributes: " << std::endl;
  // std::cout << "\tnq: " << m_cpp_->nq << std::endl;
  // std::cout << "\tnv: " << m_cpp_->nv << std::endl;
  // std::cout << "\tnbody: " << m_cpp_->nbody << std::endl;
  // std::cout << "\tngeom: " << m_cpp_->ngeom << std::endl;
  // std::cout << "\tqpos0: [";
  // for (size_t i = 0; i < m_cpp_->nq; i++) {
  //     std::cout << m_cpp_->qpos0[i] << " ";
  // }
  // std::cout << "]" << std::endl;

  // // Step 5: Access and print the C++ attributes of mjData
  // std::cout << "cpp data attributes: " << std::endl;
  // std::cout << "\tnbuffer: " << d_cpp_->nbuffer << std::endl;
  // std::cout << "\ttime: " << d_cpp_->time << std::endl;
  // std::cout << "\tqpos: [";
  // for (size_t i = 0; i < m_cpp_->nq; i++) {
  //     std::cout << d_cpp_->qpos[i] << " ";
  // }
  // std::cout << "]" << std::endl;

  // Extract the raw pointers from the pybind11-wrapped objects using .get()
  // const mjModel* mj_model = mj_model_wrapper.attr("get")().cast<mjModel*>();
  // mjData* mj_data = mj_data_wrapper.attr("get")().cast<mjData*>();

  mjModel_ = m_cpp_;
  mjData_ = mj_makeData(mjModel_);
  // mjData_ = d_cpp_;

  // // Ensure the pointers are valid
  // if (!mj_model || !mj_data) {
  //   throw std::invalid_argument("mjModel or mjData is null");
  // }

  // Log the types and confirm successful extraction
  // std::cout << "mjModel type in C++: " << typeid(mj_model).name() << std::endl;
  // std::cout << "mjData type in C++: " << typeid(mj_data).name() << std::endl;

  if (costs_->get_nu() != nu_) {
    throw_pretty("Invalid argument: Costs don't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }

  Base::set_u_lb(Scalar(-1.) * pinocchio_.effortLimit.tail(nu_));
  Base::set_u_ub(Scalar(+1.) * pinocchio_.effortLimit.tail(nu_));
}


template <typename Scalar>
DifferentialActionModelContactMjTpl<Scalar>::~DifferentialActionModelContactMjTpl() {}

///////////////////////////////////////////////
// This should be a Dummy method (never used)//
///////////////////////////////////////////////


template <typename Scalar>
void DifferentialActionModelContactMjTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  throw_pretty("This method is a dummy method, it should not be called!");
}

///////////////////////////////////////////////
// This should be a Dummy method (never used)//
///////////////////////////////////////////////
template <typename Scalar>
void DifferentialActionModelContactMjTpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  throw_pretty("This method is a dummy method, it should not be called!");
}

///////////////////////////////////////////////
// This should be a Dummy method (never used)//
///////////////////////////////////////////////
template <typename Scalar>
void DifferentialActionModelContactMjTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  throw_pretty("This method should never be called!");
}


///////////////////////////////////////////////
// This should be a Dummy method (never used)//
///////////////////////////////////////////////
template <typename Scalar>
void DifferentialActionModelContactMjTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  throw_pretty("This method should never be called!");
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelContactMjTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool DifferentialActionModelContactMjTpl<Scalar>::checkData(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void DifferentialActionModelContactMjTpl<Scalar>::quasiStatic(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data,
    Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
    const std::size_t, const Scalar) {
    throw_pretty("Quasi static method is not implemented yet");
}

template <typename Scalar>
std::size_t DifferentialActionModelContactMjTpl<Scalar>::get_ng() const {
  if (constraints_ != nullptr) {
    return constraints_->get_ng();
  } else {
    return Base::get_ng();
  }
}

template <typename Scalar>
std::size_t DifferentialActionModelContactMjTpl<Scalar>::get_nh() const {
  if (constraints_ != nullptr) {
    return constraints_->get_nh();
  } else {
    return Base::get_nh();
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactMjTpl<Scalar>::get_g_lb() const {
  if (constraints_ != nullptr) {
    return constraints_->get_lb();
  } else {
    return g_lb_;
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactMjTpl<Scalar>::get_g_ub() const {
  if (constraints_ != nullptr) {
    return constraints_->get_ub();
  } else {
    return g_lb_;
  }
}

template <typename Scalar>
void DifferentialActionModelContactMjTpl<Scalar>::print(
    std::ostream& os) const {
  os << "DifferentialActionModelFreeFwdDynamics {nx=" << state_->get_nx()
     << ", ndx=" << state_->get_ndx() << ", nu=" << nu_ << "}";
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>&
DifferentialActionModelContactMjTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

template <typename Scalar>
const boost::shared_ptr<ActuationModelAbstractTpl<Scalar> >&
DifferentialActionModelContactMjTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const boost::shared_ptr<CostModelSumTpl<Scalar> >&
DifferentialActionModelContactMjTpl<Scalar>::get_costs() const {
  return costs_;
}

template <typename Scalar>
const boost::shared_ptr<ConstraintModelManagerTpl<Scalar> >&
DifferentialActionModelContactMjTpl<Scalar>::get_constraints() const {
  return constraints_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DifferentialActionModelContactMjTpl<Scalar>::get_armature() const {
  return armature_;
}

template <typename Scalar>
void DifferentialActionModelContactMjTpl<Scalar>::set_armature(
    const VectorXs& armature) {
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "The armature dimension is wrong (it should be " +
                        std::to_string(state_->get_nv()) + ")");
  }

  armature_ = armature;
  without_armature_ = false;
}

template <typename Scalar>
mjData* DifferentialActionModelContactMjTpl<Scalar>::get_mjData() const {
  return mjData_;
}

template <typename Scalar>
const mjModel* DifferentialActionModelContactMjTpl<Scalar>::get_mjModel() const {
  return mjModel_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector4s&
DifferentialActionModelContactMjTpl<Scalar>::get_fids() const {
  return fids_;
}

template <typename Scalar>
void DifferentialActionModelContactMjTpl<Scalar>::set_mjData(mjData* mjData) {
  if (!mjData) {
    throw_pretty("Invalid argument: mjData is null.");
  }
  mjData_ = mjData;
}

template <typename Scalar>
void DifferentialActionModelContactMjTpl<Scalar>::set_mjModel(const mjModel* mjModel) {
  if (!mjModel) {
    throw_pretty("Invalid argument: mjModel is null.");
  }
  mjModel_ = mjModel;
}

template <typename Scalar>
void DifferentialActionModelContactMjTpl<Scalar>::set_fids(const Vector4s& fids) {
  if (fids.size() != 4) {
    throw_pretty("Invalid argument: fids must have a size of 4.");
  }
  fids_ = fids;
}

}  // namespace cito