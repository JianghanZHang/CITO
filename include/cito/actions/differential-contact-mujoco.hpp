// author: Jianghan Zhang
// This is a Dummy class for Integrated Action Model with Mujoco for Contact Dynamics
#ifndef CITO_DIFFERENTIAL_CONTACT_MUJOCO_HPP_
#define CITO_DIFFERENTIAL_CONTACT_MUJOCO_HPP_
#include <stdexcept>
#include <crocoddyl/core/fwd.hpp>
#include <crocoddyl/core/mathbase.hpp>
#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/constraints/constraint-manager.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/utils/exception.hpp>
#include <crocoddyl/multibody/data/multibody.hpp>
#include <crocoddyl/multibody/states/multibody.hpp> 

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "cito/fwd.hpp"

using namespace crocoddyl;

namespace cito{
namespace bp = boost::python;

template <typename _Scalar>
class DifferentialActionModelContactMjTpl
    : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataContactMjTpl<Scalar> Data;
  typedef DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector4s Vector4s;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::MatrixXsRowMajor MatrixXsRowMajor;
  typedef pinocchio::ModelTpl<Scalar> PinocchioModel;


DifferentialActionModelContactMjTpl(
      const bp::object mj_model_wrapper,
    //   bp::object mj_data_wrapper,
      boost::shared_ptr<StateMultibody> state,
      boost::shared_ptr<ActuationModelAbstract> actuation,
      boost::shared_ptr<CostModelSum> costs,
      boost::shared_ptr<ConstraintModelManager> constraints = nullptr);

  virtual ~DifferentialActionModelContactMjTpl();

  /**
   * @brief Compute the system acceleration, and cost value
   *
   * It computes the system acceleration using the free forward-dynamics.
   *
   * @param[in] data  Free forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calc(const
   * boost::shared_ptr<DifferentialActionDataAbstract>& data, const
   * Eigen::Ref<const VectorXs>& x)
   */
  virtual void calc(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the contact dynamics, and cost function
   *
   * @param[in] data  Free forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calcDiff(const
   * boost::shared_ptr<DifferentialActionDataAbstract>& data, const
   * Eigen::Ref<const VectorXs>& x)
   */
  virtual void calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the free forward-dynamics data
   *
   * @return free forward-dynamics data
   */
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();

  /**
   * @brief Check that the given data belongs to the free forward-dynamics data
   */
  virtual bool checkData(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data);

  /**
   * @brief @copydoc Base::quasiStatic()
   */
  virtual void quasiStatic(
      const boost::shared_ptr<DifferentialActionDataAbstract>& data,
      Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
      const std::size_t maxiter = 100, const Scalar tol = Scalar(1e-9));

  /**
   * @brief Return the number of inequality constraints
   *//* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */

  virtual std::size_t get_ng() const;

  /**
   * @brief Return the number of equality constraints
   */
  virtual std::size_t get_nh() const;

  /**
   * @brief Return the lower bound of the inequality constraints
   */
  virtual const VectorXs& get_g_lb() const;

  /**
   * @brief Return the upper bound of the inequality constraints
   */
  virtual const VectorXs& get_g_ub() const;

  /**
   * @brief Return the actuation model
   */
  const boost::shared_ptr<ActuationModelAbstract>& get_actuation() const;

  /**
   * @brief Return the cost model
   */
  const boost::shared_ptr<CostModelSum>& get_costs() const;

  /**
   * @brief Return the constraint model manager
   */
  const boost::shared_ptr<ConstraintModelManager>& get_constraints() const;

  /**
   * @brief Return the Pinocchio model
   */
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;

  /**
   * @brief Return the armature vector
   */
  const VectorXs& get_armature() const;

  /**
   * @brief Modify the armature vector
   */
  void set_armature(const VectorXs& armature);

  /**
   * @brief Print relevant information of the free forward-dynamics model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

  virtual mjData* get_mjData() const;

  virtual const mjModel* get_mjModel() const;

  virtual const Vector4s& get_fids() const;

  virtual void set_mjData(mjData* mjData);

  virtual void set_mjModel(const mjModel* mjModel);

  virtual void set_fids(const Vector4s& fids);

// This is bad, but needed to access these members in IntegratedActionModelContactMj
 public:
  pinocchio::ModelTpl<Scalar>& pinocchio_;                 //!< Pinocchio model
  boost::shared_ptr<CostModelSum> costs_;                  //!< Cost model
  boost::shared_ptr<ConstraintModelManager> constraints_;  //!< Constraint model
  const mjModel* mjModel_;  //!< Mujoco model, this need to be constant to be compatible with mujoco's APIs
  mjData* mjData_;  //!< Mujoco model

 protected:
  using Base::g_lb_;   //!< Lower bound of the inequality constraints
  using Base::g_ub_;   //!< Upper bound of the inequality constraints
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state
  
 private:
//   boost::shared_ptr<typename crocoddyl::StateMultibody::PinocchioModel> pin_model_;  //!< Pinocchio model
  boost::shared_ptr<ActuationModelAbstract> actuation_;    //!< Actuation model
  bool without_armature_;  //!< Indicate if we have defined an armature
  VectorXs armature_;      //!< Armature vector
  Vector4s fids_;          //!< Feet IDs
};

template <typename _Scalar>
struct DifferentialActionDataContactMjTpl
    : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef JointDataAbstractTpl<Scalar> JointDataAbstract;
  typedef DataCollectorJointActMultibodyTpl<Scalar>
      DataCollectorJointActMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Vector4s Vector4s;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataContactMjTpl(Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(
            &pinocchio, model->get_actuation()->createData(),
            boost::make_shared<JointDataAbstract>(
                model->get_state(), model->get_actuation(), model->get_nu())),
        costs(model->get_costs()->createData(&multibody)),
        Minv(model->get_state()->get_nv(), model->get_state()->get_nv()),
        u_drift(model->get_state()->get_nv()),
        dtau_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        tmp_xstatic(model->get_state()->get_nx()) {
    multibody.joint->dtau_du.diagonal().setOnes();
    costs->shareMemory(this);
    if (model->get_constraints() != nullptr) {
      constraints = model->get_constraints()->createData(&multibody);
      constraints->shareMemory(this);
    }
    Minv.setZero();
    u_drift.setZero();
    dtau_dx.setZero();
    tmp_xstatic.setZero();
  }

  
  pinocchio::DataTpl<Scalar> pinocchio;
  DataCollectorJointActMultibody multibody;
  boost::shared_ptr<CostDataSumTpl<Scalar> > costs;
  boost::shared_ptr<ConstraintDataManagerTpl<Scalar> > constraints;
  MatrixXs Minv;
  VectorXs u_drift;
  MatrixXs dtau_dx;
  VectorXs tmp_xstatic;
  
  using Base::h;
  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xout;
};

}

#include "cito/actions/differential-contact-mujoco.hxx"
#endif  // CITO_DIFFERENTIAL_CONTACT_MUJOCO_HPP_