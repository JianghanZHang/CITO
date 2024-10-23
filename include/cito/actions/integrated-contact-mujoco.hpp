/*
Author: Jianghan Zhang
This is a class for Integrated Action Model with Mujoco for Contact Dynamics
*/
#ifndef CITO_INTEGRATED_CONTACT_MUJOCO_HPP_
#define CITO_INTEGRATED_CONTACT_MUJOCO_HPP_

#include <crocoddyl/core/fwd.hpp>
#include <crocoddyl/core/mathbase.hpp>

#include <crocoddyl/core/integ-action-base.hpp>
#include "cito/fwd.hpp"
#include "cito/utils/pin-mj-conversion.hpp"


using namespace crocoddyl;
namespace cito {

template <typename _Scalar>
class IntegratedActionModelContactMjTpl
    : public crocoddyl::IntegratedActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataContactMjTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar>
      DifferentialActionModelAbstract;

  typedef ControlParametrizationModelAbstractTpl<Scalar>
      ControlParametrizationModelAbstract;
  typedef ControlParametrizationDataAbstractTpl<Scalar>
      ControlParametrizationDataAbstract;

  // This is to acess DifferentialActionModelContactMj and DifferentialActionDataContactMj
  // This cannot be achieved with DifferentialActionModelAbstract and DifferentialActionDataAbstract
  // because the base classes do not have the methods and properties we needed.
  typedef DifferentialActionDataContactMjTpl<Scalar> DifferentialActionDataContactMj;
  typedef DifferentialActionModelContactMjTpl<Scalar> DifferentialActionModelContactMj;


  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Vector4s Vector4s;
  typedef typename MathBase::MatrixXsRowMajor MatrixXsRowMajor;


  /**
   * @brief Initialize the symplectic Euler integrator
   *
   * This initialization uses `ControlParametrizationPolyZeroTpl` for the
   * control parametrization.
   *
   * @param[in] model               Differential action model
   * @param[in] time_step           Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  IntegratedActionModelContactMjTpl(
      boost::shared_ptr<DifferentialActionModelAbstract> model,
      const Scalar time_step = Scalar(1e-3),
      const bool with_cost_residual = true);
  virtual ~IntegratedActionModelContactMjTpl();

  /**
   * @brief Integrate the differential action model using symplectic Euler
   * scheme
   *
   * @param[in] data  Symplectic Euler data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Integrate the total cost value for nodes that depends only on the
   * state using symplectic Euler scheme
   *
   * It computes the total cost and defines the next state as the current one.
   * This function is used in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Symplectic Euler data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the partial derivatives of the symplectic Euler integrator
   *
   * @param[in] data  Symplectic Euler data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the partial derivatives of the cost
   *
   * It updates the derivatives of the cost function with respect to the state
   * only. This function is used in the terminal nodes of an optimal control
   * problem.
   *
   * @param[in] data  Symplectic Euler data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the symplectic Euler data
   *
   * @return the symplectic Euler data
   */
  virtual boost::shared_ptr<ActionDataAbstract> createData();

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract>& data);

  /**
   * @brief Print relevant information of the Euler integrator model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

  /**
   * @brief Get the contact
   *
   * @return Shared pointer to the contact
   */
    // Getter for contact (raw pointer version)
    virtual const mjContact* get_contact() const;

    // Setter for contact (raw pointer version)
    virtual void set_contact(mjContact* contact);

    // Getter and setter for A_
    virtual const MatrixXsRowMajor& get_A() const;
    virtual void set_A(const MatrixXsRowMajor& A);

    // Getter and setter for B_
    virtual const MatrixXsRowMajor& get_B() const;
    virtual void set_B(const MatrixXsRowMajor& B);

    // Getter and setter for A_
    virtual const MatrixXs& get_stateMappingDerivatives() const;
    virtual void set_stateMappingDerivatives(const MatrixXs& M);

    // Getter and setter for B_
    virtual const MatrixXs& get_stateMappingDerivativesNext() const;
    virtual void set_stateMappingDerivativesNext(const MatrixXs& M);

     // Getter for forces_
    virtual const MatrixXsRowMajor& get_forces() const;

    // Setter for forces_
    virtual void set_forces(const MatrixXsRowMajor& forces);





 protected:
  using Base::control_;       //!< Control parametrization
  using Base::differential_;  //!< Differential action model
//   boost::shared_ptr<DifferentialActionModelContactMj> differential_;  //!< This will hide the base class's differential_
  using Base::ng_;            //!< Number of inequality constraints
  using Base::nh_;            //!< Number of equality constraints
  using Base::nu_;            //!< Dimension of the control
  using Base::state_;         //!< Model of the state
  using Base::time_step2_;    //!< Square of the time step used for integration
  using Base::time_step_;     //!< Time step used for integration
  using Base::with_cost_residual_;  //!< Flag indicating whether a cost residual is used
  mjContact* contact_;  //!< Raw pointer to MuJoCo contact data
  
 private:
  MatrixXsRowMajor forces_;  //!< Contact Jacobians
  const mjModel* mjModel_;  //!< Mujoco model, this need to be constant to be compatible with mujoco's APIs
  mjData* mjData_;  //!< Mujoco model
  Vector4s fids_;          //!< Feet IDs
  VectorXs mj_state_; // temporary variable for mujoco state
  VectorXs mj_state_next_; // temporary variable for the next mujoco state
  MatrixXs stateMappingDerivatives_; // temporary variable for state mapping derivatives
  MatrixXs stateMappingDerivativesNext_; // temporary variable for the next state mapping derivatives
  MatrixXsRowMajor A_; // temporary variable for Fx in mujoco
  MatrixXsRowMajor B_; // temporary variable for Fu in mujoco
};

template <typename _Scalar>
struct IntegratedActionDataContactMjTpl
    : public crocoddyl::IntegratedActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionDataAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef ControlParametrizationDataAbstractTpl<Scalar>
      ControlParametrizationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Vector4s Vector4s;

  template <template <typename Scalar> class Model>
  explicit IntegratedActionDataContactMjTpl(Model<Scalar>* const model)
      : Base(model) {
    differential = model->get_differential()->createData();
    control = model->get_control()->createData();
    // const std::size_t ndx = model->get_state()->get_ndx();
    // const std::size_t nv = model->get_state()->get_nv();
    // DifferentialActionModelContactMj* differential_contact_mj = 
    // dynamic_cast<DifferentialActionModelContactMj*>(model->get());
    // if (!differential_contact_mj) {
    //     throw std::runtime_error("The differential model must be of type DifferentialActionModelContactMj.");
    //   }
    }
  virtual ~IntegratedActionDataContactMjTpl() {}

  boost::shared_ptr<DifferentialActionDataAbstract>
      differential;  //!< Differential model data
  boost::shared_ptr<ControlParametrizationDataAbstract>
      control;  //!< Control parametrization data
  boost::shared_ptr<mjContact> contact;  //!< Shared pointer to a contact
  

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;

};

}  // namespace cito

#include "cito/actions/integrated-contact-mujoco.hxx"
#endif  // CITO_INTEGRATED_CONTACT_MUJOCO_HPP_