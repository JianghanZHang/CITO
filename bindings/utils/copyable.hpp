// This file is modified from: https://github.com/loco-3d/crocoddyl/blob/master/bindings/python/crocoddyl/utils/copyable.hpp
// This file contains python methods to copy the objects
// Author: Jianghan Zhang
#ifndef CITO_UTILS_COPYABLE_HPP_
#define CITO_UTILS_COPYABLE_HPP_

#include <boost/python.hpp>

namespace cito {
namespace bp = boost::python;

///
/// \brief Add the Python method copy to allow a copy of this by calling the
/// copy constructor.
///
template <class C>
struct CopyableVisitor : public bp::def_visitor<CopyableVisitor<C> > {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def("copy", &copy, bp::arg("self"), "Returns a copy of *this.");
    cl.def("__copy__", &copy, bp::arg("self"), "Returns a copy of *this.");
    cl.def("__deepcopy__", &deepcopy, bp::args("self", "memo"),
           "Returns a deep copy of *this.");
  }

 private:
  static C copy(const C& self) { return C(self); }
  static C deepcopy(const C& self, bp::dict) { return C(self); }
};

}  // namespace cito

#endif  // CITO_UTILS_COPYABLE_HPP_