#ifndef PYBOOST11_CONVERTER_HPP
#define PYBOOST11_CONVERTER_HPP

#include <boost/python.hpp>
#include <pybind11/pybind11.h>

namespace pyboost11
{
    template <typename T> struct converter
    {
        public:
            converter() { init(); }

            void init()
            {
                static bool initialized = false;
                if (!initialized)
                {
                    namespace bpy = boost::python;
                    // From-Python conversion.
                    bpy::converter::registry::push_back
                    (
                        &convertible
                      , &construct
                      , bpy::type_id<T>()
                    );
                    // To-Python conversion.
                    bpy::to_python_converter<T, converter>();

                    initialized = true;
                }
            }

            static void * convertible(PyObject * objptr)
            {
                namespace pyb = pybind11;
                try
                {
                    pyb::handle(objptr).cast<T>();
                    return objptr;
                }
                catch (pyb::cast_error const &)
                {
                    return nullptr;
                }
            }

            static void construct
            (
                PyObject * objptr
              , boost::python::converter::rvalue_from_python_stage1_data * data
            )
            {
                namespace pyb = pybind11;
                void * storage = reinterpret_cast
                <
                    boost::python::converter::rvalue_from_python_storage<T> *
                >(data)->storage.bytes;
                new (storage) T(pyb::handle(objptr).cast<T>());
                data->convertible = storage;
            }

            static PyObject * convert(T const & t)
            {
                return pybind11::cast(t).inc_ref().ptr();
            }
    };
}

#endif // PYBOOST11_CONVERTER_HPP
