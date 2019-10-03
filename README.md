Support Vector Machine Library Wrappers
=======================================

This project provides an alternative, modern C++ interface to
the [libsvm library][1]. The original libsvm code is redistributed largely
unmodified as part of this repository, as is permissible under
the terms of its BSD 3-clause license.

For now, not all features of the upstream libsvm are supported; in particular
regression is _not_ supported for the time being.

However, these wrappers allow for an easy access to the core features of libsvm,
two-class classification and use of the kernel trick. In particular, features
include:

  * A `svm::dataset` data structure which wraps the `struct svm_node *` of
    libsvm in a safe way. STL-style bidirectional iterators are provided,
    converting constructors from STL-style containers exist, the existence of
    sentinel elements is ensured and transparent to the user, memory is managed.
  * The four built-in kernel functions (linear, polynomial, radial basis
    functions, sigmoid) provided by libsvm are exposed as types in the
    `svm::kernel` namespace.
  * Users may define further "precomputed" kernels in libsvm. In that case,
    libsvm expects their users to provide the kernel evaluations for all pairs
    of training samples rather than the actual training data. This wrapper
    handles this instead, making the use of precomputed kernels very accessible.
    One only needs to provide a struct which defines the type definition
    `input_container_type` that is to be used for training samples as well as an
    overload of `operator()` which calculates the kernel evaluation of two
    samples (of type `input_container_type`). As an example, a "precomputed"
    version of the linear kernel is provided in `kernel/linear_precomputed.hpp`.
    This is merely for demonstration and testing; in applications, the built-in
    linear kernel should be used instead.
  * `svm::parameters` represents the parameters of the SVM optimization and also
    the parameters of the built-in kernels (due to technical reasons). Thus, it
    requires the kernel type as a template parameter. Precomputed kernels should
    inject a template specialization of `parameters<Kernel>` into the `svm`
    namespace.
  * `svm::problem` represents the training data. For built-in kernels, it stores
    the samples; for precomputed kernels it does so as well but also stores the
    kernel evaluation matrix. New samples can be added with `add_sample` (as a
    `svm::dataset` for built-in kernels, or as the `input_container_type` of
    your choice for precomputed kernels). `svm::problem` takes the kernel type
    as a template parameter to discern the different behaviors; you do _not_
    need to provide a template specialization for precomputed kernels, though.
  * `svm::model` represents the result of the SVM optimization. The actual
    optimization takes place when calling the constructor. It expects both the
    problem and the parameters objects as arguments. The problem has to be
    provided as an _rvalue reference_ (i.e. as a temporary or by invoking
    `std::move`) and will be invalidated afterwards. This is because the
    resulting `svm::model` will take hold of the sample data in the problem, at
    least for those samples which become "support vectors".
    Iterating over the model gives access to its support vectors and their
    respective coefficients.
    The `svm::model` provides an `operator()` which can be called with a test
    sample and returns a pair consisting of the label (-1 or +1) and
      - _in case of binary classification_, the value of the decision function;
      - or _in case of multiclassification_ between _M_ distinct labels, a
        container of size _M(M-1)_, holding all decision function values for
        each binary classification between any two labels.
        Through `svm::model::classifiers()`, one may alternatively obtain
        a container of _M(M-1)_ views on the `svm::model`, each behaving like a
        binary classification model (i.e. its call operator returns a pair of
        the predicted label and the decision function value).
  * _introspector_ classes are defined for use with the linear
    (`linear_introspector`) and polynomial kernels (`tensor_introspector`).
    These in particular calculate contractions of multinomials of support vector
    components over all support vectors. These are useful when interpreting the
    resultant model and gaining insight from the learnt decision function. E.g.
    the `linear_introspector` gives access to the coefficients of the hyperplane
    that SVM has learned to separate the two classes as best as possible. The
    introspector concept is very general and may be used in different ways with
    custom kernels.
  * The wrapper also define an interface for serialization (saving/loading) of
    the resulting model and problem. ASCII serialization uses the text file
    input / output provided by libsvm.
  * Alternatively, when used in conjunction with the [ALPSCore][3] library, you
    can use an HDF5 serializer instead. Storing model and problem in an HDF5
    file saves on disk space and I/O bandwidth by using a binary format and
    enabling on-the-fly gzip compression. ALPSCore is otherwise not a depenency
    of these wrappers of libsvm. When you want to use HDF5 serialization, you
    need to include the additional header `<svm/serialization/hdf5.hpp>` and
    link against ALPSCore's `libalps-hdf5`.


Included Third-party Code
-------------------------

  * **Libsvm** library (BSD license): [Github][1] and [website][2],
  * **doctest** header (MIT license): [Github][4]


License
-------

Copyright © 2018–2019  Jonas Greitemann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program, see the file entitled "LICENCE" in the
repository's root directory, or see <http://www.gnu.org/licenses/>.


  [1]: https://github.com/cjlin1/libsvm
  [2]: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
  [3]: https://github.com/ALPSCore/ALPSCore
  [4]: https://github.com/onqtam/doctest
