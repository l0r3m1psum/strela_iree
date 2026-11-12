#ifndef ESTELA_OPTIONS_H_
#define ESTELA_OPTIONS_H_

#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::estela {

struct EstelaOptions {
  unsigned rows = 4;
  unsigned columns = 4;
  unsigned maxColumnWords = 16383;

  // Defined in PluginRegistration.cpp, next to IREE_DEFINE_COMPILER_OPTION_FLAGS.
  void bindOptions(iree_compiler::OptionsBinder &binder);
};

} // namespace mlir::estela

#endif // ESTELA_OPTIONS_H_
