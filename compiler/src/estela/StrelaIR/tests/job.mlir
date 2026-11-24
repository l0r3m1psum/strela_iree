// RUN: iree-opt --iree-plugin=estela --split-input-file --verify-diagnostics %s

// The strela.job verifier folds every column address (foldColumnAddress) and
// reports at the column the part that doesn't fit the schema's
// `offset + delta`: a constant plus at most one push constant.

func.func @two_push_constants() {
  %0 = strela.push_constant 0 : i32
  %1 = strela.push_constant 1 : i32
  %2 = arith.index_castui %0 : i32 to index
  %3 = arith.index_castui %1 : i32 to index
  %4 = arith.addi %2, %3 : index
  // expected-error @+1 {{column address s0 + s1 (s0 = push constant 0, s1 = push constant 1) is not a constant plus at most one push constant}}
  %5 = strela.inp_column binding(0) address(%4) words(1) stride(1)
  %6 = strela.out_column unused
  strela.job bitstream([0]) inputs(%5) outputs(%6)
  return
}

// -----

func.func @push_constant_product() {
  %0 = strela.push_constant 0 : i32
  %1 = arith.index_castui %0 : i32 to index
  %2 = arith.muli %1, %1 : index
  // expected-error @+1 {{column address multiplies two values known only at dispatch time}}
  %3 = strela.inp_column binding(0) address(%2) words(1) stride(1)
  %4 = strela.out_column unused
  strela.job bitstream([0]) inputs(%3) outputs(%4)
  return
}

// -----

// A scaled push constant is linear but still not an offset the runtime adds.

func.func @scaled_push_constant() {
  %c4 = arith.constant 4 : index
  %0 = strela.push_constant 0 : i32
  %1 = arith.index_castui %0 : i32 to index
  %2 = arith.muli %1, %c4 : index
  // expected-error @+1 {{column address s0 * 4 (s0 = push constant 0) is not a constant plus at most one push constant}}
  %3 = strela.inp_column binding(0) address(%2) words(1) stride(1)
  %4 = strela.out_column unused
  strela.job bitstream([0]) inputs(%3) outputs(%4)
  return
}

// -----

// A workgroup id becomes a term min(wg_d * step, limit), one each for x and
// y: a min of anything else, a second term on the same id, or the z id don't
// fit.

func.func @min_of_push_constant() {
  %0 = strela.push_constant 0 : i32
  %1 = arith.index_castui %0 : i32 to index
  %2 = affine.min affine_map<()[s0] -> (s0, 64)>()[%1]
  // expected-error @+1 {{has an affine.min that is not min(id * step, limit) of one workgroup id}}
  %3 = strela.inp_column binding(0) address(%2) words(1) stride(1)
  %4 = strela.out_column unused
  strela.job bitstream([0]) inputs(%3) outputs(%4)
  return
}

// -----

func.func @two_terms_on_x() {
  %0 = strela.workgroup_id[0] : index
  %1 = affine.min affine_map<()[s0] -> (s0 * 12, 20)>()[%0]
  %2 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 4)>()[%1, %0]
  // expected-error @+1 {{has a term in workgroup dimension 0 a column address can't hold}}
  %3 = strela.inp_column binding(0) address(%2) words(1) stride(1)
  %4 = strela.out_column unused
  strela.job bitstream([0]) inputs(%3) outputs(%4)
  return
}

// -----

func.func @z_term() {
  %0 = strela.workgroup_id[2] : index
  %1 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%0]
  // expected-error @+1 {{has a term in workgroup dimension 2 a column address can't hold}}
  %2 = strela.inp_column binding(0) address(%1) words(1) stride(1)
  %3 = strela.out_column unused
  strela.job bitstream([0]) inputs(%2) outputs(%3)
  return
}

// -----

// Inputs and outputs are columns of their own kind.

func.func @output_from_input_column() {
  %c0 = arith.constant 0 : index
  %0 = strela.inp_column binding(0) address(%c0) words(1) stride(1)
  // expected-error @+1 {{output 0 is not a strela.out_column}}
  strela.job bitstream([0]) inputs(%0) outputs(%0)
  return
}

// -----

// Inside a STRELA executable the job is checked against the target description
// (one input and one output column per fabric column) and the export's layout.

#layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable private @push_constant_out_of_layout {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 1, max_column_words = 16383>}>) {
    hal.executable.export public @f ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @f() {
        %0 = strela.push_constant 1 : i32
        %1 = arith.index_castui %0 : i32 to index
        // expected-error @+1 {{reads push constant 1, the layout has 1}}
        %2 = strela.inp_column binding(0) address(%1) words(1) stride(1)
        %3 = strela.out_column unused
        strela.job bitstream([0]) inputs(%2) outputs(%3)
        return
      }
    }
  }
}

// -----

hal.executable private @column_count {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @column_count() {
        %c0 = arith.constant 0 : index
        %0 = strela.inp_column binding(0) address(%c0) words(1) stride(1)
        %1 = strela.out_column unused
        // expected-error @+1 {{has 3 input and 4 output columns, the fabric has 4}}
        strela.job bitstream([0]) inputs(%0, %0, %0) outputs(%1, %1, %1, %1)
        return
      }
    }
  }
}
