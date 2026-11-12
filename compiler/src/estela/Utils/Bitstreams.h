// Hand-written STRELA CGRA configurations.
//
// These are the words that get loaded into the fabric. They live here rather
// than next to a single user because two unrelated places need them: the
// Conv2D fusion path materializes one as a constant tensor, and the target
// backend embeds one in the serialized executable. When the executable format
// comes back this belongs in Serialization/ instead.

#ifndef ESTELA_UTILS_BITSTREAMS_H_
#define ESTELA_UTILS_BITSTREAMS_H_

#include <array>
#include <cstdint>

namespace mlir::estela {

inline constexpr size_t strelaKernelRows = 4;
inline constexpr size_t strelaKernelColumns = 4;
inline constexpr size_t strelaConfigWordsPerPe = 5;
inline constexpr size_t strelaBitstreamWords =
    strelaKernelRows * strelaKernelColumns * strelaConfigWordsPerPe;

/*
 * +--+--+--+--+
 * | 0| 1| 2| 3|
 * +--+--+--+--+
 * | 4| 5| 6| 7|
 * +--+--+--+--+
 * | 8| 9|10|11|
 * +--+--+--+--+
 * |12|13|14|15|
 * +--+--+--+--+
 */
// Constants are -1=0xFFFFFFFF and delays are set to 0xDDEE
// Perform matrix vector multiplication. In the first column it takes as input
// one input vector x three rows w_{:,i+0} w_{:,i+1} w_{:,i+2} in the other
// three STRELA columns to output their scalar products y_{i+0} y_{i+1} y_{i+2}.
// PEs 5 and 6 fork x east to the next multiplier; PE 7 must not, since east of
// the last column is tied not ready (cgra.sv) and the fork would never fire.
inline constexpr std::array<uint32_t, strelaBitstreamWords> centered_matmul_bitstream{
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 12
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 8
  0x00000041, 0x02000000, 0x00000000, 0x00000000, 0x00000000, // 4
  0x00000201, 0x020C0300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 0

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 13
  0x00000201, 0xC0040400, 0xDDEE0080, 0x00000000, 0x00000000, // 9
  0x08800109, 0x003C0340, 0x00000082, 0x00000000, 0x00000000, // 5
  0x00000201, 0x020C0300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 1

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 14
  0x00000201, 0xC0040400, 0xDDEE0080, 0x00000000, 0x00000000, // 10
  0x08800109, 0x003C0340, 0x00000082, 0x00000000, 0x00000000, // 6
  0x00000201, 0x020C0300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 2

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 15
  0x00000201, 0xC0040400, 0xDDEE0080, 0x00000000, 0x00000000, // 11
  0x08000109, 0x003C0300, 0x00000082, 0x00000000, 0x00000000, // 7
  0x00000201, 0x020C0300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 3
};

// Copies every input column to the output column with the same index. The same
// words as bypass_kernel_bitstream in strela_pynq/include/strela.h.
inline constexpr std::array<uint32_t, strelaBitstreamWords> bypass_bitstream{
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 12
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 8
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 4
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 0

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 13
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 9
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 5
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 1

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 14
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 10
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 6
  0x00000021, 0x00000000, 0x00000012, 0x00000000, 0x00000000, // 2

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 15
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 11
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 7
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 3
};

// The addition of column 0 and 1 input is written in out of column 1
// The addition of column 2 and 3 input is written in out of column 3
inline constexpr std::array<uint32_t, strelaBitstreamWords> add_bitstream{
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 12
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 8
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 4
  0x00000041, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 0

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 13
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 9
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 5
  0x08000109, 0x003C0300, 0x00000080, 0x00000000, 0x00000000, // 1

  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 14
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 10
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 6
  0x00000041, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 2

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 15
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 11
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 7
  0x08000109, 0x003C0300, 0x00000080, 0x00000000, 0x00000000, // 3
};

// max(x, 0) on column 0. PE 0 sends x - 0 > 0 south and x east; PEs 1 and 5
// bring x round to PE 4, whose mux picks x or the constant 0 on that flag; the
// result flows down column 0. The same words as relu_kernel_bitstream in
// strela_pynq/include/strela.h, measured correct on the board.
inline constexpr std::array<uint32_t, strelaBitstreamWords> relu_bitstream{
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 12
  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 8
  0x00004083, 0x20CC0300, 0x000000A0, 0x00000000, 0x00000000, // 4
  0x00000241, 0x020C0300, 0x00000099, 0x00000000, 0x00000000, // 0

  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 13
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 9
  0x00000011, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 5
  0x00400008, 0x00000200, 0x00000000, 0x00000000, 0x00000000, // 1

  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 14
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 10
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 6
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 2

  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 15
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 11
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 7
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 3
};

// A STRELA operation as the CGRA runs it: its configuration and which column
// each operand enters and the result leaves by.
struct StrelaKernel {
  const std::array<uint32_t, strelaBitstreamWords> *bitstream;
  unsigned inputCount;
  // Indexed by operand.
  std::array<unsigned, 4> inputColumns;
  unsigned outputColumn;
};

// Only the adder on columns 0 and 1 is used.
inline constexpr StrelaKernel add_kernel{&add_bitstream, 2, {0, 1}, 1};
inline constexpr StrelaKernel relu_kernel{&relu_bitstream, 1, {0}, 0};

// Output features one matvec job computes, on output columns 1-3.
inline constexpr int64_t strelaMatvecFeaturesPerJob = 3;

inline constexpr std::array<uint32_t, strelaBitstreamWords> matvecBitstream(uint16_t inChan) {
  auto k = centered_matmul_bitstream;
  k[19] = k[39] = k[59] = k[79] = 0;                         // PE 0-3 constants (zero points)
  k[27] = k[47] = k[67] = (uint32_t(inChan) << 16) | 0x0080; // PE 9-11 accumulation length
  return k;
}

} // namespace mlir::estela

#endif // ESTELA_UTILS_BITSTREAMS_H_
