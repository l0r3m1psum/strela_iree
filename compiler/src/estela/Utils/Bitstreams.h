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
inline constexpr std::array<uint32_t, 5 * 4 * 4> centered_matmul_bitstream{
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
  0x08800109, 0x003C0340, 0x00000082, 0x00000000, 0x00000000, // 7
  0x00000201, 0x020C0300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 3
};

} // namespace mlir::estela

#endif // ESTELA_UTILS_BITSTREAMS_H_
