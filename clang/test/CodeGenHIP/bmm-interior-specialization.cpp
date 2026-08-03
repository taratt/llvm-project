// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -emit-llvm -o - %s \
// RUN:   -fhip-new-launch-api -fhip-bmm-interior-specialization \
// RUN:   -I %S/../CodeGenCUDA/Inputs | FileCheck %s

#include "../CodeGenCUDA/Inputs/cuda.h"

__global__ void bmm_device(const float *a, const float *b, float *c, int batch,
                           int m, int n, int k, long long sa, long long sb,
                           long long sc) {}

void launch(const float *a, const float *b, float *c, int batch, int m, int n,
            int k, long long sa, long long sb, long long sc) {
  bmm_device<<<dim3(1), dim3(256)>>>(a, b, c, batch, m, n, k, sa, sb, sc);
}

// The host module registers both entrypoints. The new-launch stub loads the
// M/N/K ABI slots, performs positive power-of-two tile tests, and selects the
// interior handle only for 128x128x32-divisible launches. The `.interior`
// handle must be a defined host symbol (same stub initializer as the original)
// so the final host .so does not have an undefined reference.
// CHECK: @{{.*bmm_device.*}}.interior ={{.*}}constant ptr @{{.*}}
// CHECK: and i32 {{.*}}, 31
// CHECK: and i32 {{.*}}, 127
// CHECK: and i32 {{.*}}, 127
// CHECK: bmm.interior.kernel
// CHECK: __hipRegisterFunction
// CHECK: .interior

