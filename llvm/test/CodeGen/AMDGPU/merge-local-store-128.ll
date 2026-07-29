; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -o - %s | FileCheck %s

; Four adjacent scalar LDS stores should be merged before selection so the
; aligned path can use one ds_write_b128 rather than two ds_write2_b32 ops.
define amdgpu_kernel void @merge_local_store_4_i32(ptr addrspace(3) align 16 %out,
                                                    i32 %a, i32 %b,
                                                    i32 %c, i32 %d) {
; CHECK-LABEL: merge_local_store_4_i32:
; CHECK: ds_write_b128
  %out.1 = getelementptr inbounds i32, ptr addrspace(3) %out, i32 1
  %out.2 = getelementptr inbounds i32, ptr addrspace(3) %out, i32 2
  %out.3 = getelementptr inbounds i32, ptr addrspace(3) %out, i32 3
  store i32 %a, ptr addrspace(3) %out, align 16
  store i32 %b, ptr addrspace(3) %out.1, align 4
  store i32 %c, ptr addrspace(3) %out.2, align 8
  store i32 %d, ptr addrspace(3) %out.3, align 4
  ret void
}
