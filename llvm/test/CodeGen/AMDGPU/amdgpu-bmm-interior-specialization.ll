; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-bmm-interior-specialization -S %s | FileCheck %s

declare i32 @llvm.amdgcn.workgroup.id.x()

; The pass is deliberately limited to this BMM kernel ABI. Accept both the
; demangled POC name and HIP Itanium mangling; clone as `<name>.interior`.
define amdgpu_kernel void @_Z10bmm_devicePKfS0_Pfiiiilll(
    ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, i32 %batch,
    i32 %m, i32 %n, i32 %k, i64 %sa, i64 %sb, i64 %sc) {
entry:
  %wg = call i32 @llvm.amdgcn.workgroup.id.x()
  %row = shl i32 %wg, 7
  %m.ok = icmp slt i32 %row, %m
  br i1 %m.ok, label %m.in, label %exit

m.in:
  %n.ok = icmp slt i32 %row, %n
  br i1 %n.ok, label %n.in, label %exit

n.in:
  %k.done = icmp sge i32 %k, 32
  br i1 %k.done, label %work, label %exit

work:
  store i32 1, ptr addrspace(1) %c, align 4
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @_Z10bmm_devicePKfS0_Pfiiiilll(
; CHECK: br i1 %m.ok, label %m.in, label %exit
; CHECK-LABEL: define amdgpu_kernel void @_Z10bmm_devicePKfS0_Pfiiiilll.interior(
; CHECK: br label %m.in
; CHECK-LABEL: m.in:
; CHECK: br label %n.in
; CHECK-LABEL: n.in:
; CHECK: br i1 %k.done, label %work, label %exit

