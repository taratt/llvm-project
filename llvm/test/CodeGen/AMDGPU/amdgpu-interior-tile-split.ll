; REQUIRES: asserts
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-interior-tile-split \
; RUN:   -debug-only=amdgpu-interior-tile-split -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-interior-tile-split \
; RUN:   -S %s | FileCheck %s --check-prefix=CFG

; The broad candidate diagnostic is retained for non-canonical boundary
; checks.  The second kernel has a closed staging region, which is cloned
; behind the uniform full-tile dispatch while its barrier remains shared.

declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @candidate(ptr addrspace(1) %out, i32 %bound) {
entry:
  %workgroup = call i32 @llvm.amdgcn.workgroup.id.x()
  %workitem = call i32 @llvm.amdgcn.workitem.id.x()
  %tile-base = mul i32 %workgroup, 64
  %index = add i32 %tile-base, %workitem
  %in-bounds = icmp ult i32 %index, %bound
  br i1 %in-bounds, label %store, label %exit

store:
  %ptr = getelementptr float, ptr addrspace(1) %out, i32 %index
  store float 0.000000e+00, ptr addrspace(1) %ptr, align 4
  br label %exit

exit:
  ret void
}

declare i32 @llvm.amdgcn.workgroup.id.y()
declare void @llvm.amdgcn.s.barrier()
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)

define amdgpu_kernel void @canonical_staging(ptr addrspace(1) %out, i32 %m,
                                             i32 %n, i32 %k) {
entry:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %workitem = call i32 @llvm.amdgcn.workitem.id.x()
  %x.base = shl i32 %workgroup.x, 7
  %y.base = shl i32 %workgroup.y, 7
  %x.last = add i32 %x.base, 127
  %y.last = add i32 %y.base, 127
  %n.remaining = sub i32 %n, %x.base
  %n.nonnegative = call i32 @llvm.smax.i32(i32 %n.remaining, i32 0)
  %n.extent = call i32 @llvm.smin.i32(i32 %n.nonnegative, i32 128)
  %m.remaining = sub i32 %m, %y.base
  %m.nonnegative = call i32 @llvm.smax.i32(i32 %m.remaining, i32 0)
  %m.extent = call i32 @llvm.smin.i32(i32 %m.nonnegative, i32 128)
  %k.remaining = sub i32 %k, 0
  %k.extent = call i32 @llvm.smin.i32(i32 %k.remaining, i32 32)
  %full.n = icmp ult i32 %x.last, %n
  %full.m = icmp ult i32 %y.last, %m
  %full.k = icmp uge i32 %k, 32
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  %full.mn = and i1 %full.m, %full.n
  %full.mnk = and i1 %full.mn, %full.k
  %full = and i1 %full.mnk, %out.aligned
  br i1 %full, label %staging, label %edge

staging:
  %index = add i32 %x.base, %workitem
  %in.bounds = icmp ult i32 %index, %n
  br i1 %in.bounds, label %stage.store, label %stage.skip

stage.store:
  %ptr = getelementptr float, ptr addrspace(1) %out, i32 %index
  store float 0.000000e+00, ptr addrspace(1) %ptr, align 4
  br label %barrier

stage.skip:
  br label %barrier

barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %exit

edge:
  br label %exit

exit:
  ret void
}

; CHECK: Cloned canonical interior-tile staging region in canonical_staging at entry; removed 1 proven lane bounds branch(es); fast staging entry staging.interior, fallback staging, shared barrier barrier
; CHECK: Potential interior-tile boundary check in candidate:

; CFG-LABEL: define amdgpu_kernel void @canonical_staging(
; CFG: br i1 %full, label %staging.interior, label %edge
; CFG-LABEL: staging:
; CFG: br i1 %in.bounds, label %stage.store, label %stage.skip
; CFG-LABEL: barrier:
; CFG-COUNT-1: call void @llvm.amdgcn.s.barrier()
; CFG-LABEL: staging.interior:
; CFG: br label %stage.store.interior
; CFG-LABEL: stage.store.interior:
; CFG: br label %barrier
