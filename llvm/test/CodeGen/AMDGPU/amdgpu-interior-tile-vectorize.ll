; REQUIRES: asserts

; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='loop-simplify,lcssa,amdgpu-interior-tile-split' \
; RUN:   -debug-only=amdgpu-interior-tile-split -S %s 2>&1 \
; RUN:   | FileCheck %s

declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workgroup.id.y()
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)

; A batched-GEMM-style outer K loop with a nested cooperative global→LDS
; staging loop.  After the interior clone strips the M/N lane guards, the
; staging loop must be rewritten to <4 x float> traffic with trip count 1024.

; CHECK: Widened interior cooperative staging loop
; CHECK: Cloned nested-loop staging region in coop_staging_float4
; CHECK-LABEL: define amdgpu_kernel void @coop_staging_float4(
; CHECK: br i1 %interior.staging.full, label %staging.interior, label %staging
; CHECK-LABEL: stage.load.interior:
; CHECK: load <4 x float>, ptr addrspace(1)
; CHECK: store <4 x float>, ptr addrspace(3)
; CHECK: icmp ult i32 %{{.*}}, 1024
define amdgpu_kernel void @coop_staging_float4(ptr addrspace(1) %a,
                                               ptr addrspace(3) %lds,
                                               i32 %m, i32 %n, i32 %k) {
entry:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %x.base = shl i32 %workgroup.x, 7
  %y.base = shl i32 %workgroup.y, 7
  br label %k.preheader

k.preheader:
  %n.remaining = sub i32 %n, %x.base
  %n.nonnegative = call i32 @llvm.smax.i32(i32 %n.remaining, i32 0)
  %n.extent = call i32 @llvm.smin.i32(i32 %n.nonnegative, i32 128)
  %m.remaining = sub i32 %m, %y.base
  %m.nonnegative = call i32 @llvm.smax.i32(i32 %m.remaining, i32 0)
  %m.extent = call i32 @llvm.smin.i32(i32 %m.nonnegative, i32 128)
  %k.remaining = sub i32 %k, 0
  %k.extent = call i32 @llvm.smin.i32(i32 %k.remaining, i32 32)
  %a.int = ptrtoint ptr addrspace(1) %a to i64
  %a.mask = and i64 %a.int, 15
  %a.aligned = icmp eq i64 %a.mask, 0
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %staging

staging:
  %idx = phi i32 [ %tid, %k.header ], [ %idx.next, %stage.latch ]
  %lm = udiv i32 %idx, 32
  %lk = urem i32 %idx, 32
  %gm = add i32 %y.base, %lm
  %gn = add i32 %x.base, %lk
  ; Direct M/N lane guards — the shape collectDirectInteriorTileBounds proves.
  %m.in.bounds = icmp slt i32 %gm, %m
  br i1 %m.in.bounds, label %stage.n.check, label %stage.latch

stage.n.check:
  %n.in.bounds = icmp slt i32 %gn, %n
  br i1 %n.in.bounds, label %stage.load, label %stage.latch

stage.load:
  %gk = add i32 %i, %lk
  %a.ptr = getelementptr float, ptr addrspace(1) %a, i32 %gm
  %a.ptr.k = getelementptr float, ptr addrspace(1) %a.ptr, i32 %gk
  %val = load float, ptr addrspace(1) %a.ptr.k, align 4
  %lds.row = mul i32 %lm, 32
  %lds.off = add i32 %lds.row, %lk
  %lds.ptr = getelementptr float, ptr addrspace(3) %lds, i32 %lds.off
  store float %val, ptr addrspace(3) %lds.ptr, align 4
  br label %stage.latch

stage.latch:
  %idx.next = add nuw i32 %idx, 256
  %stage.more = icmp ult i32 %idx.next, 4096
  br i1 %stage.more, label %staging, label %k.barrier

k.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %compute

compute:
  br label %k.latch

k.latch:
  %next = add nuw i32 %i, 32
  %more = icmp ult i32 %next, %k
  br i1 %more, label %k.header, label %k.exit

k.exit:
  ret void
}

; Adjacent scalar stores on the interior path collapse to one <4 x float> store.
; CHECK-LABEL: define amdgpu_kernel void @adjacent_store_chain(
; CHECK: store <4 x float>
define amdgpu_kernel void @adjacent_store_chain(ptr addrspace(1) %out,
                                                i32 %m, i32 %n, i32 %k) {
entry:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
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
  %lane = and i32 %tid, 127
  %index = add i32 %x.base, %lane
  %in.bounds = icmp ult i32 %index, %n
  br i1 %in.bounds, label %stage.store, label %stage.skip

stage.store:
  %ptr0 = getelementptr float, ptr addrspace(1) %out, i32 %index
  %ptr1 = getelementptr float, ptr addrspace(1) %ptr0, i32 1
  %ptr2 = getelementptr float, ptr addrspace(1) %ptr0, i32 2
  %ptr3 = getelementptr float, ptr addrspace(1) %ptr0, i32 3
  store float 1.0, ptr addrspace(1) %ptr0, align 16
  store float 2.0, ptr addrspace(1) %ptr1, align 4
  store float 3.0, ptr addrspace(1) %ptr2, align 4
  store float 4.0, ptr addrspace(1) %ptr3, align 4
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
