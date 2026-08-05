; REQUIRES: asserts

; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='loop-simplify,lcssa,amdgpu-interior-tile-split' \
; RUN:   -amdgpu-interior-tile-vectorize -debug-only=amdgpu-interior-tile-split -S %s 2>&1 \
; RUN:   | FileCheck %s

declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workgroup.id.y()
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)

; -debug-only writes to stderr, so every debug line lands ahead of the -S
; output. Check them here as one group rather than per function.
; CHECK-COUNT-5: Widened interior cooperative staging loop
; CHECK: Cloned nested-loop staging region in coop_staging_float4

; A batched-GEMM-style outer K loop with a nested cooperative global→LDS
; staging loop.  After the interior clone strips the M/N lane guards, the
; staging loop must be rewritten to <4 x float> traffic with trip count 1024.

; CHECK-LABEL: define amdgpu_kernel void @coop_staging_float4(
; SplitBlock may rename the staging entry (e.g. staging1.interior).
; CHECK: br i1 %interior.staging.full, label %{{[^,]*}}staging{{[^,]*}}.interior, label %{{[^,]*}}staging{{[^,]*}}
; The row advances 4x (idx/8 instead of idx/32) and each thread's single column
; is replaced by four contiguous ones via an address correction.
; CHECK: udiv i32 %{{.*}}, 8
; CHECK-LABEL: stage.load.interior:
; CHECK: urem i32 %{{.*}}, 8
; CHECK: interior.col4.delta = sub i32
; CHECK: load <4 x float>, ptr addrspace(1)
; CHECK: store <4 x float> {{.*}}, ptr addrspace(3)
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

; Real HIP BMM after InstCombine: step 256 is a multiple of 32, so
; lk = idx%32 sinks to the invariant tid%32; only lm = idx>>5 uses the IV.
; The invariant column is already folded into the staging bases, so re-tiling
; has to correct the addresses rather than rewrite the (out-of-loop) rem.
; CHECK-LABEL: define amdgpu_kernel void @coop_staging_float4_bmm_idx(
; CHECK: udiv i32 %{{.*}}, 8
; CHECK: interior.col4.delta = sub i32 %{{.*}}, %lk
; CHECK: load <4 x float>, ptr addrspace(1)
; CHECK: store <4 x float> {{.*}}, ptr addrspace(3)
; CHECK: icmp ult i32 %{{.*}}, 1024
define amdgpu_kernel void @coop_staging_float4_bmm_idx(ptr addrspace(1) %a,
                                                       ptr addrspace(3) %lds,
                                                       i32 %m, i32 %n, i32 %k) {
entry:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %x.base = shl i32 %workgroup.x, 7
  %y.base = shl i32 %workgroup.y, 7
  ; Invariant rem sunk out of the staging IV.
  %lk = and i32 %tid, 31
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
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %staging

staging:
  %idx = phi i32 [ %tid, %k.header ], [ %idx.next, %stage.latch ]
  %lm = lshr i32 %idx, 5
  %gm = add i32 %y.base, %lm
  %gn = add i32 %x.base, %lk
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
  br label %stage.latch

stage.latch:
  ; Real HIP diamond: load does not dominate store; phi merges zero path.
  %merged = phi float [ %val, %stage.load ], [ 0.000000e+00, %stage.n.check ],
                      [ 0.000000e+00, %staging ]
  %lds.row = mul i32 %lm, 32
  %lds.off = add i32 %lds.row, %lk
  %lds.ptr = getelementptr float, ptr addrspace(3) %lds, i32 %lds.off
  store float %merged, ptr addrspace(3) %lds.ptr, align 4
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

; The latch form real HIP emits: `idx < N-Step` (3840) instead of
; `idx.next < N` (4096). The float4 bound has to stay in that form (768); using
; the element limit directly buys an extra trip that stages a row past the tile.
; CHECK-LABEL: define amdgpu_kernel void @coop_staging_float4_iv_latch(
; CHECK: udiv i32 %{{.*}}, 8
; CHECK: load <4 x float>, ptr addrspace(1)
; CHECK: store <4 x float> {{.*}}, ptr addrspace(3)
; CHECK: icmp ult i32 %{{.*}}, 768
define amdgpu_kernel void @coop_staging_float4_iv_latch(ptr addrspace(1) %a,
                                                        ptr addrspace(3) %lds,
                                                        i32 %m, i32 %n, i32 %k) {
entry:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %x.base = shl i32 %workgroup.x, 7
  %y.base = shl i32 %workgroup.y, 7
  %lk = and i32 %tid, 31
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
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %staging

staging:
  %idx = phi i32 [ %tid, %k.header ], [ %idx.next, %stage.latch ]
  %lm = lshr i32 %idx, 5
  %gm = add i32 %y.base, %lm
  %gn = add i32 %x.base, %lk
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
  br label %stage.latch

stage.latch:
  %merged = phi float [ %val, %stage.load ], [ 0.000000e+00, %stage.n.check ],
                      [ 0.000000e+00, %staging ]
  %lds.row = mul i32 %lm, 32
  %lds.off = add i32 %lds.row, %lk
  %lds.ptr = getelementptr float, ptr addrspace(3) %lds, i32 %lds.off
  store float %merged, ptr addrspace(3) %lds.ptr, align 4
  %idx.next = add nuw i32 %idx, 256
  ; Compare the current IV against N-Step, the form clang actually emits.
  %stage.more = icmp ult i32 %idx, 3840
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

; Tile-count outer K loop: `for (t = 0; t < ceil(K/8); ++t)` with k0 = t*8
; (gemm_small_k / tall_skinny). Must still clone + widen.
; CHECK-LABEL: define amdgpu_kernel void @coop_staging_tile_count_k(
; CHECK: load <4 x float>, ptr addrspace(1)
; CHECK: store <4 x float> {{.*}}, ptr addrspace(3)
define amdgpu_kernel void @coop_staging_tile_count_k(ptr addrspace(1) %a,
                                                     ptr addrspace(3) %lds,
                                                     i32 %m, i32 %n, i32 %k) {
entry:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  ; Swapped grid vs BMM: M on blockIdx.x, N on blockIdx.y (gemm_small_k).
  %m.base = shl i32 %workgroup.x, 7
  %n.base = shl i32 %workgroup.y, 7
  %lk = and i32 %tid, 7
  %k.tile.max = lshr i32 %k, 3
  br label %k.header

k.header:
  %t = phi i32 [ 0, %entry ], [ %t.next, %k.latch ]
  %k0 = mul i32 %t, 8
  br label %staging

staging:
  %idx = phi i32 [ %tid, %k.header ], [ %idx.next, %stage.latch ]
  %lm = lshr i32 %idx, 3
  %gm = add i32 %m.base, %lm
  %gn = add i32 %n.base, %lk
  %m.in.bounds = icmp slt i32 %gm, %m
  br i1 %m.in.bounds, label %stage.n.check, label %stage.latch

stage.n.check:
  %n.in.bounds = icmp slt i32 %gn, %n
  br i1 %n.in.bounds, label %stage.load, label %stage.latch

stage.load:
  %gk = add i32 %k0, %lk
  %a.ptr = getelementptr float, ptr addrspace(1) %a, i32 %gm
  %a.ptr.k = getelementptr float, ptr addrspace(1) %a.ptr, i32 %gk
  %val = load float, ptr addrspace(1) %a.ptr.k, align 4
  br label %stage.latch

stage.latch:
  %merged = phi float [ %val, %stage.load ], [ 0.000000e+00, %stage.n.check ],
                      [ 0.000000e+00, %staging ]
  %lds.row = mul i32 %lm, 8
  %lds.off = add i32 %lds.row, %lk
  %lds.ptr = getelementptr float, ptr addrspace(3) %lds, i32 %lds.off
  store float %merged, ptr addrspace(3) %lds.ptr, align 4
  %idx.next = add nuw i32 %idx, 256
  %stage.more = icmp ult i32 %idx.next, 1024
  br i1 %stage.more, label %staging, label %k.barrier

k.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %k.latch

k.latch:
  %t.next = add nuw i32 %t, 1
  %more = icmp ult i32 %t.next, %k.tile.max
  br i1 %more, label %k.header, label %k.exit

k.exit:
  ret void
}

; Already-interior kernel: no M/N lane guards. Widen must run without a clone.
; CHECK-LABEL: define amdgpu_kernel void @already_interior_staging(
; CHECK: load <4 x float>, ptr addrspace(1)
; CHECK: store <4 x float> {{.*}}, ptr addrspace(3)
define amdgpu_kernel void @already_interior_staging(ptr addrspace(1) %a,
                                                    ptr addrspace(3) %lds,
                                                    i32 %k) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %lk = and i32 %tid, 31
  br label %k.header

k.header:
  %i = phi i32 [ 0, %entry ], [ %next, %k.latch ]
  br label %staging

staging:
  %idx = phi i32 [ %tid, %k.header ], [ %idx.next, %stage.latch ]
  %lm = lshr i32 %idx, 5
  %gk = add i32 %i, %lk
  %a.ptr = getelementptr float, ptr addrspace(1) %a, i32 %lm
  %a.ptr.k = getelementptr float, ptr addrspace(1) %a.ptr, i32 %gk
  %val = load float, ptr addrspace(1) %a.ptr.k, align 4
  %lds.row = mul i32 %lm, 32
  %lds.off = add i32 %lds.row, %lk
  %lds.ptr = getelementptr float, ptr addrspace(3) %lds, i32 %lds.off
  store float %val, ptr addrspace(3) %lds.ptr, align 4
  %idx.next = add nuw i32 %idx, 256
  %stage.more = icmp ult i32 %idx.next, 4096
  br i1 %stage.more, label %staging, label %k.barrier

k.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %k.latch

k.latch:
  %next = add nuw i32 %i, 32
  %more = icmp ult i32 %next, %k
  br i1 %more, label %k.header, label %k.exit

k.exit:
  ret void
}
