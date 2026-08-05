; REQUIRES: asserts

; Check the transformed IR and the pass remarks from separate streams. Reading
; both from one pipe forces the checks into whatever order stdout and stderr
; happen to interleave in, which has nothing to do with what is being tested.
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='loop-simplify,lcssa,amdgpu-interior-tile-split' \
; RUN:   -amdgpu-interior-tile-vectorize -S %s 2>/dev/null \
; RUN:   | FileCheck %s

; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='loop-simplify,lcssa,amdgpu-interior-tile-split' \
; RUN:   -amdgpu-interior-tile-vectorize -debug-only=amdgpu-interior-tile-split \
; RUN:   -disable-output %s 2>&1 | FileCheck %s --check-prefix=DEBUG

declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workgroup.id.y()
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)

; Which path each kernel took. The <4 x float> traffic itself is checked per
; function below.
; DEBUG-DAG: Cloned nested-loop staging region in coop_staging_float4
; DEBUG-DAG: Cloned nested-loop staging region in coop_staging_tile_count_k
; DEBUG-DAG: Cloned nested-loop staging region in coop_staging_and_mk_guard
; DEBUG-DAG: Widened already-interior staging in already_interior_staging
; DEBUG-DAG: Widen skipped (staging access is predicated)

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

; gemm_small_k spelling: one branch on `and (gm < M), (gk < K)`, with M on
; workgroup.x. The M conjunct must still seed the CTA-uniform dispatch; the K
; conjunct stays in the cloned path.
; CHECK-LABEL: define amdgpu_kernel void @coop_staging_and_mk_guard(
; CHECK: br i1 %interior.staging.full
; CHECK: load <4 x float>, ptr addrspace(1)
; CHECK: store <4 x float> {{.*}}, ptr addrspace(3)
define amdgpu_kernel void @coop_staging_and_mk_guard(ptr addrspace(1) %a,
                                                     ptr addrspace(1) %b,
                                                     ptr addrspace(3) %lds.a,
                                                     ptr addrspace(3) %lds.b,
                                                     i32 %m, i32 %n, i32 %k) {
entry:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %m.base = mul i32 %workgroup.x, 128
  %n.base = mul i32 %workgroup.y, 128
  %k.tile.max = add i32 %k, 7
  %k.tile.max.s = lshr i32 %k.tile.max, 3
  br label %k.header

k.header:
  %t = phi i32 [ 0, %entry ], [ %t.next, %k.latch ]
  %k0 = mul i32 %t, 8
  br label %stage.a

stage.a:
  %idx.a = phi i32 [ %tid, %k.header ], [ %idx.a.next, %stage.a.latch ]
  %lm = udiv i32 %idx.a, 8
  %lk.a = urem i32 %idx.a, 8
  %gm = add i32 %m.base, %lm
  %gk.a = add i32 %k0, %lk.a
  %m.ok = icmp slt i32 %gm, %m
  %k.ok.a = icmp slt i32 %gk.a, %k
  %a.in.bounds = and i1 %m.ok, %k.ok.a
  br i1 %a.in.bounds, label %stage.a.load, label %stage.a.merge

stage.a.load:
  %a.ptr = getelementptr float, ptr addrspace(1) %a, i32 %gm
  %a.ptr.k = getelementptr float, ptr addrspace(1) %a.ptr, i32 %gk.a
  %a.val = load float, ptr addrspace(1) %a.ptr.k, align 4
  br label %stage.a.merge

stage.a.merge:
  %a.staged = phi float [ %a.val, %stage.a.load ], [ 0.000000e+00, %stage.a ]
  %lds.a.row = mul i32 %lm, 8
  %lds.a.off = add i32 %lds.a.row, %lk.a
  %lds.a.ptr = getelementptr float, ptr addrspace(3) %lds.a, i32 %lds.a.off
  store float %a.staged, ptr addrspace(3) %lds.a.ptr, align 4
  br label %stage.a.latch

stage.a.latch:
  %idx.a.next = add nuw i32 %idx.a, 256
  %a.more = icmp ult i32 %idx.a.next, 1024
  br i1 %a.more, label %stage.a, label %stage.b.pre

stage.b.pre:
  br label %stage.b

stage.b:
  %idx.b = phi i32 [ %tid, %stage.b.pre ], [ %idx.b.next, %stage.b.latch ]
  %ln = udiv i32 %idx.b, 8
  %lk.b = urem i32 %idx.b, 8
  %gn = add i32 %n.base, %ln
  %gk.b = add i32 %k0, %lk.b
  %n.ok = icmp slt i32 %gn, %n
  %k.ok.b = icmp slt i32 %gk.b, %k
  %b.in.bounds = and i1 %n.ok, %k.ok.b
  br i1 %b.in.bounds, label %stage.b.load, label %stage.b.merge

stage.b.load:
  %b.ptr = getelementptr float, ptr addrspace(1) %b, i32 %gk.b
  %b.ptr.n = getelementptr float, ptr addrspace(1) %b.ptr, i32 %gn
  %b.val = load float, ptr addrspace(1) %b.ptr.n, align 4
  br label %stage.b.merge

stage.b.merge:
  %b.staged = phi float [ %b.val, %stage.b.load ], [ 0.000000e+00, %stage.b ]
  %lds.b.row = mul i32 %ln, 8
  %lds.b.off = add i32 %lds.b.row, %lk.b
  %lds.b.ptr = getelementptr float, ptr addrspace(3) %lds.b, i32 %lds.b.off
  store float %b.staged, ptr addrspace(3) %lds.b.ptr, align 4
  br label %stage.b.latch

stage.b.latch:
  %idx.b.next = add nuw i32 %idx.b, 256
  %b.more = icmp ult i32 %idx.b.next, 1024
  br i1 %b.more, label %stage.b, label %k.barrier

k.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %k.latch

k.latch:
  %t.next = add nuw i32 %t, 1
  %more = icmp ult i32 %t.next, %k.tile.max.s
  br i1 %more, label %k.header, label %k.exit

k.exit:
  ret void
}

; Not a GEMM: mixed-address-space float traffic and stored values defined
; between the stores. Merging adjacent accesses across a whole arbitrary
; function used to assert in the compiler here, once on the differing pointer
; index widths and once on an insertelement chain built before its operands.
; CHECK-LABEL: define amdgpu_kernel void @mixed_addrspace_no_crash(
; CHECK-NOT: store <4 x float>
define amdgpu_kernel void @mixed_addrspace_no_crash(ptr addrspace(1) %g,
                                                    ptr addrspace(3) %l,
                                                    float %x) {
entry:
  %g0 = getelementptr float, ptr addrspace(1) %g, i32 0
  %g1 = getelementptr float, ptr addrspace(1) %g, i32 1
  %l0 = getelementptr float, ptr addrspace(3) %l, i32 0
  %l1 = getelementptr float, ptr addrspace(3) %l, i32 1
  %a = load float, ptr addrspace(1) %g0, align 4
  %b = load float, ptr addrspace(3) %l0, align 4
  %c = load float, ptr addrspace(1) %g1, align 4
  %d = load float, ptr addrspace(3) %l1, align 4
  %s0 = fadd float %a, %x
  store float %s0, ptr addrspace(1) %g0, align 4
  %s1 = fadd float %b, %s0
  store float %s1, ptr addrspace(1) %g1, align 4
  %s2 = fadd float %c, %s1
  store float %s2, ptr addrspace(3) %l0, align 4
  %s3 = fadd float %d, %s2
  store float %s3, ptr addrspace(3) %l1, align 4
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
  %idx = phi i32 [ %tid, %k.header ], [ %idx.next, %staging ]
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

; A staged load predicated on a bound this pass does not model. Widening would
; speculate three neighbours of every in-bounds element, so the loop must stay
; scalar: an unrecognized guard is not a missing guard.
; CHECK-LABEL: define amdgpu_kernel void @guarded_staging_not_widened(
; CHECK-NOT: load <4 x float>
define amdgpu_kernel void @guarded_staging_not_widened(ptr addrspace(1) %a,
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
  %in.bounds = icmp slt i32 %gk, %k
  br i1 %in.bounds, label %stage.load, label %stage.merge

stage.load:
  %a.ptr = getelementptr float, ptr addrspace(1) %a, i32 %lm
  %a.ptr.k = getelementptr float, ptr addrspace(1) %a.ptr, i32 %gk
  %val = load float, ptr addrspace(1) %a.ptr.k, align 4
  br label %stage.merge

stage.merge:
  %staged = phi float [ %val, %stage.load ], [ 0.000000e+00, %staging ]
  %lds.row = mul i32 %lm, 32
  %lds.off = add i32 %lds.row, %lk
  %lds.ptr = getelementptr float, ptr addrspace(3) %lds, i32 %lds.off
  store float %staged, ptr addrspace(3) %lds.ptr, align 4
  br label %stage.latch

stage.latch:
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
