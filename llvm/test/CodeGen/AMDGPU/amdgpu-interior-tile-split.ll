; REQUIRES: asserts
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='loop-simplify,lcssa,amdgpu-interior-tile-split' \
; RUN:   -debug-only=amdgpu-interior-tile-split -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='loop-simplify,lcssa,amdgpu-interior-tile-split' \
; RUN:   -verify-each -S %s | FileCheck %s --check-prefix=CFG

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
declare i64 @llvm.smax.i64(i64, i64)
declare i64 @llvm.smin.i64(i64, i64)
declare i32 @llvm.umin.i32(i32, i32)

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
  %lane = and i32 %workitem, 127
  %index = add i32 %x.base, %lane
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

; The dynamic K loop is already in LoopSimplify and LCSSA form.  Its preheader
; has a canonical, CTA-uniform M/N/alignment proof.  The pass must run a
; 32-element prefix and enter the original guarded tail only for K % 32.
define amdgpu_kernel void @canonical_k_loop(ptr addrspace(1) %out, i32 %m,
                                            i32 %n, i32 %k) {
entry:
  br label %dispatch

dispatch:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
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
  br label %k.preheader

k.preheader:
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  %acc = phi i32 [ 0, %k.preheader ], [ %acc.next, %k.latch ]
  br label %k.body

k.body:
  %k.remaining.loop = sub i32 %k, %i
  %k.extent.loop = call i32 @llvm.umin.i32(i32 %k.remaining.loop, i32 32)
  %k.full.loop = icmp uge i32 %k.extent.loop, 32
  br i1 %k.full.loop, label %k.barrier, label %k.skip

k.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %k.latch

k.skip:
  br label %k.latch

k.latch:
  %next = add nuw i32 %i, 32
  %acc.next = add i32 %acc, 1
  %more = icmp ult i32 %next, %k
  br i1 %more, label %k.header, label %k.exit

; A dynamic bound is essential: a fixed trip count cannot exercise both the
; full-prefix and guarded-tail paths.
k.exit:
  %result.lcssa = phi i32 [ %next, %k.latch ]
  %acc.lcssa = phi i32 [ %acc.next, %k.latch ]
  store i32 %result.lcssa, ptr addrspace(1) %out, align 4
  store i32 %acc.lcssa, ptr addrspace(1) %out, align 4
  ret void
}

; This multi-tile K loop has its canonical M/N clamp remainders in the actual
; preheader.  The pass synthesizes its static selector there, then removes the
; exact M/N lane checks and K tile check from the cloned full-K prefix.
define amdgpu_kernel void @canonical_synthesized_k_loop(
    ptr addrspace(1) %out, i32 %m, i32 %n, i32 %k) {
entry:
  br label %dispatch

dispatch:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %workitem = call i32 @llvm.amdgcn.workitem.id.x()
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
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %k.body

k.body:
  ; The lane expression is deliberately affine but not the old exact
  ; workitem-id spelling.  Its largest value is 63, below the 128-wide tile.
  %lane.masked = and i32 %workitem, 31
  %lane.narrow = trunc i32 %lane.masked to i8
  %lane.extended = zext i8 %lane.narrow to i32
  %lane.scaled = mul i32 %lane.extended, 2
  %lane = add i32 %lane.scaled, 1
  %m.index = add i32 %y.base, %lane
  %m.in.bounds = icmp slt i32 %m.index, %m
  br i1 %m.in.bounds, label %k.n.check, label %k.barrier

k.n.check:
  %n.index = add i32 %x.base, %lane
  %n.in.bounds = icmp slt i32 %n.index, %n
  br i1 %n.in.bounds, label %k.k.check, label %k.barrier

k.k.check:
  %k.remaining.loop = sub i32 %k, %i
  %k.extent.loop = call i32 @llvm.smin.i32(i32 %k.remaining.loop, i32 32)
  %k.full.loop = icmp sge i32 %k.extent.loop, 32
  br i1 %k.full.loop, label %k.barrier, label %k.skip

k.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %k.latch

k.skip:
  br label %k.latch

k.latch:
  %next = add nuw i32 %i, 32
  %more = icmp ult i32 %next, %k
  br i1 %more, label %k.header, label %k.exit

k.exit:
  %result.lcssa = phi i32 [ %next, %k.latch ]
  store i32 %result.lcssa, ptr addrspace(1) %out, align 4
  ret void
}

; Clang can lower the M/N clamp setup in the block immediately before the
; otherwise-empty K preheader.  That predecessor dominates the preheader, so
; its values are safe selector inputs without looking through arbitrary CFG.
define amdgpu_kernel void @canonical_predecessor_setup_k_loop(
    ptr addrspace(1) %out, i32 %m, i32 %n, i32 %k) {
entry:
  br label %dispatch

dispatch:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %workitem = call i32 @llvm.amdgcn.workitem.id.x()
  %x.base = shl i32 %workgroup.x, 7
  %y.base = shl i32 %workgroup.y, 7
  %n.remaining = sub i32 %n, %x.base
  %n.nonnegative = call i32 @llvm.smax.i32(i32 %n.remaining, i32 0)
  %n.extent = call i32 @llvm.smin.i32(i32 %n.nonnegative, i32 128)
  %m.remaining = sub i32 %m, %y.base
  %m.nonnegative = call i32 @llvm.smax.i32(i32 %m.remaining, i32 0)
  %m.extent = call i32 @llvm.smin.i32(i32 %m.nonnegative, i32 128)
  %k.remaining = sub i32 %k, 0
  %k.extent = call i32 @llvm.smin.i32(i32 %k.remaining, i32 32)
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  br label %k.preheader

k.preheader:
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %k.body

k.body:
  %lane = and i32 %workitem, 127
  %m.index = add i32 %y.base, %lane
  %m.in.bounds = icmp slt i32 %m.index, %m
  br i1 %m.in.bounds, label %k.n.check, label %k.barrier

k.n.check:
  %n.index = add i32 %x.base, %lane
  %n.in.bounds = icmp slt i32 %n.index, %n
  br i1 %n.in.bounds, label %k.k.check, label %k.barrier

k.k.check:
  %k.remaining.loop = sub i32 %k, %i
  %k.extent.loop = call i32 @llvm.smin.i32(i32 %k.remaining.loop, i32 32)
  %k.full.loop = icmp sge i32 %k.extent.loop, 32
  br i1 %k.full.loop, label %k.barrier, label %k.skip

k.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %k.latch

k.skip:
  br label %k.latch

k.latch:
  %next = add nuw i32 %i, 32
  %more = icmp ult i32 %next, %k
  br i1 %more, label %k.header, label %k.exit

k.exit:
  %result.lcssa = phi i32 [ %next, %k.latch ]
  store i32 %result.lcssa, ptr addrspace(1) %out, align 4
  ret void
}

; The staging loop is nested inside the 32-wide K loop.  Its offset reaches
; the M/N/K guards through affine adds, as in real GEMM IR.  The combined
; select is the short-circuit lowering of K-in-bounds && N-in-bounds.  Only
; the clone's true-path select guard is removed; the original edge stays
; guarded.
define amdgpu_kernel void @canonical_nested_staging_k_loop(
    ptr addrspace(1) %out, i32 %m, i32 %n, i32 %k) {
entry:
  br label %dispatch

dispatch:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %workitem = call i32 @llvm.amdgcn.workitem.id.x()
  %x.base.i32 = shl i32 %workgroup.x, 7
  %y.base.i32 = shl i32 %workgroup.y, 7
  %x.base = sext i32 %x.base.i32 to i64
  %y.base = sext i32 %y.base.i32 to i64
  br label %k.preheader

k.preheader:
  %n64 = sext i32 %n to i64
  %m64 = sext i32 %m to i64
  %n.remaining = sub i64 %n64, %x.base
  %n.nonnegative = call i64 @llvm.smax.i64(i64 %n.remaining, i64 0)
  %n.extent = call i64 @llvm.smin.i64(i64 %n.nonnegative, i64 128)
  %m.remaining = sub i64 %m64, %y.base
  %m.nonnegative = call i64 @llvm.smax.i64(i64 %m.remaining, i64 0)
  %m.extent = call i64 @llvm.smin.i64(i64 %m.nonnegative, i64 128)
  %k.remaining = sub i32 %k, 0
  %k.extent = call i32 @llvm.smin.i32(i32 %k.remaining, i32 32)
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %k.body

k.body:
  br label %stage.header

stage.header:
  %stage.i = phi i32 [ 0, %k.body ], [ %stage.next, %stage.latch ]
  br label %stage.m.check

stage.m.check:
  %stage.i64 = sext i32 %stage.i to i64
  %m.index = add i64 %y.base, %stage.i64
  %m.in.bounds = icmp slt i64 %m.index, %m64
  br i1 %m.in.bounds, label %stage.n.check, label %stage.latch

stage.n.check:
  ; This is the exact gemm_ts_outer quotient/remainder guard shape.  A
  ; nonnegative 7-bit index is divided by a no-wrap constant subtraction;
  ; the quotient selects K and the reconstructed remainder contributes to N.
  %cond215 = add nsw i32 %stage.i, 0
  %idx273.i32 = and i32 %workitem, 127
  %idx273 = zext i32 %idx273.i32 to i64
  %sub270 = sub nsw i64 5, 1
  %div280 = udiv i64 %idx273, %sub270
  %conv283 = trunc i64 %div280 to i32
  %mul284 = mul nuw i64 %div280, %sub270
  %sub285 = sub nuw i64 %idx273, %mul284
  %conv286 = trunc i64 %sub285 to i32
  %add288 = add nsw i32 %cond215, %conv286
  %conv293 = sext i32 %add288 to i64
  %add294 = add nsw i64 %conv293, %x.base
  %n.in.bounds = icmp slt i64 %add294, %n64
  %add291 = add nsw i32 %conv283, %i
  %cmp296 = icmp slt i32 %add291, %k
  %cmp298 = icmp slt i64 %add294, %n64
  %or.cond = select i1 %cmp296, i1 %cmp298, i1 false
  br i1 %or.cond, label %stage.k.check, label %stage.latch

stage.k.check:
  br label %stage.barrier

stage.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %stage.latch

stage.latch:
  %stage.next = add nuw i32 %stage.i, 1
  %stage.more = icmp ult i32 %stage.next, 32
  br i1 %stage.more, label %stage.header, label %k.latch

k.latch:
  %next = add nuw i32 %i, 32
  %more = icmp ult i32 %next, %k
  br i1 %more, label %k.header, label %k.exit

k.exit:
  %result.lcssa = phi i32 [ %next, %k.latch ]
  store i32 %result.lcssa, ptr addrspace(1) %out, align 4
  ret void
}

; This is the real-GEMM N-edge dispatch spelling.  The false edge fallback
; contains quotient/remainder work, but the split must prove and remove only
; cmp267: on the synthesized full-N and same-alignment prefix, cond215 is
; exactly 128, so cmp267 is false.  The quotient/remainder guard is not a
; target of the proof.
define amdgpu_kernel void @canonical_n_edge_fallback_k_loop(
    ptr addrspace(1) %out, i32 %m, i32 %n, i32 %k) {
entry:
  br label %dispatch

dispatch:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %workitem = call i32 @llvm.amdgcn.workitem.id.x()
  %x.base.i32 = shl i32 %workgroup.x, 7
  %y.base.i32 = shl i32 %workgroup.y, 7
  %x.base = sext i32 %x.base.i32 to i64
  %y.base = sext i32 %y.base.i32 to i64
  br label %k.preheader

k.preheader:
  %n64 = sext i32 %n to i64
  %m64 = sext i32 %m to i64
  %n.remaining = sub i64 %n64, %x.base
  %n.nonnegative = call i64 @llvm.smax.i64(i64 %n.remaining, i64 0)
  %n.extent = call i64 @llvm.smin.i64(i64 %n.nonnegative, i64 128)
  %m.remaining = sub i64 %m64, %y.base
  %m.nonnegative = call i64 @llvm.smax.i64(i64 %m.remaining, i64 0)
  %m.extent = call i64 @llvm.smin.i64(i64 %m.nonnegative, i64 128)
  %k.remaining = sub i32 %k, 0
  %k.extent = call i32 @llvm.smin.i32(i32 %k.remaining, i32 32)
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %k.body

k.body:
  %n.extent.i32 = trunc i64 %n.extent to i32
  %n.masked = and i32 %n.extent.i32, 252
  %cond215 = select i1 %out.aligned, i32 %n.masked, i32 0
  %cmp267 = icmp ult i32 %cond215, 128
  br i1 %cmp267, label %if.then268, label %if.end312

if.then268:
  br label %for.body278

for.body278:
  ; This fallback-only quotient/remainder guard must not be recognized.
  %lane = and i32 %workitem, 127
  %quotient = udiv i32 %lane, 4
  %remainder = urem i32 %lane, 4
  %remainder.in.range = icmp ult i32 %remainder, 4
  br i1 %remainder.in.range, label %k.barrier, label %k.barrier

if.end312:
  br label %k.barrier

k.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %k.latch

k.latch:
  %next = add nuw i32 %i, 32
  %more = icmp ult i32 %next, %k
  br i1 %more, label %k.header, label %k.exit

k.exit:
  %result.lcssa = phi i32 [ %next, %k.latch ]
  store i32 %result.lcssa, ptr addrspace(1) %out, align 4
  ret void
}

; Regression for the staging-only POC: only k.body and its pre-barrier
; fallback graph may acquire `.interior` blocks.  The compute block and the
; outer latch remain single shared blocks after the common barrier.
define amdgpu_kernel void @staging_only_outer_k_loop(
    ptr addrspace(1) %out, i32 %m, i32 %n, i32 %k) {
entry:
  br label %dispatch

dispatch:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %x.base.i32 = shl i32 %workgroup.x, 7
  %y.base.i32 = shl i32 %workgroup.y, 7
  %x.base = sext i32 %x.base.i32 to i64
  %y.base = sext i32 %y.base.i32 to i64
  br label %k.preheader

k.preheader:
  %n64 = sext i32 %n to i64
  %m64 = sext i32 %m to i64
  %n.remaining = sub i64 %n64, %x.base
  %n.nonnegative = call i64 @llvm.smax.i64(i64 %n.remaining, i64 0)
  %n.extent = call i64 @llvm.smin.i64(i64 %n.nonnegative, i64 128)
  %m.remaining = sub i64 %m64, %y.base
  %m.nonnegative = call i64 @llvm.smax.i64(i64 %m.remaining, i64 0)
  %m.extent = call i64 @llvm.smin.i64(i64 %m.nonnegative, i64 128)
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %k.body

k.body:
  %n.extent.i32 = trunc i64 %n.extent to i32
  %n.masked = and i32 %n.extent.i32, 252
  %cond215 = select i1 %out.aligned, i32 %n.masked, i32 0
  %cmp267 = icmp ult i32 %cond215, 128
  br i1 %cmp267, label %edge.fallback, label %stage.full

edge.fallback:
  %edge.work = add i32 %i, 1
  br label %k.barrier

stage.full:
  br label %stage.inner.header

stage.inner.header:
  %stage.j = phi i32 [ 0, %stage.full ], [ %stage.j.next, %stage.inner.latch ]
  br label %stage.inner.body

stage.inner.body:
  %stage.work = add i32 %i, %stage.j
  br label %stage.inner.latch

stage.inner.latch:
  %stage.j.next = add nuw i32 %stage.j, 1
  %stage.more = icmp ult i32 %stage.j.next, 2
  br i1 %stage.more, label %stage.inner.header, label %k.barrier

k.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %compute

compute:
  %compute.value = add i32 %i, 3
  store i32 %compute.value, ptr addrspace(1) %out, align 4
  br label %k.latch

k.latch:
  %next = add nuw i32 %i, 32
  %more = icmp ult i32 %next, %k
  br i1 %more, label %k.header, label %k.exit

k.exit:
  %result.lcssa = phi i32 [ %next, %k.latch ]
  store i32 %result.lcssa, ptr addrspace(1) %out, align 4
  ret void
}

; The equivalent one-K-tile loop is structurally proven to execute once.
define amdgpu_kernel void @canonical_synthesized_one_k_loop(
    ptr addrspace(1) %out, i32 %m, i32 %n, i32 %k) {
entry:
  br label %dispatch

dispatch:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %x.base = shl i32 %workgroup.x, 7
  %y.base = shl i32 %workgroup.y, 7
  %n.remaining = sub i32 %n, %x.base
  %n.nonnegative = call i32 @llvm.smax.i32(i32 %n.remaining, i32 0)
  %n.extent = call i32 @llvm.smin.i32(i32 %n.nonnegative, i32 128)
  %m.remaining = sub i32 %m, %y.base
  %m.nonnegative = call i32 @llvm.smax.i32(i32 %m.remaining, i32 0)
  %m.extent = call i32 @llvm.smin.i32(i32 %m.nonnegative, i32 128)
  %k.remaining = sub i32 %k, 0
  %k.extent = call i32 @llvm.smin.i32(i32 %k.remaining, i32 32)
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  br label %k.preheader

k.preheader:
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %k.body

k.body:
  call void @llvm.amdgcn.s.barrier()
  %next = add nuw i32 %i, 1
  %done = icmp eq i32 %next, 1
  br i1 %done, label %k.exit, label %k.latch

k.latch:
  br label %k.header

k.exit:
  %result.lcssa = phi i32 [ %next, %k.body ]
  store i32 %result.lcssa, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: Potential interior-tile boundary check in candidate:
; CHECK: Cloned nested-loop staging region in staging_only_outer_k_loop; outer K latch and shared barrier were retained

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

; CFG-LABEL: define amdgpu_kernel void @staging_only_outer_k_loop(
; The distinct split-edge dispatch has different fast and fallback targets:
; fast is the cloned staging entry and fallback is the original entry.
; CFG: br i1 %interior.staging.full, label %k.body.interior, label %k.body
; CFG-NOT: br i1 %interior.staging.full, label %[[SELF:[^, ]+]], label %[[SELF]]
; CFG-LABEL: k.body:
; CFG: %cond215 = select i1 %out.aligned, i32 %n.masked, i32 0
; CFG: br i1 %cmp267, label %edge.fallback, label %stage.full
; CFG-LABEL: k.barrier:
; CFG-COUNT-1: call void @llvm.amdgcn.s.barrier()
; CFG: br label %compute
; CFG-LABEL: compute:
; CFG: br label %k.latch
; CFG-LABEL: k.latch:
; CFG: br i1 %more, label %k.header, label %k.exit
; CFG-LABEL: k.body.interior:
; CFG: %cond215.interior = select i1 %out.aligned, i32 %n.masked.interior, i32 0
; CFG: br label %stage.full.interior
; CFG-LABEL: stage.full.interior:
; CFG: br label %stage.inner.header.interior
; CFG-LABEL: stage.inner.header.interior:
; CFG: %stage.j.interior = phi i32 [ 0, %stage.full.interior ], [ %stage.j.next.interior, %stage.inner.latch.interior ]
; CFG: br label %stage.inner.body.interior
; CFG-LABEL: stage.inner.body.interior:
; CFG: br label %stage.inner.latch.interior
; CFG-LABEL: stage.inner.latch.interior:
; CFG: br i1 %stage.more.interior, label %stage.inner.header.interior, label %k.barrier
; The clone contains no barrier; both paths converge at the original one.
; CFG-NOT: k.barrier.interior
; CFG-NOT: compute.interior
; CFG-NOT: k.latch.interior
; CFG-LABEL: define amdgpu_kernel void @canonical_synthesized_one_k_loop(
